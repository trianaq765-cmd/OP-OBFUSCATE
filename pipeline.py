# ============================================
# File: pipeline.py (SIMPLE VERSION)
# Obfuscation Pipeline for Replit/Discord Bot
# ============================================

import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import modules
from lua_parser import parse_bytecode_file, LuaChunk, ControlFlowAnalyzer
from lua_transformer import BytecodeTransformer
from lua_vm_generator import LuaVMGenerator, BytecodeSerializer
from lua_encryption import EncryptionManager
from lua_antitamper import AntiTamperGenerator
from config_manager import ObfuscatorConfig

# Import Luraph style
try:
    from luraph_style import apply_luraph_style
    LURAPH_AVAILABLE = True
except ImportError:
    LURAPH_AVAILABLE = False

# ============================================
# Pipeline Result
# ============================================

@dataclass
class PipelineResult:
    """Result dari pipeline execution"""
    success: bool = False
    output_path: Optional[str] = None
    total_time: float = 0.0
    input_size: int = 0
    output_size: int = 0
    size_ratio: float = 0.0
    stats: Dict[str, Any] = None
    errors: list = None
    warnings: list = None
    
    def __post_init__(self):
        if self.stats is None:
            self.stats = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

# ============================================
# Main Pipeline
# ============================================

class ObfuscationPipeline:
    """Simple obfuscation pipeline"""
    
    def __init__(self, config: ObfuscatorConfig):
        self.config = config
        
        # State
        self.chunk: Optional[LuaChunk] = None
        self.transformed_chunk: Optional[LuaChunk] = None
        self.vm_code: str = ""
        self.protection_code: str = ""
        self.final_code: str = ""
        self.metadata: Dict[str, Any] = {}
    
    def process(self, input_path: str, output_path: str) -> PipelineResult:
        """Execute pipeline"""
        result = PipelineResult()
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input not found: {input_path}")
            
            result.input_size = os.path.getsize(input_path)
            
            # Stage 1: Parse
            print("[1/7] Parsing bytecode...")
            self.chunk = parse_bytecode_file(input_path)
            print(f"  ✓ Parsed {len(self.chunk.all_functions)} functions")
            
            # Stage 2: Analyze
            print("[2/7] Analyzing...")
            for func in self.chunk.all_functions:
                analyzer = ControlFlowAnalyzer(func)
                analyzer.analyze()
            print("  ✓ Analysis complete")
            
            # Stage 3: Transform
            print("[3/7] Transforming...")
            self.config.transform.generate_random_values()
            transformer = BytecodeTransformer(self.config.transform)
            self.transformed_chunk, metadata = transformer.transform(self.chunk)
            self.metadata['transform'] = metadata
            print("  ✓ Transformed")
            
            # Stage 4: Encrypt
            print("[4/7] Encrypting...")
            self.config.encryption.generate_keys()
            enc_manager = EncryptionManager(self.config.encryption)
            print("  ✓ Encrypted")
            
            # Stage 5: Generate VM
            print("[5/7] Generating VM...")
            generator = LuaVMGenerator(self.config.transform, self.config.vm)
            self.vm_code = generator.generate_vm()
            print(f"  ✓ VM generated ({len(self.vm_code)} chars)")
            
            # Stage 6: Anti-tamper
            print("[6/7] Adding protection...")
            self.config.antitamper.generate_keys()
            antitamper = AntiTamperGenerator(self.config.antitamper)
            self.protection_code = antitamper.generate_all_protections()
            print(f"  ✓ Protection added ({len(self.protection_code)} chars)")
            
            # Stage 7: Combine
            print("[7/7] Combining...")
            self._stage_combine()
            print(f"  ✓ Combined ({len(self.final_code)} chars)")
            
            # Write output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.final_code)
            
            # Success
            result.success = True
            result.output_path = output_path
            result.output_size = len(self.final_code.encode('utf-8'))
            result.size_ratio = result.output_size / result.input_size if result.input_size > 0 else 0
            result.stats = self._collect_stats()
            
        except Exception as e:
            result.errors.append(str(e))
            result.success = False
            print(f"✗ Error: {e}")
        
        result.total_time = time.time() - start_time
        return result
    
    def _stage_combine(self):
        """Stage 7: Combine all components"""
        
        # Serialize bytecode
        serializer = BytecodeSerializer(None)
        bytecode_str = serializer.serialize_function(self.transformed_chunk.main_function)
        
        # Check if Luraph style requested
        if LURAPH_AVAILABLE and self.config.name == 'luraph':
            print("  → Using Luraph-style output")
            
            # Extract constants
            constants = [c.value for c in self.transformed_chunk.main_function.constants]
            
            # Apply Luraph styling
            self.final_code = apply_luraph_style(
                self.vm_code, 
                constants,
                seed=self.config.transform.opcode_seed
            )
        else:
            # Normal combining
            parts = []
            
            # Header
            if self.config.add_header:
                parts.append(self._generate_header())
            
            # Protection layer
            parts.append(self.protection_code)
            
            # VM
            parts.append(self.vm_code)
            
            # Bytecode
            parts.append(f"-- Bytecode\nlocal bytecode = {bytecode_str}")
            
            # Execution
            parts.append(self._generate_execution_code())
            
            # Combine
            self.final_code = "\n\n".join(parts)
        
        # Minify if requested
        if self.config.minify:
            self.final_code = self._minify_code(self.final_code)
    
    def _generate_header(self) -> str:
        """Generate file header"""
        if self.config.header_text:
            return f"-- {self.config.header_text}"
        
        return f"""-- Protected by Lua Obfuscator
-- Configuration: {self.config.name}
-- DO NOT MODIFY
"""
    
    def _generate_execution_code(self) -> str:
        """Generate execution wrapper"""
        return """-- Execute
local _wrapper = _WRAP or wrap
if _wrapper then
    local vm = _wrapper(bytecode)
    if vm then
        return vm({}, _G or _ENV)()
    end
end
error("VM initialization failed")"""
    
    def _minify_code(self, code: str) -> str:
        """Basic minification"""
        lines = []
        for line in code.split('\n'):
            if '--' in line and not line.strip().startswith('--[['):
                line = line[:line.index('--')]
            line = line.strip()
            if line and not line.startswith('--'):
                lines.append(line)
        
        result = ' '.join(lines)
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        return result
    
    def _collect_stats(self) -> Dict[str, Any]:
        """Collect statistics"""
        if not self.chunk:
            return {}
        
        return {
            'input': {
                'functions': len(self.chunk.all_functions),
                'instructions': self.chunk.stats.get('total_instructions', 0),
                'constants': self.chunk.stats.get('total_constants', 0),
            },
            'output': {
                'code_size': len(self.final_code),
                'vm_size': len(self.vm_code),
                'protection_size': len(self.protection_code),
            },
          }
