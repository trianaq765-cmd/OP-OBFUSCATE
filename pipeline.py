# ============================================
# File: pipeline.py
# Complete Obfuscation Pipeline
# Part 3: Pipeline Integration
# ============================================

import os
import sys
import time
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
import json

# Import semua komponen
from lua_parser import (
    parse_bytecode, parse_bytecode_file,
    LuaChunk, AdvancedBytecodeParser,
    ControlFlowAnalyzer, disassemble
)
from lua_transformer import (
    TransformConfig, BytecodeTransformer
)
from lua_vm_generator import (
    VMConfig, LuaVMGenerator,
    BytecodeSerializer
)
from lua_encryption import (
    EncryptionConfig, EncryptionManager
)
from lua_antitamper import (
    AntiTamperConfig, AntiTamperGenerator
)
from config_manager import (
    ConfigManager, ObfuscatorConfig
)

# ============================================
# Pipeline Stages
# ============================================

class PipelineStage(IntEnum):
    """Tahapan dalam pipeline obfuscation"""
    PARSE = auto()
    ANALYZE = auto()
    TRANSFORM = auto()
    ENCRYPT = auto()
    GENERATE_VM = auto()
    ADD_PROTECTION = auto()
    COMBINE = auto()
    VALIDATE = auto()
    WRITE = auto()

# ============================================
# Pipeline Result
# ============================================

@dataclass
class PipelineResult:
    """Result dari pipeline execution"""
    success: bool = False
    output_path: Optional[str] = None
    
    # Timing info
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    # Size info
    input_size: int = 0
    output_size: int = 0
    size_ratio: float = 0.0
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    chunk_info: Dict[str, Any] = field(default_factory=dict)
    transform_info: Dict[str, Any] = field(default_factory=dict)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

# ============================================
# Progress Callback
# ============================================

class ProgressCallback:
    """Base class untuk progress tracking"""
    
    def on_stage_start(self, stage: PipelineStage, stage_name: str):
        """Called when stage starts"""
        pass
    
    def on_stage_complete(self, stage: PipelineStage, stage_name: str, elapsed: float):
        """Called when stage completes"""
        pass
    
    def on_progress(self, stage: PipelineStage, progress: float, message: str = ""):
        """Called during stage execution"""
        pass
    
    def on_error(self, stage: PipelineStage, error: str):
        """Called on error"""
        pass
    
    def on_warning(self, stage: PipelineStage, warning: str):
        """Called on warning"""
        pass

class ConsoleProgress(ProgressCallback):
    """Console-based progress display"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_stage = None
    
    def on_stage_start(self, stage: PipelineStage, stage_name: str):
        self.current_stage = stage
        print(f"[{stage.value}/9] {stage_name}...", end='', flush=True)
    
    def on_stage_complete(self, stage: PipelineStage, stage_name: str, elapsed: float):
        print(f" ✓ ({elapsed:.2f}s)")
    
    def on_progress(self, stage: PipelineStage, progress: float, message: str = ""):
        if self.verbose and message:
            print(f"      {message}")
    
    def on_error(self, stage: PipelineStage, error: str):
        print(f"\n      ✗ Error: {error}")
    
    def on_warning(self, stage: PipelineStage, warning: str):
        if self.verbose:
            print(f"      ⚠ Warning: {warning}")

# ============================================
# Cache Manager
# ============================================

class CacheManager:
    """Manages caching of intermediate results"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enabled = True
    
    def get_cache_key(self, input_path: str, config_hash: str) -> str:
        """Generate cache key"""
        combined = f"{input_path}:{config_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def has_cached(self, key: str) -> bool:
        """Check if cached result exists"""
        if not self.enabled:
            return False
        cache_file = self.cache_dir / f"{key}.cache"
        return cache_file.exists()
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if not self.enabled:
            return None
        
        cache_file = self.cache_dir / f"{key}.cache"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def set_cached(self, key: str, data: Any):
        """Store result in cache"""
        if not self.enabled:
            return
        
        cache_file = self.cache_dir / f"{key}.cache"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def clear_cache(self):
        """Clear all cached results"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()

# ============================================
# Main Pipeline
# ============================================

class ObfuscationPipeline:
    """Main obfuscation pipeline"""
    
    def __init__(self, config: ObfuscatorConfig,
                 progress_callback: Optional[ProgressCallback] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.progress = progress_callback or ProgressCallback()
        self.logger = logger or self._create_logger()
        
        # Components
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        
        # State
        self.current_stage = None
        self.chunk: Optional[LuaChunk] = None
        self.transformed_chunk: Optional[LuaChunk] = None
        self.vm_code: str = ""
        self.protection_code: str = ""
        self.final_code: str = ""
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
    
    def _create_logger(self) -> logging.Logger:
        """Create default logger"""
        logger = logging.getLogger('ObfuscationPipeline')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # File handler
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    # ========================================
    # Main Pipeline Execution
    # ========================================
    
    def process(self, input_path: str, output_path: str) -> PipelineResult:
        """Execute complete pipeline"""
        result = PipelineResult()
        start_time = time.time()
        
        self.logger.info(f"Starting obfuscation pipeline")
        self.logger.info(f"Input: {input_path}")
        self.logger.info(f"Output: {output_path}")
        
        try:
            # Validate input
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            result.input_size = os.path.getsize(input_path)
            
            # Execute stages
            self._execute_stage(PipelineStage.PARSE, "Parsing bytecode", 
                              lambda: self._stage_parse(input_path), result)
            
            self._execute_stage(PipelineStage.ANALYZE, "Analyzing code",
                              lambda: self._stage_analyze(), result)
            
            self._execute_stage(PipelineStage.TRANSFORM, "Transforming bytecode",
                              lambda: self._stage_transform(), result)
            
            self._execute_stage(PipelineStage.ENCRYPT, "Encrypting data",
                              lambda: self._stage_encrypt(), result)
            
            self._execute_stage(PipelineStage.GENERATE_VM, "Generating VM",
                              lambda: self._stage_generate_vm(), result)
            
            self._execute_stage(PipelineStage.ADD_PROTECTION, "Adding protection",
                              lambda: self._stage_add_protection(), result)
            
            self._execute_stage(PipelineStage.COMBINE, "Combining components",
                              lambda: self._stage_combine(), result)
            
            self._execute_stage(PipelineStage.VALIDATE, "Validating output",
                              lambda: self._stage_validate(), result)
            
            self._execute_stage(PipelineStage.WRITE, "Writing output",
                              lambda: self._stage_write(output_path), result)
            
            # Success
            result.success = True
            result.output_path = output_path
            result.output_size = len(self.final_code.encode('utf-8'))
            result.size_ratio = result.output_size / result.input_size if result.input_size > 0 else 0
            
            self.logger.info(f"Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            result.errors.append(str(e))
            result.success = False
        
        result.total_time = time.time() - start_time
        
        # Collect statistics
        result.stats = self._collect_statistics()
        result.chunk_info = self._collect_chunk_info()
        result.transform_info = self.metadata.copy()
        
        return result
    
    def _execute_stage(self, stage: PipelineStage, name: str,
                      func: Callable, result: PipelineResult):
        """Execute a single pipeline stage"""
        self.current_stage = stage
        self.progress.on_stage_start(stage, name)
        
        stage_start = time.time()
        
        try:
            func()
            elapsed = time.time() - stage_start
            result.stage_times[name] = elapsed
            self.progress.on_stage_complete(stage, name, elapsed)
            
        except Exception as e:
            self.logger.error(f"Stage {name} failed: {e}", exc_info=True)
            self.progress.on_error(stage, str(e))
            raise
    
    # ========================================
    # Pipeline Stages
    # ========================================
    
    def _stage_parse(self, input_path: str):
        """Stage 1: Parse bytecode"""
        self.logger.info(f"Parsing: {input_path}")
        
        # Read file
        with open(input_path, 'rb') as f:
            bytecode = f.read()
        
        # Parse
        parser = AdvancedBytecodeParser(bytecode)
        self.chunk = parser.parse()
        
        self.logger.info(f"Parsed {len(self.chunk.all_functions)} functions")
        self.progress.on_progress(
            PipelineStage.PARSE,
            1.0,
            f"Parsed {len(self.chunk.all_functions)} functions, "
            f"{self.chunk.stats.get('total_instructions', 0)} instructions"
        )
    
    def _stage_analyze(self):
        """Stage 2: Analyze code structure"""
        self.logger.info("Analyzing code structure")
        
        # Analyze each function
        for i, func in enumerate(self.chunk.all_functions):
            self.progress.on_progress(
                PipelineStage.ANALYZE,
                (i + 1) / len(self.chunk.all_functions),
                f"Analyzing function {i + 1}/{len(self.chunk.all_functions)}"
            )
            
            # Control flow analysis
            analyzer = ControlFlowAnalyzer(func)
            analyzer.analyze()
        
        self.logger.info("Analysis complete")
    
    def _stage_transform(self):
        """Stage 3: Transform bytecode"""
        self.logger.info("Transforming bytecode")
        
        # Generate keys if needed
        self.config.transform.generate_random_values()
        
        # Create transformer
        transformer = BytecodeTransformer(self.config.transform)
        
        # Transform
        self.transformed_chunk, metadata = transformer.transform(self.chunk)
        self.metadata['transform'] = metadata
        
        self.logger.info(f"Applied transformations")
        self.progress.on_progress(
            PipelineStage.TRANSFORM,
            1.0,
            f"Transformed {len(self.transformed_chunk.all_functions)} functions"
        )
    
    def _stage_encrypt(self):
        """Stage 4: Encrypt data"""
        if not (self.config.encryption.encrypt_strings or 
                self.config.encryption.encrypt_numbers or
                self.config.encryption.encrypt_bytecode):
            self.logger.info("Encryption disabled, skipping")
            return
        
        self.logger.info("Encrypting data")
        
        # Generate encryption keys
        self.config.encryption.generate_keys()
        
        # Create encryption manager
        enc_manager = EncryptionManager(self.config.encryption)
        
        # Encrypt constants
        for func in self.transformed_chunk.all_functions:
            encrypted_count = 0
            
            for const in func.constants:
                if const.type == 4 and self.config.encryption.encrypt_strings:
                    # String constant - mark as encrypted
                    const.is_encrypted = True
                    encrypted_count += 1
                elif const.type == 3 and self.config.encryption.encrypt_numbers:
                    # Number constant - mark as encrypted
                    const.is_encrypted = True
                    encrypted_count += 1
            
            if encrypted_count > 0:
                self.logger.debug(f"Encrypted {encrypted_count} constants in function")
        
        self.metadata['encryption'] = {
            'strings_encrypted': self.config.encryption.encrypt_strings,
            'numbers_encrypted': self.config.encryption.encrypt_numbers,
            'algorithm': self.config.encryption.algorithm.name,
        }
        
        self.progress.on_progress(
            PipelineStage.ENCRYPT,
            1.0,
            "Data encryption complete"
        )
    
    def _stage_generate_vm(self):
        """Stage 5: Generate custom VM"""
        self.logger.info("Generating custom VM")
        
        # Create VM generator
        generator = LuaVMGenerator(
            self.config.transform,
            self.config.vm
        )
        
        # Generate VM code
        self.vm_code = generator.generate_vm()
        
        self.logger.info(f"Generated VM: {len(self.vm_code)} characters")
        self.progress.on_progress(
            PipelineStage.GENERATE_VM,
            1.0,
            f"VM generated ({len(self.vm_code)} chars)"
        )
    
    def _stage_add_protection(self):
        """Stage 6: Add anti-tamper protection"""
        self.logger.info("Adding anti-tamper protection")
        
        # Generate keys
        self.config.antitamper.generate_keys()
        
        # Create protection generator
        antitamper = AntiTamperGenerator(self.config.antitamper)
        
        # Generate protection code
        self.protection_code = antitamper.generate_all_protections()
        
        # Get summary
        summary = antitamper.generate_protection_summary()
        enabled_count = sum(1 for v in summary.values() if v)
        
        self.logger.info(f"Added {enabled_count} protection mechanisms")
        self.progress.on_progress(
            PipelineStage.ADD_PROTECTION,
            1.0,
            f"{enabled_count} protections added ({len(self.protection_code)} chars)"
        )
    
    def _stage_combine(self):
        """Stage 7: Combine all components"""
        self.logger.info("Combining components")
        
        # Serialize bytecode
        serializer = BytecodeSerializer(None)
        bytecode_str = serializer.serialize_function(self.transformed_chunk.main_function)
        
        # Build final code
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
        
        self.logger.info(f"Combined output: {len(self.final_code)} characters")
        self.progress.on_progress(
            PipelineStage.COMBINE,
            1.0,
            f"Combined {len(parts)} components"
        )
    
    def _stage_validate(self):
        """Stage 8: Validate output"""
        self.logger.info("Validating output")
        
        # Basic validation
        if not self.final_code:
            raise ValueError("Output code is empty")
        
        if len(self.final_code) < 100:
            self.progress.on_warning(
                PipelineStage.VALIDATE,
                "Output seems unusually small"
            )
        
        # Check for critical components
        required_patterns = ['function', 'return', 'local']
        missing = [p for p in required_patterns if p not in self.final_code]
        
        if missing:
            self.progress.on_warning(
                PipelineStage.VALIDATE,
                f"Missing expected patterns: {missing}"
            )
        
        self.logger.info("Validation complete")
    
    def _stage_write(self, output_path: str):
        """Stage 9: Write output file"""
        self.logger.info(f"Writing output to: {output_path}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.final_code)
        
        self.logger.info(f"Wrote {len(self.final_code)} characters")
        self.progress.on_progress(
            PipelineStage.WRITE,
            1.0,
            f"Written to {output_path}"
        )
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _generate_header(self) -> str:
        """Generate file header"""
        if self.config.header_text:
            return f"-- {self.config.header_text}"
        
        return f"""--[[
    Protected by Lua Obfuscator
    Configuration: {self.config.name}
    Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
    
    DO NOT MODIFY THIS FILE
    Unauthorized reverse engineering is prohibited
]]"""
    
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
        """Basic code minification"""
        lines = []
        
        for line in code.split('\n'):
            # Remove comments
            if '--' in line and not line.strip().startswith('--[['):
                line = line[:line.index('--')]
            
            # Strip whitespace
            line = line.strip()
            
            if line and not line.startswith('--'):
                lines.append(line)
        
        # Join with minimal spacing
        result = ' '.join(lines)
        
        # Reduce multiple spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        return result
    
    def _collect_statistics(self) -> Dict[str, Any]:
        """Collect pipeline statistics"""
        if not self.chunk:
            return {}
        
        return {
            'input': {
                'functions': len(self.chunk.all_functions),
                'instructions': self.chunk.stats.get('total_instructions', 0),
                'constants': self.chunk.stats.get('total_constants', 0),
                'strings': self.chunk.stats.get('total_strings', 0),
            },
            'output': {
                'code_size': len(self.final_code),
                'vm_size': len(self.vm_code),
                'protection_size': len(self.protection_code),
            },
        }
    
    def _collect_chunk_info(self) -> Dict[str, Any]:
        """Collect chunk information"""
        if not self.chunk:
            return {}
        
        return {
            'version': hex(self.chunk.header.version),
            'format': self.chunk.header.format_type.name,
            'endianness': 'little' if self.chunk.header.is_little_endian else 'big',
            'checksum': self.chunk.checksum,
        }

# ============================================
# Batch Processing
# ============================================

class BatchProcessor:
    """Process multiple files in batch"""
    
    def __init__(self, config: ObfuscatorConfig,
                 progress_callback: Optional[ProgressCallback] = None):
        self.config = config
        self.progress = progress_callback or ConsoleProgress()
    
    def process_batch(self, file_pairs: List[Tuple[str, str]]) -> List[PipelineResult]:
        """Process multiple files"""
        results = []
        total = len(file_pairs)
        
        print(f"\nProcessing {total} files...\n")
        
        for i, (input_path, output_path) in enumerate(file_pairs):
            print(f"[{i+1}/{total}] {input_path}")
            
            # Create pipeline
            pipeline = ObfuscationPipeline(
                self.config,
                progress_callback=self.progress
            )
            
            # Process
            result = pipeline.process(input_path, output_path)
            results.append(result)
            
            if result.success:
                print(f"  ✓ Success: {output_path}")
            else:
                print(f"  ✗ Failed: {result.errors[0] if result.errors else 'Unknown error'}")
            
            print()
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[PipelineResult]):
        """Print batch processing summary"""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        total_time = sum(r.total_time for r in results)
        total_input = sum(r.input_size for r in results)
        total_output = sum(r.output_size for r in results if r.success)
        
        print("=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files:     {len(results)}")
        print(f"Successful:      {successful}")
        print(f"Failed:          {failed}")
        print(f"Total time:      {total_time:.2f}s")
        print(f"Average time:    {total_time/len(results):.2f}s per file")
        print(f"Total input:     {total_input:,} bytes")
        print(f"Total output:    {total_output:,} bytes")
        if total_input > 0:
            print(f"Size ratio:      {total_output/total_input:.2f}x")
        print("=" * 60)

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    print("=== Obfuscation Pipeline Test ===\n")
    
    # Create test config
    config_manager = ConfigManager()
    config = config_manager.get_preset('medium')
    config.verbose = True
    
    print(f"Using preset: {config.name}")
    print(f"Description: {config.description}")
    print()
    
    # Validate config
    issues = config_manager.validate_config(config)
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Configuration is valid")
    print()
    
    # Create pipeline
    progress = ConsoleProgress(verbose=True)
    pipeline = ObfuscationPipeline(config, progress_callback=progress)
    
    print("Pipeline components initialized:")
    print(f"  Config: {config.name}")
    print(f"  Transform: shuffle={config.transform.shuffle_opcodes}")
    print(f"  Encryption: strings={config.encryption.encrypt_strings}")
    print(f"  Anti-tamper: debug_detect={config.antitamper.enable_debugger_detect}")
    print()
    
    print("✅ Pipeline ready for processing!")
    print("\nTo process a file:")
    print("  result = pipeline.process('input.luac', 'output.lua')")
