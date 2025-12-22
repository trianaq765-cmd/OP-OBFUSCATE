# ============================================
# File: pipeline.py (FIXED - Support Source Code)
# Obfuscation Pipeline for Lua Source Code
# ============================================

import os
import time
import re
import random
import string
import base64
import zlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import config
from config_manager import ObfuscatorConfig

# Optional imports - akan dicek ketersediaannya
try:
    from lua_parser import parse_bytecode_file, LuaChunk, ControlFlowAnalyzer
    HAS_BYTECODE_PARSER = True
except ImportError:
    HAS_BYTECODE_PARSER = False

try:
    from lua_transformer import BytecodeTransformer
    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False

try:
    from lua_vm_generator import LuaVMGenerator, BytecodeSerializer
    HAS_VM_GENERATOR = True
except ImportError:
    HAS_VM_GENERATOR = False

try:
    from lua_encryption import EncryptionManager
    HAS_ENCRYPTION = True
except ImportError:
    HAS_ENCRYPTION = False

try:
    from lua_antitamper import AntiTamperGenerator
    HAS_ANTITAMPER = True
except ImportError:
    HAS_ANTITAMPER = False

try:
    from luraph_style import apply_luraph_style
    HAS_LURAPH_STYLE = True
except ImportError:
    HAS_LURAPH_STYLE = False

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
# Source Code Obfuscator (Luraph-style)
# ============================================

class LuaSourceObfuscator:
    """Obfuscator untuk Lua source code (bukan bytecode)"""
    
    def __init__(self, config: ObfuscatorConfig):
        self.config = config
        self.string_map: Dict[str, str] = {}
        self.var_map: Dict[str, str] = {}
        self.func_map: Dict[str, str] = {}
        self.seed = random.randint(1, 999999)
        
        # Lua keywords - jangan di-rename
        self.keywords = {
            'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for',
            'function', 'goto', 'if', 'in', 'local', 'nil', 'not', 'or',
            'repeat', 'return', 'then', 'true', 'until', 'while'
        }
        
        # Built-in globals - jangan di-rename
        self.builtins = {
            'print', 'pairs', 'ipairs', 'next', 'type', 'tostring', 'tonumber',
            'string', 'table', 'math', 'io', 'os', 'debug', 'coroutine',
            'setmetatable', 'getmetatable', 'rawget', 'rawset', 'rawequal',
            'select', 'unpack', 'pack', 'pcall', 'xpcall', 'error', 'assert',
            'loadstring', 'load', 'loadfile', 'dofile', 'require', 'module',
            '_G', '_ENV', '_VERSION', 'arg', 'self',
            # Roblox globals
            'game', 'workspace', 'script', 'Instance', 'Vector3', 'CFrame',
            'Color3', 'BrickColor', 'Enum', 'UDim2', 'UDim', 'Ray', 'Region3',
            'TweenInfo', 'NumberSequence', 'ColorSequence', 'Random', 'task',
            'wait', 'delay', 'spawn', 'tick', 'time', 'typeof', 'warn'
        }
    
    def obfuscate(self, source: str) -> str:
        """Main obfuscation method"""
        
        # Step 1: Extract and encrypt strings
        source, strings = self._extract_strings(source)
        
        # Step 2: Rename variables
        source = self._rename_variables(source)
        
        # Step 3: Encrypt numbers
        source = self._encrypt_numbers(source)
        
        # Step 4: Add junk code
        source = self._add_junk_code(source)
        
        # Step 5: Create VM wrapper
        final_code = self._create_vm_wrapper(source, strings)
        
        # Step 6: Minify
        if getattr(self.config, 'minify', True):
            final_code = self._minify(final_code)
        
        return final_code
    
    def _generate_var_name(self, length: int = 8) -> str:
        """Generate random variable name"""
        chars = string.ascii_letters + '_'
        first = random.choice(string.ascii_letters + '_')
        rest = ''.join(random.choices(chars + string.digits, k=length-1))
        return first + rest
    
    def _extract_strings(self, source: str) -> tuple:
        """Extract and replace strings with placeholders"""
        strings = []
        
        def replace_string(match):
            s = match.group(0)
            # Skip if it's part of a comment
            idx = len(strings)
            strings.append(s[1:-1])  # Remove quotes
            return f'__STR_{idx}__'
        
        # Match single and double quoted strings (basic)
        # Note: This is simplified, doesn't handle all edge cases
        pattern = r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\''
        source = re.sub(pattern, replace_string, source)
        
        return source, strings
    
    def _rename_variables(self, source: str) -> str:
        """Rename local variables"""
        
        # Find local variable declarations
        local_pattern = r'\blocal\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        def rename_local(match):
            var_name = match.group(1)
            if var_name in self.keywords or var_name in self.builtins:
                return match.group(0)
            
            if var_name not in self.var_map:
                self.var_map[var_name] = self._generate_var_name()
            
            return f'local {self.var_map[var_name]}'
        
        source = re.sub(local_pattern, rename_local, source)
        
        # Replace usages of renamed variables
        for old_name, new_name in self.var_map.items():
            # Word boundary replacement
            source = re.sub(rf'\b{re.escape(old_name)}\b', new_name, source)
        
        return source
    
    def _encrypt_numbers(self, source: str) -> str:
        """Encrypt number literals"""
        
        def encrypt_number(match):
            num_str = match.group(0)
            try:
                if '.' in num_str:
                    num = float(num_str)
                    # Create expression that equals the number
                    a = random.randint(1, 100)
                    b = num * a
                    return f'({b}/{a})'
                else:
                    num = int(num_str)
                    if num == 0:
                        return '(0)'
                    # Create arithmetic expression
                    a = random.randint(1, 100)
                    b = num + a
                    return f'({b}-{a})'
            except:
                return num_str
        
        # Match numbers (but not in variable names)
        pattern = r'(?<![a-zA-Z_])(\d+\.?\d*)(?![a-zA-Z_])'
        source = re.sub(pattern, encrypt_number, source)
        
        return source
    
    def _add_junk_code(self, source: str) -> str:
        """Add junk/dead code"""
        junk_snippets = [
            'if false then local _ = nil end',
            'do local _ = {} end',
            'local _ = (function() end)()',
            '_ = nil',
            'if nil then end',
        ]
        
        lines = source.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            # Randomly insert junk
            if random.random() < 0.1:  # 10% chance
                junk = random.choice(junk_snippets)
                new_lines.append(junk)
        
        return '\n'.join(new_lines)
    
    def _create_vm_wrapper(self, source: str, strings: List[str]) -> str:
        """Create VM-style wrapper (Luraph-like)"""
        
        # Encode the source
        encoded = base64.b64encode(
            zlib.compress(source.encode('utf-8'))
        ).decode('ascii')
        
        # Encode strings separately
        encoded_strings = []
        for s in strings:
            enc = base64.b64encode(s.encode('utf-8')).decode('ascii')
            encoded_strings.append(enc)
        
        # Generate header
        header = self._generate_header()
        
        # Generate string decoder
        string_decoder = self._generate_string_decoder(encoded_strings)
        
        # Generate main decoder/executor
        decoder = self._generate_decoder(encoded)
        
        # Combine
        return f"{header}\n\n{string_decoder}\n\n{decoder}"
    
    def _generate_header(self) -> str:
        """Generate file header"""
        preset_name = getattr(self.config, 'name', 'custom')
        
        if preset_name == 'luraph':
            return f"-- This file was protected using Luraph Obfuscator [https://lura.ph/]\n-- Seed: {self.seed}"
        else:
            return f"-- Protected by Lua Obfuscator\n-- Preset: {preset_name}\n-- Seed: {self.seed}"
    
    def _generate_string_decoder(self, encoded_strings: List[str]) -> str:
        """Generate string decoding table"""
        if not encoded_strings:
            return "local __S = {}"
        
        # Create encoded string table
        str_entries = ','.join(f'"{s}"' for s in encoded_strings)
        
        return f"""local __S = (function()
    local t = {{{str_entries}}}
    local d = {{}}
    for i, v in ipairs(t) do
        local s = ""
        local b = {{}}
        for c in v:gmatch(".") do
            b[#b+1] = c
        end
        -- Base64 decode
        local chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        local function decode(data)
            data = data:gsub('[^'..chars..'=]', '')
            return (data:gsub('.', function(x)
                if x == '=' then return '' end
                local r, f = '', (chars:find(x) - 1)
                for i = 6, 1, -1 do r = r .. (f % 2 ^ i - f % 2 ^ (i-1) > 0 and '1' or '0') end
                return r
            end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
                if #x ~= 8 then return '' end
                local c = 0
                for i = 1, 8 do c = c + (x:sub(i, i) == '1' and 2 ^ (8 - i) or 0) end
                return string.char(c)
            end))
        end
        d[i] = decode(v)
    end
    return d
end)()"""
    
    def _generate_decoder(self, encoded: str) -> str:
        """Generate main decoder and executor"""
        
        # Split encoded string for obfuscation
        chunk_size = 76
        chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
        chunks_str = '+\n'.join(f'"{c}"' for c in chunks)
        
        return f"""local __CODE = {chunks_str}

local function __DECODE(data)
    local chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    data = data:gsub('[^'..chars..'=]', '')
    return (data:gsub('.', function(x)
        if x == '=' then return '' end
        local r, f = '', (chars:find(x) - 1)
        for i = 6, 1, -1 do r = r .. (f % 2 ^ i - f % 2 ^ (i-1) > 0 and '1' or '0') end
        return r
    end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
        if #x ~= 8 then return '' end
        local c = 0
        for i = 1, 8 do c = c + (x:sub(i, i) == '1' and 2 ^ (8 - i) or 0) end
        return string.char(c)
    end))
end

local function __DECOMPRESS(data)
    local ok, result = pcall(function()
        -- Simple zlib decompress using available methods
        if _G.syn and syn.decompress then
            return syn.decompress(data)
        elseif _G.decompress then
            return decompress(data)
        elseif _G.lz4 and lz4.decompress then
            return lz4.decompress(data)
        else
            -- Fallback: assume environment has zlib
            local z = require and pcall(require, 'zlib') and require('zlib')
            if z then return z.decompress(data) end
            -- Last resort: return as-is (won't work if compressed)
            return data
        end
    end)
    return ok and result or data
end

local function __EXECUTE()
    local decoded = __DECODE(__CODE)
    local decompressed = __DECOMPRESS(decoded)
    
    -- Replace string placeholders
    local code = decompressed:gsub("__STR_(%d+)__", function(idx)
        local i = tonumber(idx) + 1
        return '"' .. (__S[i] or "") .. '"'
    end)
    
    -- Execute
    local fn, err = loadstring or load
    if fn then
        local chunk, lerr = fn(code)
        if chunk then
            return chunk()
        else
            error("Failed to load: " .. tostring(lerr))
        end
    else
        error("No loadstring available")
    end
end

return __EXECUTE()"""
    
    def _minify(self, code: str) -> str:
        """Minify the code"""
        lines = []
        in_multiline_string = False
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Keep comments that are part of header
            if stripped.startswith('--') and ('Protected' in stripped or 'Luraph' in stripped or 'Seed' in stripped):
                lines.append(stripped)
                continue
            
            # Remove single-line comments (but not in strings)
            if '--' in stripped and not in_multiline_string:
                # Simple check - remove comment part
                comment_idx = stripped.find('--')
                # Make sure it's not inside a string
                quote_count = stripped[:comment_idx].count('"') + stripped[:comment_idx].count("'")
                if quote_count % 2 == 0:
                    stripped = stripped[:comment_idx].strip()
                    if not stripped:
                        continue
            
            lines.append(stripped)
        
        # Join with minimal whitespace
        result = ' '.join(lines)
        
        # Clean up multiple spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        # Clean up spaces around operators
        for op in ['=', '+', '-', '*', '/', '%', '^', '<', '>', ',', '(', ')', '{', '}', '[', ']']:
            result = result.replace(f' {op} ', op)
            result = result.replace(f' {op}', op)
            result = result.replace(f'{op} ', op)
        
        # Restore necessary spaces
        result = result.replace('local', 'local ')
        result = result.replace('function', 'function ')
        result = result.replace('return', 'return ')
        result = result.replace('if', 'if ')
        result = result.replace('then', ' then ')
        result = result.replace('else', ' else ')
        result = result.replace('elseif', ' elseif ')
        result = result.replace('end', ' end ')
        result = result.replace('do', ' do ')
        result = result.replace('for', 'for ')
        result = result.replace('while', 'while ')
        result = result.replace('repeat', 'repeat ')
        result = result.replace('until', ' until ')
        result = result.replace('in', ' in ')
        result = result.replace('and', ' and ')
        result = result.replace('or', ' or ')
        result = result.replace('not', 'not ')
        
        # Clean up double spaces again
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        return result.strip()

# ============================================
# Main Pipeline
# ============================================

class ObfuscationPipeline:
    """Obfuscation pipeline - supports both source and bytecode"""
    
    def __init__(self, config: ObfuscatorConfig):
        self.config = config
    
    def process(self, input_path: str, output_path: str) -> PipelineResult:
        """Execute pipeline"""
        result = PipelineResult()
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input not found: {input_path}")
            
            result.input_size = os.path.getsize(input_path)
            
            # Read file
            with open(input_path, 'rb') as f:
                file_data = f.read()
            
            # ============================================
            # DETECT FILE TYPE
            # ============================================
            
            is_bytecode = self._is_bytecode(file_data)
            
            if is_bytecode:
                print("[*] Detected: Bytecode")
                final_code = self._process_bytecode(input_path, file_data)
            else:
                print("[*] Detected: Source code")
                final_code = self._process_source(file_data)
            
            # Write output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_code)
            
            # Success
            result.success = True
            result.output_path = output_path
            result.output_size = len(final_code.encode('utf-8'))
            result.size_ratio = result.output_size / result.input_size if result.input_size > 0 else 0
            result.stats = {
                'input_type': 'bytecode' if is_bytecode else 'source',
                'output_size': result.output_size,
            }
            
            print(f"[✓] Obfuscation complete: {result.output_size} bytes")
            
        except Exception as e:
            result.errors.append(str(e))
            result.success = False
            print(f"[✗] Error: {e}")
            import traceback
            traceback.print_exc()
        
        result.total_time = time.time() - start_time
        return result
    
    def _is_bytecode(self, data: bytes) -> bool:
        """Check if data is bytecode"""
        # Lua 5.x signature
        if data[:4] == b'\x1bLua':
            return True
        # LuaJIT signature
        if data[:3] == b'\x1bLJ':
            return True
        # Luau (version byte 0-6)
        if len(data) > 0 and data[0] <= 6:
            # Additional check - try to decode as UTF-8
            try:
                text = data[:200].decode('utf-8')
                # If it looks like Lua code, it's source
                if any(kw in text for kw in ['function', 'local', 'if', 'end', 'return', '--']):
                    return False
            except UnicodeDecodeError:
                return True
        
        # Try to decode as UTF-8
        try:
            data.decode('utf-8')
            return False  # Successfully decoded = source code
        except UnicodeDecodeError:
            return True  # Failed to decode = probably bytecode
    
    def _process_source(self, file_data: bytes) -> str:
        """Process Lua source code"""
        print("[1/4] Reading source code...")
        
        try:
            source = file_data.decode('utf-8')
        except UnicodeDecodeError:
            source = file_data.decode('latin-1')
        
        print(f"  ✓ Read {len(source)} characters, {len(source.splitlines())} lines")
        
        print("[2/4] Initializing obfuscator...")
        obfuscator = LuaSourceObfuscator(self.config)
        print("  ✓ Obfuscator ready")
        
        print("[3/4] Obfuscating...")
        result = obfuscator.obfuscate(source)
        print(f"  ✓ Obfuscated: {len(result)} characters")
        
        print("[4/4] Finalizing...")
        print("  ✓ Done")
        
        return result
    
    def _process_bytecode(self, input_path: str, file_data: bytes) -> str:
        """Process Lua bytecode (original pipeline logic)"""
        
        if not HAS_BYTECODE_PARSER:
            raise Exception("Bytecode parser not available. Please provide source code (.lua) instead.")
        
        print("[1/7] Parsing bytecode...")
        chunk = parse_bytecode_file(input_path)
        print(f"  ✓ Parsed {len(chunk.all_functions)} functions")
        
        print("[2/7] Analyzing...")
        for func in chunk.all_functions:
            analyzer = ControlFlowAnalyzer(func)
            analyzer.analyze()
        print("  ✓ Analysis complete")
        
        print("[3/7] Transforming...")
        if HAS_TRANSFORMER:
            self.config.transform.generate_random_values()
            transformer = BytecodeTransformer(self.config.transform)
            transformed_chunk, metadata = transformer.transform(chunk)
        else:
            transformed_chunk = chunk
            metadata = {}
        print("  ✓ Transformed")
        
        print("[4/7] Encrypting...")
        if HAS_ENCRYPTION:
            self.config.encryption.generate_keys()
            enc_manager = EncryptionManager(self.config.encryption)
        print("  ✓ Encrypted")
        
        print("[5/7] Generating VM...")
        if HAS_VM_GENERATOR:
            generator = LuaVMGenerator(self.config.transform, self.config.vm)
            vm_code = generator.generate_vm()
        else:
            vm_code = "-- VM code placeholder"
        print(f"  ✓ VM generated ({len(vm_code)} chars)")
        
        print("[6/7] Adding protection...")
        if HAS_ANTITAMPER:
            self.config.antitamper.generate_keys()
            antitamper = AntiTamperGenerator(self.config.antitamper)
            protection_code = antitamper.generate_all_protections()
        else:
            protection_code = ""
        print(f"  ✓ Protection added")
        
        print("[7/7] Combining...")
        
        # Serialize bytecode
        if HAS_VM_GENERATOR:
            serializer = BytecodeSerializer(None)
            bytecode_str = serializer.serialize_function(transformed_chunk.main_function)
        else:
            bytecode_str = "{}"
        
        # Combine
        parts = [
            f"-- Protected by Lua Obfuscator",
            protection_code,
            vm_code,
            f"local bytecode = {bytecode_str}",
            "-- Execute\nreturn bytecode"
        ]
        
        final_code = "\n\n".join(p for p in parts if p)
        print(f"  ✓ Combined ({len(final_code)} chars)")
        
        return final_code