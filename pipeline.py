# ============================================
# File: pipeline.py (FULL VERSION untuk Replit)
# Dengan Compile Step + Proper Obfuscation
# ============================================

import os
import time
import subprocess
import tempfile
import random
import string
import struct
import base64
import zlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import hashlib

from config_manager import ObfuscatorConfig

# Import komponen obfuscator
try:
    from lua_parser import parse_bytecode, parse_bytecode_file, ControlFlowAnalyzer, LuaChunk
    HAS_PARSER = True
except ImportError as e:
    print(f"Warning: lua_parser not available: {e}")
    HAS_PARSER = False

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

# ============================================
# Pipeline Result
# ============================================

@dataclass
class PipelineResult:
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
# Lua Compiler
# ============================================

class LuaCompiler:
    """Compile Lua source to bytecode"""
    
    LUAC_PATHS = [
        'luac', 'luac5.1', 'luac5.3', 'luac5.4',
        '/usr/bin/luac', '/usr/local/bin/luac',
        '/nix/store/*/bin/luac',  # Replit nix
        './luac', './bin/luac',
    ]
    
    def __init__(self):
        self.luac_path = self._find_luac()
        self.version = self._get_version() if self.luac_path else None
    
    def _find_luac(self) -> Optional[str]:
        """Find luac executable"""
        import glob
        
        for pattern in self.LUAC_PATHS:
            if '*' in pattern:
                matches = glob.glob(pattern)
                for path in matches:
                    if self._test_luac(path):
                        return path
            else:
                if self._test_luac(pattern):
                    return pattern
        return None
    
    def _test_luac(self, path: str) -> bool:
        """Test if luac works"""
        try:
            result = subprocess.run(
                [path, '-v'],
                capture_output=True,
                timeout=5
            )
            return b'Lua' in result.stdout or b'Lua' in result.stderr
        except:
            return False
    
    def _get_version(self) -> str:
        try:
            result = subprocess.run(
                [self.luac_path, '-v'],
                capture_output=True,
                timeout=5
            )
            output = (result.stdout + result.stderr).decode('utf-8', errors='ignore')
            return output.strip()
        except:
            return "unknown"
    
    def is_available(self) -> bool:
        return self.luac_path is not None
    
    def compile(self, source_path: str) -> bytes:
        """Compile Lua source to bytecode"""
        if not self.is_available():
            raise RuntimeError("luac not found")
        
        fd, output_path = tempfile.mkstemp(suffix='.luac')
        os.close(fd)
        
        try:
            result = subprocess.run(
                [self.luac_path, '-o', output_path, source_path],
                capture_output=True,
                timeout=60
            )
            
            if result.returncode != 0:
                error = result.stderr.decode('utf-8', errors='replace')
                raise RuntimeError(f"Compilation failed: {error}")
            
            with open(output_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def compile_source(self, source: str) -> bytes:
        """Compile source string"""
        fd, source_path = tempfile.mkstemp(suffix='.lua')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(source)
            return self.compile(source_path)
        finally:
            if os.path.exists(source_path):
                os.unlink(source_path)

# ============================================
# Advanced VM Generator (Luraph-style)
# ============================================

class AdvancedVMGenerator:
    """Generate obfuscated VM interpreter"""
    
    def __init__(self, config, seed: int = None):
        self.config = config
        self.seed = seed or random.randint(1, 999999999)
        random.seed(self.seed)
        
        # Opcode mapping (shuffled)
        self.opcode_map = self._generate_opcode_map()
        
        # Variable name generator
        self.var_counter = 0
    
    def _generate_opcode_map(self) -> Dict[int, int]:
        """Generate shuffled opcode mapping"""
        opcodes = list(range(48))  # Lua 5.1 has 38 opcodes, extra for padding
        shuffled = opcodes.copy()
        random.shuffle(shuffled)
        return {orig: shuf for orig, shuf in zip(opcodes, shuffled)}
    
    def _gen_var(self, prefix: str = "v") -> str:
        """Generate random variable name"""
        self.var_counter += 1
        chars = string.ascii_letters + '_'
        name = random.choice(string.ascii_letters)
        name += ''.join(random.choices(chars + string.digits, k=random.randint(5, 12)))
        return name
    
    def generate_vm(self, chunk: 'LuaChunk') -> str:
        """Generate complete VM code"""
        
        parts = []
        
        # Header
        parts.append(self._generate_header())
        
        # Environment capture
        parts.append(self._generate_environment())
        
        # Utility functions
        parts.append(self._generate_utilities())
        
        # String table
        parts.append(self._generate_string_table(chunk))
        
        # Bytecode data
        parts.append(self._generate_bytecode_data(chunk))
        
        # VM interpreter
        parts.append(self._generate_interpreter())
        
        # Anti-tamper checks
        parts.append(self._generate_antitamper())
        
        # Execution wrapper
        parts.append(self._generate_executor())
        
        return '\n'.join(parts)
    
    def _generate_header(self) -> str:
        preset = getattr(self.config, 'name', 'custom')
        
        if preset == 'luraph':
            return f"""-- This file was protected using Luraph Obfuscator v15.2 [https://lura.ph/]
-- Seed: {self.seed}
-- WARNING: Tampering with this file will cause it to fail!
"""
        else:
            return f"""-- Protected by Advanced Lua Obfuscator
-- Preset: {preset}
-- Seed: {self.seed}
-- DO NOT MODIFY THIS FILE
"""
    
    def _generate_environment(self) -> str:
        """Capture environment references"""
        env_var = self._gen_var()
        
        return f"""
local {env_var} = _G or _ENV or getfenv and getfenv() or {{}}
local {self._gen_var()} = setmetatable
local {self._gen_var()} = getmetatable
local {self._gen_var()} = type
local {self._gen_var()} = select
local {self._gen_var()} = unpack or table.unpack
local {self._gen_var()} = tonumber
local {self._gen_var()} = tostring
local {self._gen_var()} = pairs
local {self._gen_var()} = ipairs
local {self._gen_var()} = next
local {self._gen_var()} = rawget
local {self._gen_var()} = rawset
local {self._gen_var()} = pcall
local {self._gen_var()} = error
local {self._gen_var()} = coroutine.create
local {self._gen_var()} = coroutine.wrap
local {self._gen_var()} = coroutine.yield
local {self._gen_var()} = coroutine.resume
local {self._gen_var()} = string.byte
local {self._gen_var()} = string.char
local {self._gen_var()} = string.sub
local {self._gen_var()} = string.gsub
local {self._gen_var()} = string.rep
local {self._gen_var()} = string.format
local {self._gen_var()} = math.floor
local {self._gen_var()} = math.abs
local {self._gen_var()} = math.fmod
local {self._gen_var()} = table.insert
local {self._gen_var()} = table.remove
local {self._gen_var()} = table.concat
local {self._gen_var()} = bit32 or bit or {{}}
"""
    
    def _generate_utilities(self) -> str:
        """Generate utility functions"""
        
        # Generate obfuscated utility function names
        decode_fn = self._gen_var()
        xor_fn = self._gen_var()
        unpack_fn = self._gen_var()
        
        return f"""
local function {xor_fn}(a, b)
    local r = 0
    local p = 1
    while a > 0 or b > 0 do
        local x = a % 2
        local y = b % 2
        if x ~= y then r = r + p end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        p = p * 2
    end
    return r
end

local function {decode_fn}(s, k)
    local r = {{}}
    local kl = #k
    for i = 1, #s do
        local c = string.byte(s, i)
        local ki = ((i - 1) % kl) + 1
        local kc = string.byte(k, ki)
        r[i] = string.char({xor_fn}(c, kc))
    end
    return table.concat(r)
end

local function {unpack_fn}(data, pos)
    local b1, b2, b3, b4 = string.byte(data, pos, pos + 3)
    return b1 + b2 * 256 + b3 * 65536 + b4 * 16777216, pos + 4
end
"""
    
    def _generate_string_table(self, chunk: 'LuaChunk') -> str:
        """Generate encrypted string table"""
        
        # Collect all strings
        strings = []
        for func in chunk.all_functions:
            for const in func.constants:
                if const.type == 4 and isinstance(const.value, str):
                    if const.value not in strings:
                        strings.append(const.value)
        
        if not strings:
            return f"local {self._gen_var()} = {{}}"
        
        # Generate encryption key
        key = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        
        # Encrypt strings
        encrypted = []
        for s in strings:
            enc_bytes = []
            for i, c in enumerate(s):
                ki = i % len(key)
                enc_bytes.append(chr(ord(c) ^ ord(key[ki])))
            enc = ''.join(enc_bytes)
            # Escape for Lua string
            escaped = ''
            for c in enc:
                o = ord(c)
                if o < 32 or o > 126 or c in '"\\':
                    escaped += f'\\{o}'
                else:
                    escaped += c
            encrypted.append(f'"{escaped}"')
        
        table_var = self._gen_var()
        key_var = self._gen_var()
        decode_var = self._gen_var()
        
        return f"""
local {key_var} = "{key}"
local {table_var} = {{{','.join(encrypted)}}}
local {decode_var} = {{}}
for i, v in ipairs({table_var}) do
    local r = {{}}
    for j = 1, #v do
        local c = string.byte(v, j)
        local k = string.byte({key_var}, ((j - 1) % #{key_var}) + 1)
        r[j] = string.char(c ~ k or (c + k) % 256)
    end
    {decode_var}[i] = table.concat(r)
end
"""
    
    def _generate_bytecode_data(self, chunk: 'LuaChunk') -> str:
        """Generate encoded bytecode data"""
        
        # Serialize function data
        def serialize_func(func) -> List[int]:
            data = []
            
            # Header
            data.append(func.num_params)
            data.append(func.is_vararg)
            data.append(func.max_stack_size)
            
            # Instructions count
            data.extend(self._int_to_bytes(len(func.instructions)))
            
            # Instructions (with opcode shuffling)
            for instr in func.instructions:
                # Shuffle opcode
                orig_op = instr.raw & 0x3F
                new_op = self.opcode_map.get(orig_op, orig_op)
                new_raw = (instr.raw & ~0x3F) | new_op
                data.extend(self._int_to_bytes(new_raw))
            
            # Constants count
            data.extend(self._int_to_bytes(len(func.constants)))
            
            # Constants
            for const in func.constants:
                data.append(const.type)
                if const.type == 0:  # nil
                    pass
                elif const.type == 1:  # bool
                    data.append(1 if const.value else 0)
                elif const.type == 3:  # number
                    data.extend(self._double_to_bytes(const.value))
                elif const.type == 4:  # string
                    s = const.value.encode('utf-8') if isinstance(const.value, str) else b''
                    data.extend(self._int_to_bytes(len(s)))
                    data.extend(s)
            
            # Prototypes count
            data.extend(self._int_to_bytes(len(func.prototypes)))
            
            return data
        
        main_data = serialize_func(chunk.main_function)
        
        # Compress data
        compressed = zlib.compress(bytes(main_data), level=9)
        
        # Encode to base64-like format
        encoded = base64.b64encode(compressed).decode('ascii')
        
        # Split into chunks for readability
        chunk_size = 76
        chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
        
        data_var = self._gen_var()
        
        return f"""
local {data_var} = table.concat({{
{chr(10).join(f'    "{c}",' for c in chunks)}
}})
"""
    
    def _int_to_bytes(self, n: int) -> List[int]:
        """Convert int to 4 bytes (little endian)"""
        return [
            n & 0xFF,
            (n >> 8) & 0xFF,
            (n >> 16) & 0xFF,
            (n >> 24) & 0xFF
        ]
    
    def _double_to_bytes(self, n: float) -> List[int]:
        """Convert double to 8 bytes"""
        packed = struct.pack('<d', n)
        return list(packed)
    
    def _generate_interpreter(self) -> str:
        """Generate VM interpreter"""
        
        vm_var = self._gen_var()
        
        # Generate opcode handlers
        handlers = self._generate_opcode_handlers()
        
        return f"""
local function {vm_var}(bytecode, env)
    -- Decode bytecode
    local data = (function()
        local b64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        local dec = {{}}
        for i = 1, 64 do dec[b64:sub(i,i)] = i - 1 end
        
        local str = bytecode:gsub('[^'..b64..'=]', '')
        local result = {{}}
        local i = 1
        while i <= #str do
            local a = dec[str:sub(i,i)] or 0
            local b = dec[str:sub(i+1,i+1)] or 0
            local c = dec[str:sub(i+2,i+2)] or 0
            local d = dec[str:sub(i+3,i+3)] or 0
            
            local n = a * 262144 + b * 4096 + c * 64 + d
            
            result[#result + 1] = string.char(math.floor(n / 65536) % 256)
            if str:sub(i+2,i+2) ~= '=' then
                result[#result + 1] = string.char(math.floor(n / 256) % 256)
            end
            if str:sub(i+3,i+3) ~= '=' then
                result[#result + 1] = string.char(n % 256)
            end
            i = i + 4
        end
        return table.concat(result)
    end)()
    
    -- Decompress
    local ok, decompressed = pcall(function()
        -- Try built-in decompress methods
        if _G.syn and syn.decompress then
            return syn.decompress(data)
        elseif _G.lz4 and lz4.decompress then
            return lz4.decompress(data)
        elseif _G.zlib and zlib.decompress then
            return zlib.decompress(data)
        else
            -- Fallback: assume not compressed or use custom
            return data
        end
    end)
    
    if not ok then
        error("Decompression failed")
    end
    
    data = decompressed
    
    -- Parse bytecode
    local pos = 1
    local function readByte()
        local b = string.byte(data, pos)
        pos = pos + 1
        return b
    end
    
    local function readInt()
        local b1, b2, b3, b4 = string.byte(data, pos, pos + 3)
        pos = pos + 4
        return b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
    end
    
    local function readDouble()
        local bytes = {{string.byte(data, pos, pos + 7)}}
        pos = pos + 8
        -- IEEE 754 decode
        local sign = bytes[8] >= 128 and -1 or 1
        local exp = (bytes[8] % 128) * 16 + math.floor(bytes[7] / 16)
        local mantissa = 0
        for i = 1, 6 do
            mantissa = mantissa + bytes[i] * (2 ^ ((i - 1) * 8))
        end
        mantissa = mantissa + (bytes[7] % 16) * (2 ^ 48)
        
        if exp == 0 then
            return sign * mantissa * (2 ^ -1074)
        elseif exp == 2047 then
            return mantissa == 0 and (sign * math.huge) or (0/0)
        else
            return sign * (1 + mantissa / (2 ^ 52)) * (2 ^ (exp - 1023))
        end
    end
    
    local function readString()
        local len = readInt()
        if len == 0 then return "" end
        local s = data:sub(pos, pos + len - 1)
        pos = pos + len
        return s
    end
    
    -- Opcode reverse mapping
    local opcodeMap = {{}}
    {self._generate_opcode_reverse_map()}
    
    -- Parse function
    local function parseFunction()
        local func = {{}}
        
        func.numParams = readByte()
        func.isVararg = readByte()
        func.maxStack = readByte()
        
        -- Instructions
        local numInstr = readInt()
        func.instructions = {{}}
        for i = 1, numInstr do
            local raw = readInt()
            -- Unshuffle opcode
            local op = raw % 64
            local realOp = opcodeMap[op] or op
            func.instructions[i] = (raw - op) + realOp
        end
        
        -- Constants
        local numConst = readInt()
        func.constants = {{}}
        for i = 1, numConst do
            local t = readByte()
            if t == 0 then
                func.constants[i] = nil
            elseif t == 1 then
                func.constants[i] = readByte() ~= 0
            elseif t == 3 then
                func.constants[i] = readDouble()
            elseif t == 4 then
                func.constants[i] = readString()
            end
        end
        
        -- Prototypes
        local numProtos = readInt()
        func.prototypes = {{}}
        for i = 1, numProtos do
            func.prototypes[i] = parseFunction()
        end
        
        return func
    end
    
    local mainFunc = parseFunction()
    
    -- VM Execution
    local function execute(func, upvals)
        local stack = {{}}
        local top = 0
        local pc = 1
        local constants = func.constants
        local instructions = func.instructions
        local protos = func.prototypes
        local openUpvals = {{}}
        
        local function getUpval(idx)
            return upvals[idx + 1]
        end
        
        local function setUpval(idx, val)
            upvals[idx + 1] = val
        end
        
        while true do
            local instr = instructions[pc]
            if not instr then break end
            
            local op = instr % 64
            local A = math.floor(instr / 64) % 256
            local B = math.floor(instr / 8388608) % 512
            local C = math.floor(instr / 16384) % 512
            local Bx = math.floor(instr / 16384)
            local sBx = Bx - 131071
            
            pc = pc + 1
            
{handlers}
        end
        
        return stack[1]
    end
    
    return function(...)
        local args = {{...}}
        return execute(mainFunc, {{}})
    end
end
"""
    
    def _generate_opcode_reverse_map(self) -> str:
        """Generate reverse opcode mapping"""
        lines = []
        for orig, shuf in self.opcode_map.items():
            lines.append(f"    opcodeMap[{shuf}] = {orig}")
        return '\n'.join(lines)
    
    def _generate_opcode_handlers(self) -> str:
        """Generate opcode handlers"""
        
        # Simplified handlers for common opcodes
        return """
            -- MOVE
            if op == 0 then
                stack[A + 1] = stack[B + 1]
            
            -- LOADK
            elseif op == 1 then
                stack[A + 1] = constants[Bx + 1]
            
            -- LOADBOOL
            elseif op == 2 then
                stack[A + 1] = B ~= 0
                if C ~= 0 then pc = pc + 1 end
            
            -- LOADNIL
            elseif op == 3 then
                for i = A, B do
                    stack[i + 1] = nil
                end
            
            -- GETUPVAL
            elseif op == 4 then
                stack[A + 1] = getUpval(B)
            
            -- GETGLOBAL
            elseif op == 5 then
                stack[A + 1] = env[constants[Bx + 1]]
            
            -- GETTABLE
            elseif op == 6 then
                local key = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1] = stack[B + 1][key]
            
            -- SETGLOBAL
            elseif op == 7 then
                env[constants[Bx + 1]] = stack[A + 1]
            
            -- SETUPVAL
            elseif op == 8 then
                setUpval(B, stack[A + 1])
            
            -- SETTABLE
            elseif op == 9 then
                local key = B >= 256 and constants[B - 255] or stack[B + 1]
                local val = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1][key] = val
            
            -- NEWTABLE
            elseif op == 10 then
                stack[A + 1] = {}
            
            -- SELF
            elseif op == 11 then
                local key = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 2] = stack[B + 1]
                stack[A + 1] = stack[B + 1][key]
            
            -- ADD
            elseif op == 12 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1] = b + c
            
            -- SUB
            elseif op == 13 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1] = b - c
            
            -- MUL
            elseif op == 14 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1] = b * c
            
            -- DIV
            elseif op == 15 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1] = b / c
            
            -- MOD
            elseif op == 16 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1] = b % c
            
            -- POW
            elseif op == 17 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                stack[A + 1] = b ^ c
            
            -- UNM
            elseif op == 18 then
                stack[A + 1] = -stack[B + 1]
            
            -- NOT
            elseif op == 19 then
                stack[A + 1] = not stack[B + 1]
            
            -- LEN
            elseif op == 20 then
                stack[A + 1] = #stack[B + 1]
            
            -- CONCAT
            elseif op == 21 then
                local t = {}
                for i = B, C do
                    t[#t + 1] = stack[i + 1]
                end
                stack[A + 1] = table.concat(t)
            
            -- JMP
            elseif op == 22 then
                pc = pc + sBx
            
            -- EQ
            elseif op == 23 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                if (b == c) ~= (A ~= 0) then
                    pc = pc + 1
                end
            
            -- LT
            elseif op == 24 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                if (b < c) ~= (A ~= 0) then
                    pc = pc + 1
                end
            
            -- LE
            elseif op == 25 then
                local b = B >= 256 and constants[B - 255] or stack[B + 1]
                local c = C >= 256 and constants[C - 255] or stack[C + 1]
                if (b <= c) ~= (A ~= 0) then
                    pc = pc + 1
                end
            
            -- TEST
            elseif op == 26 then
                if (not not stack[A + 1]) ~= (C ~= 0) then
                    pc = pc + 1
                end
            
            -- TESTSET
            elseif op == 27 then
                if (not not stack[B + 1]) ~= (C ~= 0) then
                    pc = pc + 1
                else
                    stack[A + 1] = stack[B + 1]
                end
            
            -- CALL
            elseif op == 28 then
                local func = stack[A + 1]
                local args = {}
                local nArgs = B - 1
                if nArgs < 0 then nArgs = top - A - 1 end
                for i = 1, nArgs do
                    args[i] = stack[A + 1 + i]
                end
                local rets = {func(unpack(args))}
                local nRets = C - 1
                if nRets < 0 then
                    for i = 1, #rets do
                        stack[A + i] = rets[i]
                    end
                    top = A + #rets
                else
                    for i = 1, nRets do
                        stack[A + i] = rets[i]
                    end
                end
            
            -- TAILCALL
            elseif op == 29 then
                local func = stack[A + 1]
                local args = {}
                local nArgs = B - 1
                if nArgs < 0 then nArgs = top - A - 1 end
                for i = 1, nArgs do
                    args[i] = stack[A + 1 + i]
                end
                return func(unpack(args))
            
            -- RETURN
            elseif op == 30 then
                local rets = {}
                local nRets = B - 1
                if nRets < 0 then nRets = top - A end
                for i = 1, nRets do
                    rets[i] = stack[A + i]
                end
                return unpack(rets)
            
            -- FORLOOP
            elseif op == 31 then
                local step = stack[A + 3]
                local idx = stack[A + 1] + step
                local limit = stack[A + 2]
                if (step > 0 and idx <= limit) or (step <= 0 and idx >= limit) then
                    pc = pc + sBx
                    stack[A + 1] = idx
                    stack[A + 4] = idx
                end
            
            -- FORPREP
            elseif op == 32 then
                stack[A + 1] = stack[A + 1] - stack[A + 3]
                pc = pc + sBx
            
            -- TFORLOOP
            elseif op == 33 then
                local rets = {stack[A + 1](stack[A + 2], stack[A + 3])}
                for i = 1, C do
                    stack[A + 3 + i] = rets[i]
                end
                if stack[A + 4] ~= nil then
                    stack[A + 3] = stack[A + 4]
                else
                    pc = pc + 1
                end
            
            -- SETLIST
            elseif op == 34 then
                local tbl = stack[A + 1]
                local nElems = B == 0 and (top - A) or B
                local offset = (C - 1) * 50
                for i = 1, nElems do
                    tbl[offset + i] = stack[A + 1 + i]
                end
            
            -- CLOSE
            elseif op == 35 then
                -- Close upvalues (simplified)
            
            -- CLOSURE
            elseif op == 36 then
                local proto = protos[Bx + 1]
                local newUpvals = {}
                stack[A + 1] = function(...)
                    return execute(proto, newUpvals)(...)
                end
            
            -- VARARG
            elseif op == 37 then
                -- Handle varargs (simplified)
            
            end
"""
    
    def _generate_antitamper(self) -> str:
        """Generate anti-tamper checks"""
        return f"""
-- Anti-tamper
local function {self._gen_var()}()
    local ok, err = pcall(function()
        -- Environment check
        if not _G or not _ENV and not getfenv then
            return false
        end
        -- Debug check
        if debug and debug.getinfo then
            local info = debug.getinfo(1)
            if info and info.what == "C" then
                return false
            end
        end
        return true
    end)
    return ok and err
end
"""
    
    def _generate_executor(self) -> str:
        """Generate execution wrapper"""
        
        vm_call = self._gen_var()
        data_ref = self._gen_var()
        
        return f"""
-- Initialize and execute
local {vm_call} = {self._gen_var()}({self._gen_var()}, _G or _ENV)
if {vm_call} then
    return {vm_call}()
else
    error("VM initialization failed - file may be corrupted")
end
"""

# ============================================
# Main Pipeline Class
# ============================================

class ObfuscationPipeline:
    """Main obfuscation pipeline"""
    
    def __init__(self, config: ObfuscatorConfig):
        self.config = config
        self.compiler = LuaCompiler()
        
        # Check components
        print(f"[*] Components:")
        print(f"    Parser: {'✓' if HAS_PARSER else '✗'}")
        print(f"    Transformer: {'✓' if HAS_TRANSFORMER else '✗'}")
        print(f"    VM Generator: {'✓' if HAS_VM_GENERATOR else '✗ (using built-in)'}")
        print(f"    Compiler: {'✓ ' + self.compiler.version if self.compiler.is_available() else '✗'}")
    
    def process(self, input_path: str, output_path: str) -> PipelineResult:
        """Execute obfuscation pipeline"""
        result = PipelineResult()
        start_time = time.time()
        
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input not found: {input_path}")
            
            result.input_size = os.path.getsize(input_path)
            
            with open(input_path, 'rb') as f:
                file_data = f.read()
            
            # Detect file type
            is_source = self._is_source_code(file_data)
            
            print(f"\n[*] Input: {input_path}")
            print(f"[*] Size: {result.input_size:,} bytes")
            print(f"[*] Type: {'Source code' if is_source else 'Bytecode'}")
            print(f"[*] Preset: {getattr(self.config, 'name', 'custom')}")
            print()
            
            # Step 1: Get bytecode
            if is_source:
                print("[1/6] Compiling source to bytecode...")
                if not self.compiler.is_available():
                    raise RuntimeError(
                        "Lua compiler (luac) not found!\n"
                        "Install with: apt install lua5.3\n"
                        "Or add to replit.nix: pkgs.lua5_3"
                    )
                bytecode = self.compiler.compile(input_path)
                print(f"  ✓ Compiled: {len(bytecode):,} bytes")
            else:
                print("[1/6] Reading bytecode...")
                bytecode = file_data
                print(f"  ✓ Read: {len(bytecode):,} bytes")
            
            # Step 2: Parse bytecode
            print("[2/6] Parsing bytecode...")
            if not HAS_PARSER:
                raise RuntimeError("lua_parser module not available")
            
            chunk = parse_bytecode(bytecode)
            
            if not chunk.header.is_valid:
                errors = chunk.header.validation_errors
                raise RuntimeError(f"Invalid bytecode: {errors}")
            
            print(f"  ✓ Functions: {len(chunk.all_functions)}")
            print(f"  ✓ Instructions: {chunk.stats.get('total_instructions', 0):,}")
            print(f"  ✓ Constants: {chunk.stats.get('total_constants', 0):,}")
            
            # Step 3: Analyze
            print("[3/6] Analyzing control flow...")
            for func in chunk.all_functions:
                analyzer = ControlFlowAnalyzer(func)
                analyzer.analyze()
            print("  ✓ Analysis complete")
            
            # Step 4: Transform
            print("[4/6] Transforming bytecode...")
            if HAS_TRANSFORMER and hasattr(self.config, 'transform'):
                self.config.transform.generate_random_values()
                transformer = BytecodeTransformer(self.config.transform)
                chunk, _ = transformer.transform(chunk)
            print("  ✓ Transformed")
            
            # Step 5: Generate VM
            print("[5/6] Generating obfuscated VM...")
            
            seed = getattr(self.config, 'seed', None) or random.randint(1, 999999999)
            vm_gen = AdvancedVMGenerator(self.config, seed=seed)
            
            if HAS_VM_GENERATOR:
                # Use external generator if available
                generator = LuaVMGenerator(
                    getattr(self.config, 'transform', None),
                    getattr(self.config, 'vm', None)
                )
                vm_code = generator.generate_vm()
            else:
                # Use built-in generator
                vm_code = vm_gen.generate_vm(chunk)
            
            print(f"  ✓ VM generated: {len(vm_code):,} chars")
            
            # Step 6: Add protection layers
            print("[6/6] Adding protection layers...")
            
            final_code = vm_code
            
            # Add junk code if enabled
            if hasattr(self.config, 'transform') and getattr(self.config.transform, 'add_junk_code', False):
                final_code = self._add_junk_code(final_code)
            
            # Add watermark
            if hasattr(self.config, 'watermark') and self.config.watermark:
                final_code = self._add_watermark(final_code, self.config.watermark)
            
            print(f"  ✓ Final size: {len(final_code):,} chars")
            
            # Write output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_code)
            
            # Success
            result.success = True
            result.output_path = output_path
            result.output_size = len(final_code.encode('utf-8'))
            result.size_ratio = result.output_size / result.input_size
            result.stats = {
                'functions': len(chunk.all_functions),
                'instructions': chunk.stats.get('total_instructions', 0),
                'vm_size': len(vm_code),
            }
            
            print()
            print(f"[✓] Obfuscation complete!")
            print(f"    Output: {output_path}")
            print(f"    Size: {result.output_size:,} bytes ({result.size_ratio:.1f}x)")
            
        except Exception as e:
            result.errors.append(str(e))
            result.success = False
            print(f"\n[✗] Error: {e}")
            import traceback
            traceback.print_exc()
        
        result.total_time = time.time() - start_time
        return result
    
    def _is_source_code(self, data: bytes) -> bool:
        """Detect if input is source code"""
        if data[:4] == b'\x1bLua':
            return False
        if data[:3] == b'\x1bLJ':
            return False
        
        try:
            text = data[:500].decode('utf-8')
            keywords = ['function', 'local', 'if', 'then', 'end', 'for', 'while', 'return']
            return any(kw in text for kw in keywords)
        except:
            return False
    
    def _add_junk_code(self, code: str) -> str:
        """Add junk code to increase size and complexity"""
        junk_templates = [
            'if false then local {} = {} end',
            'do local {} = function() end end',
            '_ = nil',
            'local {} = (function() return nil end)()',
            'if nil then {} = {} end',
        ]
        
        lines = code.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            if random.random() < 0.15:  # 15% chance
                template = random.choice(junk_templates)
                var1 = ''.join(random.choices(string.ascii_letters, k=8))
                var2 = ''.join(random.choices(string.ascii_letters, k=8))
                junk = template.format(var1, var2)
                new_lines.append(junk)
        
        return '\n'.join(new_lines)
    
    def _add_watermark(self, code: str, watermark: str) -> str:
        """Add hidden watermark"""
        encoded = base64.b64encode(watermark.encode()).decode()
        return f'-- {encoded}\n{code}'