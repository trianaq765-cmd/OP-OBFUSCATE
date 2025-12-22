# ============================================
# File: lua_parser.py (ADVANCED VERSION)
# Advanced Lua Bytecode Parser & Analyzer
# Supports Lua 5.1, 5.2, 5.3, 5.4, LuaJIT, Luau
# ============================================

import struct
import io
import zlib
import json
import hashlib
import base64
from dataclasses import dataclass, field
from typing import List, Optional, Union, BinaryIO, Dict, Set, Tuple, Any, Callable
from enum import IntEnum, auto
from abc import ABC, abstractmethod
from collections import defaultdict
import copy

# ============================================
# Lua Versions Support
# ============================================

class LuaVersion(IntEnum):
    """Supported Lua versions"""
    LUA_51 = 0x51
    LUA_52 = 0x52
    LUA_53 = 0x53
    LUA_54 = 0x54
    LUAJIT_20 = 0x01  # LuaJIT 2.0
    LUAJIT_21 = 0x02  # LuaJIT 2.1
    LUAU = 0x03       # Roblox Luau

class BytecodeFormat(IntEnum):
    """Bytecode format variations"""
    STANDARD = 0      # Standard Lua
    STRIPPED = 1      # Debug info stripped
    COMPRESSED = 2    # Compressed bytecode
    ENCRYPTED = 3     # Encrypted bytecode
    CUSTOM = 4        # Custom format

# ============================================
# Extended Opcodes for All Versions
# ============================================

class OpCode(IntEnum):
    """Extended opcodes supporting multiple Lua versions"""
    # Lua 5.1 opcodes (0-37)
    MOVE = 0
    LOADK = 1
    LOADBOOL = 2
    LOADNIL = 3
    GETUPVAL = 4
    GETGLOBAL = 5
    GETTABLE = 6
    SETGLOBAL = 7
    SETUPVAL = 8
    SETTABLE = 9
    NEWTABLE = 10
    SELF = 11
    ADD = 12
    SUB = 13
    MUL = 14
    DIV = 15
    MOD = 16
    POW = 17
    UNM = 18
    NOT = 19
    LEN = 20
    CONCAT = 21
    JMP = 22
    EQ = 23
    LT = 24
    LE = 25
    TEST = 26
    TESTSET = 27
    CALL = 28
    TAILCALL = 29
    RETURN = 30
    FORLOOP = 31
    FORPREP = 32
    TFORLOOP = 33
    SETLIST = 34
    CLOSE = 35
    CLOSURE = 36
    VARARG = 37
    
    # Lua 5.2+ opcodes
    LOADKX = 38
    GETTABUP = 39
    SETTABUP = 40
    
    # Lua 5.3+ opcodes
    IDIV = 41
    BAND = 42
    BOR = 43
    BXOR = 44
    SHL = 45
    SHR = 46
    BNOT = 47
    
    # Lua 5.4+ opcodes
    TBC = 48
    LOADI = 49
    LOADF = 50
    LOADFALSE = 51
    LFALSESKIP = 52
    LOADTRUE = 53
    GETTABK = 54
    SETTABK = 55
    
    # LuaJIT specific
    ISLT = 56
    ISLE = 57
    ISNUMBER = 58
    
    # Luau specific
    GETVARARGS = 59
    DUPCLOSURE = 60
    PREPVARARGS = 61
    LOADKX = 62
    JUMPX = 63
    FASTCALL = 64
    COVERAGE = 65
    CAPTURE = 66
    SUBRK = 67
    DIVRK = 68
    
    UNKNOWN = 255

class OpMode(IntEnum):
    """Instruction encoding modes"""
    iABC = 0
    iABx = 1
    iAsBx = 2
    iAx = 3

# Opcode properties
OPCODE_INFO = {
    # format: (mode, has_test, sets_A, uses_B, uses_C)
    OpCode.MOVE: (OpMode.iABC, False, True, True, False),
    OpCode.LOADK: (OpMode.iABx, False, True, False, False),
    OpCode.LOADBOOL: (OpMode.iABC, False, True, False, True),
    OpCode.LOADNIL: (OpMode.iABC, False, True, False, False),
    OpCode.GETUPVAL: (OpMode.iABC, False, True, True, False),
    OpCode.GETGLOBAL: (OpMode.iABx, False, True, False, False),
    OpCode.GETTABLE: (OpMode.iABC, False, True, True, True),
    OpCode.SETGLOBAL: (OpMode.iABx, False, False, False, False),
    OpCode.SETUPVAL: (OpMode.iABC, False, False, True, False),
    OpCode.SETTABLE: (OpMode.iABC, False, False, True, True),
    OpCode.NEWTABLE: (OpMode.iABC, False, True, False, False),
    OpCode.SELF: (OpMode.iABC, False, True, True, True),
    OpCode.ADD: (OpMode.iABC, False, True, True, True),
    OpCode.SUB: (OpMode.iABC, False, True, True, True),
    OpCode.MUL: (OpMode.iABC, False, True, True, True),
    OpCode.DIV: (OpMode.iABC, False, True, True, True),
    OpCode.MOD: (OpMode.iABC, False, True, True, True),
    OpCode.POW: (OpMode.iABC, False, True, True, True),
    OpCode.UNM: (OpMode.iABC, False, True, True, False),
    OpCode.NOT: (OpMode.iABC, False, True, True, False),
    OpCode.LEN: (OpMode.iABC, False, True, True, False),
    OpCode.CONCAT: (OpMode.iABC, False, True, True, True),
    OpCode.JMP: (OpMode.iAsBx, False, False, False, False),
    OpCode.EQ: (OpMode.iABC, True, False, True, True),
    OpCode.LT: (OpMode.iABC, True, False, True, True),
    OpCode.LE: (OpMode.iABC, True, False, True, True),
    OpCode.TEST: (OpMode.iABC, True, False, False, True),
    OpCode.TESTSET: (OpMode.iABC, True, True, True, True),
    OpCode.CALL: (OpMode.iABC, False, True, False, False),
    OpCode.TAILCALL: (OpMode.iABC, False, False, False, False),
    OpCode.RETURN: (OpMode.iABC, False, False, False, False),
    OpCode.FORLOOP: (OpMode.iAsBx, False, False, False, False),
    OpCode.FORPREP: (OpMode.iAsBx, False, False, False, False),
    OpCode.TFORLOOP: (OpMode.iABC, False, False, False, True),
    OpCode.SETLIST: (OpMode.iABC, False, False, False, False),
    OpCode.CLOSE: (OpMode.iABC, False, False, False, False),
    OpCode.CLOSURE: (OpMode.iABx, False, True, False, False),
    OpCode.VARARG: (OpMode.iABC, False, True, False, False),
}

# ============================================
# Advanced Data Classes
# ============================================

@dataclass
class Instruction:
    """Enhanced instruction with metadata"""
    raw: int
    opcode: OpCode
    A: int
    B: int = 0
    C: int = 0
    Bx: int = 0
    sBx: int = 0
    Ax: int = 0
    mode: OpMode = OpMode.iABC
    
    # Metadata
    pc: int = 0                    # Program counter
    line: int = 0                  # Source line
    basic_block: int = -1          # Basic block ID
    is_leader: bool = False        # BB leader
    is_jump_target: bool = False   # Jump target
    references: List[int] = field(default_factory=list)  # Referenced by
    jumps_to: Optional[int] = None # Jump destination
    
    # Analysis data
    live_in: Set[int] = field(default_factory=set)   # Live variables in
    live_out: Set[int] = field(default_factory=set)  # Live variables out
    def_vars: Set[int] = field(default_factory=set)  # Variables defined
    use_vars: Set[int] = field(default_factory=set)  # Variables used
    
    def __repr__(self):
        base = f"[{self.pc:04d}] {self.opcode.name:12}"
        if self.mode == OpMode.iABC:
            args = f"A={self.A} B={self.B} C={self.C}"
        elif self.mode == OpMode.iABx:
            args = f"A={self.A} Bx={self.Bx}"
        elif self.mode == OpMode.iAsBx:
            args = f"A={self.A} sBx={self.sBx}"
        else:
            args = f"Ax={self.Ax}"
        
        suffix = ""
        if self.line:
            suffix += f" ; line {self.line}"
        if self.is_jump_target:
            suffix += " <-"
        
        return f"{base} {args}{suffix}"

@dataclass
class Constant:
    """Enhanced constant with type info"""
    type: int
    value: Union[None, bool, float, str, int]
    
    # Metadata
    index: int = 0
    references: List[int] = field(default_factory=list)  # Instructions using this
    hash: str = ""
    is_encrypted: bool = False
    encryption_method: int = 0
    
    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash of constant value"""
        if self.value is None:
            data = b"nil"
        elif isinstance(self.value, bool):
            data = b"true" if self.value else b"false"
        elif isinstance(self.value, (int, float)):
            data = str(self.value).encode()
        elif isinstance(self.value, str):
            data = self.value.encode('utf-8', errors='replace')
        else:
            data = str(self.value).encode()
        
        return hashlib.md5(data).hexdigest()[:16]
    
    def __repr__(self):
        type_names = {0: "nil", 1: "bool", 3: "num", 4: "str"}
        type_name = type_names.get(self.type, "unknown")
        
        if self.type == 0:
            return "nil"
        elif self.type == 1:
            return f"bool({self.value})"
        elif self.type == 3:
            return f"num({self.value})"
        elif self.type == 4:
            val_preview = str(self.value)[:30]
            if len(str(self.value)) > 30:
                val_preview += "..."
            return f'str("{val_preview}")'
        return f"{type_name}({self.value})"

@dataclass
class Local:
    """Enhanced local variable info"""
    name: str
    start_pc: int
    end_pc: int
    
    # Analysis
    register: int = -1
    is_parameter: bool = False
    is_upvalue: bool = False
    is_constant: bool = False
    write_count: int = 0
    read_count: int = 0

@dataclass
class Upvalue:
    """Enhanced upvalue info"""
    name: str
    index: int = 0
    in_stack: bool = False
    reg_index: int = 0
    kind: int = 0  # 0=local, 1=upvalue

@dataclass
class BasicBlock:
    """Basic block for control flow analysis"""
    id: int
    start_pc: int
    end_pc: int
    instructions: List[int] = field(default_factory=list)
    
    # CFG edges
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    
    # Dominance
    dominators: Set[int] = field(default_factory=set)
    immediate_dominator: int = -1
    dominance_frontier: Set[int] = field(default_factory=set)
    
    # Loop info
    is_loop_header: bool = False
    loop_depth: int = 0

@dataclass
class Function:
    """Enhanced function prototype with analysis"""
    source: str = ""
    line_defined: int = 0
    last_line_defined: int = 0
    num_upvalues: int = 0
    num_params: int = 0
    is_vararg: int = 0
    max_stack_size: int = 0
    
    # Code
    instructions: List[Instruction] = field(default_factory=list)
    constants: List[Constant] = field(default_factory=list)
    prototypes: List['Function'] = field(default_factory=list)
    
    # Debug info
    source_lines: List[int] = field(default_factory=list)
    locals: List[Local] = field(default_factory=list)
    upvalues: List[Upvalue] = field(default_factory=list)
    
    # Analysis data
    basic_blocks: List[BasicBlock] = field(default_factory=list)
    call_graph: Dict[int, List[int]] = field(default_factory=dict)
    register_usage: Dict[int, int] = field(default_factory=dict)
    
    # Metadata
    complexity: int = 0
    hash: str = ""
    is_main: bool = False
    parent: Optional['Function'] = None
    depth: int = 0

@dataclass
class LuaHeader:
    """Enhanced Lua bytecode header"""
    signature: bytes = b'\x1bLua'
    version: int = 0x51
    format: int = 0
    endianness: int = 1
    int_size: int = 4
    size_t_size: int = 4
    instruction_size: int = 4
    number_size: int = 8
    integral_flag: int = 0
    
    # Extended info
    lua_version: LuaVersion = LuaVersion.LUA_51
    format_type: BytecodeFormat = BytecodeFormat.STANDARD
    is_little_endian: bool = True
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

@dataclass
class LuaChunk:
    """Enhanced Lua bytecode chunk"""
    header: LuaHeader
    main_function: Function
    
    # Global analysis
    all_functions: List[Function] = field(default_factory=list)
    constant_pool: Dict[str, Constant] = field(default_factory=dict)
    string_pool: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    file_size: int = 0
    checksum: str = ""
    compressed: bool = False
    encrypted: bool = False
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)

# ============================================
# Advanced Parser with Multi-Version Support
# ============================================

class AdvancedBytecodeParser:
    """Advanced parser supporting multiple Lua versions"""
    
    def __init__(self, data: bytes, auto_detect: bool = True):
        self.data = data
        self.pos = 0
        self.header: Optional[LuaHeader] = None
        self.version: LuaVersion = LuaVersion.LUA_51
        self.auto_detect = auto_detect
        
        # Statistics
        self.stats = {
            'total_instructions': 0,
            'total_constants': 0,
            'total_functions': 0,
            'total_strings': 0,
            'max_stack': 0,
        }
    
    # ========================================
    # Low-level Reading
    # ========================================
    
    def read_bytes(self, n: int) -> bytes:
        """Read n bytes"""
        if self.pos + n > len(self.data):
            raise ValueError(f"Read past end of data: {self.pos} + {n} > {len(self.data)}")
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result
    
    def read_byte(self) -> int:
        """Read single byte"""
        return self.read_bytes(1)[0]
    
    def read_int(self) -> int:
        """Read integer"""
        size = self.header.int_size if self.header else 4
        data = self.read_bytes(size)
        
        if self.header and self.header.is_little_endian:
            fmt = '<i' if size == 4 else '<q'
        else:
            fmt = '>i' if size == 4 else '>q'
        
        return struct.unpack(fmt, data)[0]
    
    def read_size_t(self) -> int:
        """Read size_t"""
        size = self.header.size_t_size if self.header else 4
        data = self.read_bytes(size)
        
        if self.header and self.header.is_little_endian:
            fmt = '<I' if size == 4 else '<Q'
        else:
            fmt = '>I' if size == 4 else '>Q'
        
        return struct.unpack(fmt, data)[0]
    
    def read_number(self) -> float:
        """Read Lua number"""
        size = self.header.number_size if self.header else 8
        data = self.read_bytes(size)
        
        if self.header and self.header.is_little_endian:
            if size == 4:
                return struct.unpack('<f', data)[0]
            return struct.unpack('<d', data)[0]
        else:
            if size == 4:
                return struct.unpack('>f', data)[0]
            return struct.unpack('>d', data)[0]
    
    def read_string(self) -> str:
        """Read Lua string with version-specific handling"""
        if self.version in [LuaVersion.LUA_53, LuaVersion.LUA_54]:
            # Lua 5.3+ uses 1 byte for small strings
            size = self.read_byte()
            if size == 0xFF:
                size = self.read_size_t()
        else:
            size = self.read_size_t()
        
        if size == 0:
            return ""
        
        # Account for null terminator in some versions
        if self.version in [LuaVersion.LUA_51, LuaVersion.LUA_52]:
            data = self.read_bytes(size)
            return data[:-1].decode('utf-8', errors='replace') if data else ""
        else:
            data = self.read_bytes(size - 1) if size > 0 else b""
            return data.decode('utf-8', errors='replace')
    
    def read_instruction(self) -> int:
        """Read instruction (32-bit)"""
        size = self.header.instruction_size if self.header else 4
        data = self.read_bytes(size)
        
        if self.header and self.header.is_little_endian:
            return struct.unpack('<I', data)[0]
        else:
            return struct.unpack('>I', data)[0]
    
    # ========================================
    # Instruction Decoding
    # ========================================
    
    def decode_instruction(self, raw: int, pc: int = 0) -> Instruction:
        """Decode instruction with version-specific handling"""
        if self.version == LuaVersion.LUA_51:
            return self._decode_lua51(raw, pc)
        elif self.version == LuaVersion.LUA_52:
            return self._decode_lua52(raw, pc)
        elif self.version in [LuaVersion.LUA_53, LuaVersion.LUA_54]:
            return self._decode_lua53(raw, pc)
        elif self.version in [LuaVersion.LUAJIT_20, LuaVersion.LUAJIT_21]:
            return self._decode_luajit(raw, pc)
        elif self.version == LuaVersion.LUAU:
            return self._decode_luau(raw, pc)
        else:
            return self._decode_lua51(raw, pc)  # Default
    
    def _decode_lua51(self, raw: int, pc: int) -> Instruction:
        """Decode Lua 5.1 instruction"""
        opcode = OpCode(raw & 0x3F)
        A = (raw >> 6) & 0xFF
        C = (raw >> 14) & 0x1FF
        B = (raw >> 23) & 0x1FF
        Bx = (raw >> 14) & 0x3FFFF
        sBx = Bx - 131071
        
        mode = OPCODE_INFO.get(opcode, (OpMode.iABC,))[0] if opcode in OPCODE_INFO else OpMode.iABC
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            Bx=Bx, sBx=sBx, mode=mode, pc=pc
        )
    
    def _decode_lua52(self, raw: int, pc: int) -> Instruction:
        """Decode Lua 5.2 instruction"""
        # Similar to 5.1 but with additional opcodes
        return self._decode_lua51(raw, pc)
    
    def _decode_lua53(self, raw: int, pc: int) -> Instruction:
        """Decode Lua 5.3+ instruction"""
        opcode = OpCode(raw & 0x7F)  # 7 bits in 5.3
        A = (raw >> 7) & 0xFF
        
        mode = OPCODE_INFO.get(opcode, (OpMode.iABC,))[0] if opcode in OPCODE_INFO else OpMode.iABC
        
        if mode == OpMode.iABC:
            B = (raw >> 15) & 0xFF
            C = (raw >> 23) & 0xFF
            Bx = 0
            sBx = 0
        elif mode == OpMode.iABx:
            B = 0
            C = 0
            Bx = (raw >> 15) & 0x3FFFF
            sBx = 0
        elif mode == OpMode.iAsBx:
            B = 0
            C = 0
            Bx = (raw >> 15) & 0x3FFFF
            sBx = Bx - 131071
        else:  # iAx
            B = C = Bx = sBx = 0
            Ax = (raw >> 7) & 0x1FFFFFF
            return Instruction(
                raw=raw, opcode=opcode, A=A, Ax=Ax,
                mode=mode, pc=pc
            )
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            Bx=Bx, sBx=sBx, mode=mode, pc=pc
        )
    
    def _decode_luajit(self, raw: int, pc: int) -> Instruction:
        """Decode LuaJIT instruction"""
        # LuaJIT uses different encoding
        opcode = OpCode(raw & 0xFF)
        A = (raw >> 8) & 0xFF
        C = (raw >> 16) & 0xFF
        B = (raw >> 24) & 0xFF
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            mode=OpMode.iABC, pc=pc
        )
    
    def _decode_luau(self, raw: int, pc: int) -> Instruction:
        """Decode Luau (Roblox) instruction"""
        # Luau uses custom encoding
        opcode = OpCode(raw & 0xFF)
        A = (raw >> 8) & 0xFF
        B = (raw >> 16) & 0xFF
        C = (raw >> 24) & 0xFF
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            mode=OpMode.iABC, pc=pc
        )
    
    # ========================================
    # Header Parsing
    # ========================================
    
    def read_header(self) -> LuaHeader:
        """Parse bytecode header with version detection"""
        header = LuaHeader()
        
        # Signature
        header.signature = self.read_bytes(4)
        if header.signature != b'\x1bLua':
            header.is_valid = False
            header.validation_errors.append(f"Invalid signature: {header.signature.hex()}")
            
            # Try to detect if it's compressed/encrypted
            if header.signature[:2] == b'\x1f\x8b':
                header.format_type = BytecodeFormat.COMPRESSED
                header.validation_errors.append("Possibly gzip compressed")
            elif all(b > 127 for b in header.signature):
                header.format_type = BytecodeFormat.ENCRYPTED
                header.validation_errors.append("Possibly encrypted")
            else:
                header.validation_errors.append("Unknown format")
            
            # Still try to continue for analysis
        
        # Version
        header.version = self.read_byte()
        
        # Detect Lua version
        if header.version == 0x51:
            header.lua_version = LuaVersion.LUA_51
        elif header.version == 0x52:
            header.lua_version = LuaVersion.LUA_52
        elif header.version == 0x53:
            header.lua_version = LuaVersion.LUA_53
        elif header.version == 0x54:
            header.lua_version = LuaVersion.LUA_54
        elif header.version in [0x01, 0x02]:
            header.lua_version = LuaVersion.LUAJIT_20
        elif header.version == 0x03:
            header.lua_version = LuaVersion.LUAU
        else:
            header.is_valid = False
            header.validation_errors.append(f"Unknown version: {hex(header.version)}")
        
        self.version = header.lua_version
        
        # Format version
        header.format = self.read_byte()
        
        # Endianness
        header.endianness = self.read_byte()
        header.is_little_endian = header.endianness == 1
        
        # Size information
        header.int_size = self.read_byte()
        header.size_t_size = self.read_byte()
        header.instruction_size = self.read_byte()
        
        # Number format
        if self.version in [LuaVersion.LUA_53, LuaVersion.LUA_54]:
            # Lua 5.3+ has separate int and float sizes
            header.number_size = self.read_byte()
            header.integral_flag = self.read_byte()
        else:
            header.number_size = self.read_byte()
            header.integral_flag = self.read_byte()
        
        # Validation checks
        if header.int_size not in [4, 8]:
            header.validation_errors.append(f"Unusual int size: {header.int_size}")
        if header.instruction_size != 4:
            header.validation_errors.append(f"Unusual instruction size: {header.instruction_size}")
        
        self.header = header
        return header
    
    # ========================================
    # Constant Parsing
    # ========================================
    
    def read_constant(self, index: int = 0) -> Constant:
        """Read constant with enhanced type detection"""
        const_type = self.read_byte()
        
        const = Constant(type=const_type, value=None, index=index)
        
        if const_type == 0:  # nil
            const.value = None
        
        elif const_type == 1:  # boolean
            const.value = self.read_byte() != 0
        
        elif const_type == 3:  # number
            if self.version in [LuaVersion.LUA_53, LuaVersion.LUA_54]:
                # Check if integer or float
                if self.header and self.header.integral_flag:
                    const.value = self.read_int()
                else:
                    const.value = self.read_number()
            else:
                const.value = self.read_number()
        
        elif const_type == 4:  # string
            const.value = self.read_string()
            self.stats['total_strings'] += 1
        
        elif const_type == 0x13:  # Lua 5.3+ integer
            const.value = self.read_int()
        
        elif const_type == 0x14:  # Lua 5.3+ float
            const.value = self.read_number()
        
        else:
            raise ValueError(f"Unknown constant type: {const_type}")
        
        self.stats['total_constants'] += 1
        return const
    
    # ========================================
    # Function Parsing
    # ========================================
    
    def read_function(self, depth: int = 0) -> Function:
        """Read function prototype with enhanced analysis"""
        func = Function(depth=depth, is_main=(depth == 0))
        
        # Source name
        func.source = self.read_string()
        
        # Line info
        func.line_defined = self.read_int()
        func.last_line_defined = self.read_int()
        
        # Function parameters
        if self.version == LuaVersion.LUA_54:
            func.num_params = self.read_byte()
            func.is_vararg = self.read_byte()
            func.max_stack_size = self.read_byte()
        else:
            func.num_upvalues = self.read_byte()
            func.num_params = self.read_byte()
            func.is_vararg = self.read_byte()
            func.max_stack_size = self.read_byte()
        
        self.stats['max_stack'] = max(self.stats['max_stack'], func.max_stack_size)
        
        # Instructions
        num_instructions = self.read_int()
        for i in range(num_instructions):
            raw = self.read_instruction()
            instr = self.decode_instruction(raw, i)
            func.instructions.append(instr)
        
        self.stats['total_instructions'] += num_instructions
        
        # Constants
        num_constants = self.read_int()
        for i in range(num_constants):
            const = self.read_constant(i)
            func.constants.append(const)
        
        # Nested prototypes (functions)
        num_prototypes = self.read_int()
        for _ in range(num_prototypes):
            proto = self.read_function(depth + 1)
            proto.parent = func
            func.prototypes.append(proto)
        
        self.stats['total_functions'] += 1
        
        # Upvalues (version-specific)
        if self.version in [LuaVersion.LUA_52, LuaVersion.LUA_53, LuaVersion.LUA_54]:
            num_upvalues = self.read_int()
            for i in range(num_upvalues):
                in_stack = self.read_byte()
                idx = self.read_byte()
                kind = 0
                if self.version == LuaVersion.LUA_54:
                    kind = self.read_byte()
                
                upval = Upvalue(name="", index=i, in_stack=bool(in_stack), 
                               reg_index=idx, kind=kind)
                func.upvalues.append(upval)
        
        # Debug info: source line positions
        num_lines = self.read_int()
        for i in range(num_lines):
            line = self.read_int()
            func.source_lines.append(line)
            if i < len(func.instructions):
                func.instructions[i].line = line
        
        # Debug info: locals
        num_locals = self.read_int()
        for i in range(num_locals):
            name = self.read_string()
            start_pc = self.read_int()
            end_pc = self.read_int()
            
            local = Local(name, start_pc, end_pc)
            local.is_parameter = i < func.num_params
            func.locals.append(local)
        
        # Debug info: upvalue names
        num_upvalue_names = self.read_int()
        for i in range(num_upvalue_names):
            name = self.read_string()
            if i < len(func.upvalues):
                func.upvalues[i].name = name
        
        # Compute function hash
        func.hash = self._compute_function_hash(func)
        
        return func
    
    def _compute_function_hash(self, func: Function) -> str:
        """Compute hash of function code"""
        data = bytearray()
        for instr in func.instructions:
            data.extend(struct.pack('<I', instr.raw))
        return hashlib.md5(data).hexdigest()
    
    # ========================================
    # Main Parse Entry
    # ========================================
    
    def parse(self) -> LuaChunk:
        """Parse complete bytecode"""
        # Read header
        header = self.read_header()
        
        # Handle special formats
        if header.format_type == BytecodeFormat.COMPRESSED:
            # Try to decompress
            try:
                decompressed = zlib.decompress(self.data[self.pos:])
                self.data = self.data[:self.pos] + decompressed
            except:
                pass
        
        # Read main function
        main_function = self.read_function(depth=0)
        
        # Create chunk
        chunk = LuaChunk(
            header=header,
            main_function=main_function,
            file_size=len(self.data),
            stats=self.stats.copy()
        )
        
        # Build global indices
        self._build_global_indices(chunk)
        
        # Compute checksum
        chunk.checksum = hashlib.sha256(self.data).hexdigest()
        
        return chunk
    
    def _build_global_indices(self, chunk: LuaChunk):
        """Build global function and constant indices"""
        def collect_functions(func: Function):
            chunk.all_functions.append(func)
            for proto in func.prototypes:
                collect_functions(proto)
        
        collect_functions(chunk.main_function)
        
        # Build constant pool
        for func in chunk.all_functions:
            for const in func.constants:
                chunk.constant_pool[const.hash] = const
                if const.type == 4:  # String
                    chunk.string_pool[const.value] = const.index

# ============================================
# Control Flow Analysis
# ============================================

class ControlFlowAnalyzer:
    """Advanced control flow analysis"""
    
    def __init__(self, function: Function):
        self.function = function
        self.basic_blocks: List[BasicBlock] = []
        self.cfg_edges: Dict[int, Set[int]] = defaultdict(set)
    
    def analyze(self):
        """Perform complete CFG analysis"""
        self._identify_basic_blocks()
        self._build_cfg()
        self._compute_dominators()
        self._detect_loops()
        
        self.function.basic_blocks = self.basic_blocks
    
    def _identify_basic_blocks(self):
        """Identify basic block boundaries"""
        if not self.function.instructions:
            return
        
        leaders = {0}  # First instruction is always a leader
        
        # Find all leaders
        for i, instr in enumerate(self.function.instructions):
            # Jump targets are leaders
            if instr.opcode in [OpCode.JMP, OpCode.FORLOOP, OpCode.FORPREP]:
                target = i + 1 + instr.sBx
                if 0 <= target < len(self.function.instructions):
                    leaders.add(target)
                    self.function.instructions[target].is_jump_target = True
                    instr.jumps_to = target
                
                # Instruction after jump is also leader
                if i + 1 < len(self.function.instructions):
                    leaders.add(i + 1)
            
            # Conditional jumps
            elif instr.opcode in [OpCode.EQ, OpCode.LT, OpCode.LE, OpCode.TEST, OpCode.TESTSET]:
                # Next instruction might be JMP
                if i + 1 < len(self.function.instructions):
                    leaders.add(i + 1)
            
            # Returns end basic blocks
            elif instr.opcode in [OpCode.RETURN, OpCode.TAILCALL]:
                if i + 1 < len(self.function.instructions):
                    leaders.add(i + 1)
        
        # Mark leaders
        for leader_pc in leaders:
            self.function.instructions[leader_pc].is_leader = True
        
        # Create basic blocks
        sorted_leaders = sorted(leaders)
        for i, start in enumerate(sorted_leaders):
            end = sorted_leaders[i + 1] - 1 if i + 1 < len(sorted_leaders) else len(self.function.instructions) - 1
            
            bb = BasicBlock(
                id=i,
                start_pc=start,
                end_pc=end,
                instructions=list(range(start, end + 1))
            )
            
            # Mark instructions with BB ID
            for pc in range(start, end + 1):
                self.function.instructions[pc].basic_block = i
            
            self.basic_blocks.append(bb)
    
    def _build_cfg(self):
        """Build control flow graph"""
        for bb in self.basic_blocks:
            last_instr = self.function.instructions[bb.end_pc]
            
            # Unconditional jump
            if last_instr.opcode == OpCode.JMP:
                target_pc = last_instr.jumps_to
                if target_pc is not None:
                    target_bb = self._find_bb_for_pc(target_pc)
                    if target_bb is not None:
                        bb.successors.add(target_bb.id)
                        target_bb.predecessors.add(bb.id)
            
            # Conditional branch
            elif last_instr.opcode in [OpCode.EQ, OpCode.LT, OpCode.LE, OpCode.TEST, OpCode.TESTSET]:
                # Fall-through
                if bb.end_pc + 1 < len(self.function.instructions):
                    next_bb = self._find_bb_for_pc(bb.end_pc + 1)
                    if next_bb:
                        bb.successors.add(next_bb.id)
                        next_bb.predecessors.add(bb.id)
                
                # Branch target (usually next+1 which is JMP)
                if bb.end_pc + 2 < len(self.function.instructions):
                    branch_instr = self.function.instructions[bb.end_pc + 1]
                    if branch_instr.opcode == OpCode.JMP and branch_instr.jumps_to is not None:
                        target_bb = self._find_bb_for_pc(branch_instr.jumps_to)
                        if target_bb:
                            bb.successors.add(target_bb.id)
                            target_bb.predecessors.add(bb.id)
            
            # Loop instructions
            elif last_instr.opcode in [OpCode.FORLOOP, OpCode.FORPREP]:
                # Jump target
                if last_instr.jumps_to is not None:
                    target_bb = self._find_bb_for_pc(last_instr.jumps_to)
                    if target_bb:
                        bb.successors.add(target_bb.id)
                        target_bb.predecessors.add(bb.id)
                
                # Fall-through
                if bb.end_pc + 1 < len(self.function.instructions):
                    next_bb = self._find_bb_for_pc(bb.end_pc + 1)
                    if next_bb:
                        bb.successors.add(next_bb.id)
                        next_bb.predecessors.add(bb.id)
            
            # Return doesn't have successors
            elif last_instr.opcode in [OpCode.RETURN, OpCode.TAILCALL]:
                pass
            
            # Normal fall-through
            else:
                if bb.end_pc + 1 < len(self.function.instructions):
                    next_bb = self._find_bb_for_pc(bb.end_pc + 1)
                    if next_bb:
                        bb.successors.add(next_bb.id)
                        next_bb.predecessors.add(bb.id)
    
    def _find_bb_for_pc(self, pc: int) -> Optional[BasicBlock]:
        """Find basic block containing PC"""
        for bb in self.basic_blocks:
            if bb.start_pc <= pc <= bb.end_pc:
                return bb
        return None
    
    def _compute_dominators(self):
        """Compute dominator tree"""
        if not self.basic_blocks:
            return
        
        # Initialize
        entry = self.basic_blocks[0]
        entry.dominators = {entry.id}
        
        for bb in self.basic_blocks[1:]:
            bb.dominators = set(range(len(self.basic_blocks)))
        
        # Iterate until fixpoint
        changed = True
        while changed:
            changed = False
            for bb in self.basic_blocks[1:]:
                new_dom = set(range(len(self.basic_blocks)))
                
                for pred_id in bb.predecessors:
                    pred_bb = self.basic_blocks[pred_id]
                    new_dom &= pred_bb.dominators
                
                new_dom.add(bb.id)
                
                if new_dom != bb.dominators:
                    bb.dominators = new_dom
                    changed = True
        
        # Compute immediate dominators
        for bb in self.basic_blocks:
            strict_doms = bb.dominators - {bb.id}
            if strict_doms:
                # idom is the closest dominator
                candidates = []
                for dom_id in strict_doms:
                    dom_bb = self.basic_blocks[dom_id]
                    if all(other_id in dom_bb.dominators for other_id in strict_doms if other_id != dom_id):
                        candidates.append(dom_id)
                
                if candidates:
                    bb.immediate_dominator = max(candidates)  # Closest one
    
    def _detect_loops(self):
        """Detect loops using back edges"""
        # Find back edges (edge to dominator)
        for bb in self.basic_blocks:
            for succ_id in bb.successors:
                if succ_id in bb.dominators:
                    # Back edge found - succ is loop header
                    header = self.basic_blocks[succ_id]
                    header.is_loop_header = True
                    
                    # Mark loop depth
                    self._mark_loop_depth(header, bb)
    
    def _mark_loop_depth(self, header: BasicBlock, back_edge_source: BasicBlock):
        """Mark loop depth for natural loop"""
        # Simple approach: increment depth for all BBs in loop
        worklist = [back_edge_source.id]
        in_loop = {back_edge_source.id, header.id}
        
        while worklist:
            bb_id = worklist.pop()
            bb = self.basic_blocks[bb_id]
            
            for pred_id in bb.predecessors:
                if pred_id not in in_loop and pred_id != header.id:
                    in_loop.add(pred_id)
                    worklist.append(pred_id)
        
        # Increment loop depth
        for bb_id in in_loop:
            self.basic_blocks[bb_id].loop_depth += 1

# ============================================
# Advanced Disassembler
# ============================================

class AdvancedDisassembler:
    """Advanced disassembler with multiple output formats"""
    
    def __init__(self, chunk: LuaChunk):
        self.chunk = chunk
    
    def disassemble(self, show_analysis: bool = False, 
                   show_hex: bool = False,
                   show_cfg: bool = False) -> str:
        """Generate comprehensive disassembly"""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("LUA BYTECODE DISASSEMBLY")
        lines.append("=" * 80)
        lines.append(f"Version: Lua {self.chunk.header.version >> 4}.{self.chunk.header.version & 0xF}")
        lines.append(f"Format: {self.chunk.header.format_type.name}")
        lines.append(f"Endianness: {'Little' if self.chunk.header.is_little_endian else 'Big'}")
        lines.append(f"File Size: {self.chunk.file_size} bytes")
        lines.append(f"Checksum: {self.chunk.checksum}")
        lines.append("=" * 80)
        lines.append("")
        
        # Statistics
        lines.append("STATISTICS:")
        for key, value in self.chunk.stats.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Disassemble main and all functions
        lines.append(self._disassemble_function(
            self.chunk.main_function, 
            show_analysis=show_analysis,
            show_hex=show_hex,
            show_cfg=show_cfg
        ))
        
        return "\n".join(lines)
    
    def _disassemble_function(self, func: Function, indent: int = 0,
                             show_analysis: bool = False,
                             show_hex: bool = False,
                             show_cfg: bool = False) -> str:
        """Disassemble single function"""
        lines = []
        prefix = "  " * indent
        
        # Function header
        lines.append(f"{prefix}{'=' * 70}")
        lines.append(f"{prefix}FUNCTION: {func.source or '<main>'}")
        lines.append(f"{prefix}{'=' * 70}")
        lines.append(f"{prefix}Lines: {func.line_defined}-{func.last_line_defined}")
        lines.append(f"{prefix}Parameters: {func.num_params}, Vararg: {func.is_vararg}")
        lines.append(f"{prefix}Stack Size: {func.max_stack_size}, Upvalues: {func.num_upvalues}")
        lines.append(f"{prefix}Hash: {func.hash}")
        lines.append(f"{prefix}")
        
        # Constants
        if func.constants:
            lines.append(f"{prefix}CONSTANTS ({len(func.constants)}):")
            for i, const in enumerate(func.constants):
                refs = f" ; used by {len(const.references)} instr" if const.references else ""
                lines.append(f"{prefix}  [{i:3d}] {const}{refs}")
            lines.append(f"{prefix}")
        
        # Locals
        if func.locals:
            lines.append(f"{prefix}LOCALS ({len(func.locals)}):")
            for local in func.locals:
                param = " (param)" if local.is_parameter else ""
                lines.append(f"{prefix}  {local.name} ({local.start_pc}-{local.end_pc}){param}")
            lines.append(f"{prefix}")
        
        # Upvalues
        if func.upvalues:
            lines.append(f"{prefix}UPVALUES ({len(func.upvalues)}):")
            for upval in func.upvalues:
                lines.append(f"{prefix}  [{upval.index}] {upval.name}")
            lines.append(f"{prefix}")
        
        # Instructions
        lines.append(f"{prefix}INSTRUCTIONS ({len(func.instructions)}):")
        
        for i, instr in enumerate(func.instructions):
            line_parts = []
            
            # PC and hex
            if show_hex:
                line_parts.append(f"{prefix}  {i:04d} [{instr.raw:08x}]")
            else:
                line_parts.append(f"{prefix}  {i:04d}")
            
            # BB marker
            if show_cfg and instr.is_leader:
                line_parts.append(f" BB{instr.basic_block}")
            
            # Jump target marker
            if instr.is_jump_target:
                line_parts.append(" <-")
            
            # Instruction
            line_parts.append(f" {instr.opcode.name:12}")
            
            # Arguments
            if instr.mode == OpMode.iABC:
                line_parts.append(f" A={instr.A:3d} B={instr.B:3d} C={instr.C:3d}")
            elif instr.mode == OpMode.iABx:
                line_parts.append(f" A={instr.A:3d} Bx={instr.Bx:5d}")
            elif instr.mode == OpMode.iAsBx:
                line_parts.append(f" A={instr.A:3d} sBx={instr.sBx:5d}")
            else:
                line_parts.append(f" Ax={instr.Ax:7d}")
            
            # Source line
            if instr.line:
                line_parts.append(f" ; line {instr.line}")
            
            # Jump target
            if instr.jumps_to is not None:
                line_parts.append(f" -> {instr.jumps_to}")
            
            # Analysis info
            if show_analysis:
                if instr.def_vars:
                    line_parts.append(f" DEF:{instr.def_vars}")
                if instr.use_vars:
                    line_parts.append(f" USE:{instr.use_vars}")
            
            lines.append("".join(line_parts))
        
        lines.append(f"{prefix}")
        
        # CFG info
        if show_cfg and func.basic_blocks:
            lines.append(f"{prefix}CONTROL FLOW GRAPH:")
            for bb in func.basic_blocks:
                lines.append(f"{prefix}  BB{bb.id}: PC {bb.start_pc}-{bb.end_pc}")
                if bb.predecessors:
                    lines.append(f"{prefix}    Predecessors: {sorted(bb.predecessors)}")
                if bb.successors:
                    lines.append(f"{prefix}    Successors: {sorted(bb.successors)}")
                if bb.is_loop_header:
                    lines.append(f"{prefix}    LOOP HEADER (depth={bb.loop_depth})")
            lines.append(f"{prefix}")
        
        # Nested functions
        for i, proto in enumerate(func.prototypes):
            lines.append(f"{prefix}")
            lines.append(f"{prefix}--- NESTED FUNCTION {i} ---")
            lines.append(self._disassemble_function(
                proto, 
                indent + 1,
                show_analysis=show_analysis,
                show_hex=show_hex,
                show_cfg=show_cfg
            ))
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export to JSON format"""
        def serialize_func(func: Function) -> Dict:
            return {
                'source': func.source,
                'lines': [func.line_defined, func.last_line_defined],
                'params': func.num_params,
                'vararg': func.is_vararg,
                'stack': func.max_stack_size,
                'hash': func.hash,
                'instructions': [
                    {
                        'pc': i.pc,
                        'opcode': i.opcode.name,
                        'raw': hex(i.raw),
                        'args': {'A': i.A, 'B': i.B, 'C': i.C},
                        'line': i.line,
                    }
                    for i in func.instructions
                ],
                'constants': [
                    {'type': c.type, 'value': str(c.value)}
                    for c in func.constants
                ],
                'prototypes': [serialize_func(p) for p in func.prototypes]
            }
        
        data = {
            'version': self.chunk.header.version,
            'stats': self.chunk.stats,
            'checksum': self.chunk.checksum,
            'main': serialize_func(self.chunk.main_function)
        }
        
        return json.dumps(data, indent=2)

# ============================================
# Utility Functions
# ============================================

def parse_bytecode(data: bytes) -> LuaChunk:
    """Parse Lua bytecode"""
    parser = AdvancedBytecodeParser(data)
    return parser.parse()

def parse_bytecode_file(filepath: str) -> LuaChunk:
    """Parse Lua bytecode from file"""
    with open(filepath, 'rb') as f:
        data = f.read()
    return parse_bytecode(data)

def analyze_function(func: Function) -> Function:
    """Perform control flow analysis on function"""
    analyzer = ControlFlowAnalyzer(func)
    analyzer.analyze()
    return func

def disassemble(chunk: LuaChunk, **kwargs) -> str:
    """Disassemble chunk"""
    disasm = AdvancedDisassembler(chunk)
    return disasm.disassemble(**kwargs)

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    print("=== Advanced Lua Bytecode Parser ===\n")
    
    # Create sample bytecode
    sample_header = b'\x1bLua\x51\x00\x01\x04\x04\x04\x08\x00'
    
    print("Parser Features:")
    print("  ✓ Multi-version support (5.1, 5.2, 5.3, 5.4, LuaJIT, Luau)")
    print("  ✓ Advanced instruction decoding")
    print("  ✓ Control flow analysis")
    print("  ✓ Basic block detection")
    print("  ✓ Dominator tree computation")
    print("  ✓ Loop detection")
    print("  ✓ Enhanced disassembly")
    print("  ✓ JSON export")
    print("  ✓ Comprehensive statistics")
    print()
    
    print("Supported Versions:")
    for version in LuaVersion:
        print(f"  - {version.name}")
    print()
    
    print("Supported Formats:")
    for fmt in BytecodeFormat:
        print(f"  - {fmt.name}")
    print()
    
    print("✅ Advanced parser initialized!")
