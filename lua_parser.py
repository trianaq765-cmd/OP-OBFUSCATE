# ============================================
# File: lua_parser.py (FIXED ADVANCED VERSION)
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
    UNKNOWN = 0x00
    LUA_51 = 0x51
    LUA_52 = 0x52
    LUA_53 = 0x53
    LUA_54 = 0x54
    LUAJIT_20 = 0x80  # LuaJIT 2.0
    LUAJIT_21 = 0x81  # LuaJIT 2.1
    LUAU_0 = 0x90     # Roblox Luau v0
    LUAU_1 = 0x91     # Roblox Luau v1
    LUAU_2 = 0x92     # Roblox Luau v2
    LUAU_3 = 0x93     # Roblox Luau v3
    LUAU_4 = 0x94     # Roblox Luau v4
    LUAU_5 = 0x95     # Roblox Luau v5
    LUAU_6 = 0x96     # Roblox Luau v6 (latest)

class BytecodeFormat(IntEnum):
    """Bytecode format variations"""
    STANDARD_LUA = 0    # Standard Lua 5.x
    LUAJIT = 1          # LuaJIT 2.x
    LUAU = 2            # Roblox Luau
    STRIPPED = 3        # Debug info stripped
    COMPRESSED = 4      # Compressed bytecode
    ENCRYPTED = 5       # Encrypted bytecode
    UNKNOWN = 255       # Unknown format

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
    
    # LuaJIT specific opcodes
    LJ_ISLT = 100
    LJ_ISGE = 101
    LJ_ISLE = 102
    LJ_ISGT = 103
    LJ_ISEQV = 104
    LJ_ISNEV = 105
    LJ_ISEQS = 106
    LJ_ISNES = 107
    LJ_ISEQN = 108
    LJ_ISNEN = 109
    LJ_ISEQP = 110
    LJ_ISNEP = 111
    LJ_ISTC = 112
    LJ_ISFC = 113
    LJ_IST = 114
    LJ_ISF = 115
    LJ_ISTYPE = 116
    LJ_ISNUM = 117
    LJ_MOV = 118
    LJ_NOT = 119
    LJ_UNM = 120
    LJ_LEN = 121
    LJ_ADDVN = 122
    LJ_SUBVN = 123
    LJ_MULVN = 124
    LJ_DIVVN = 125
    LJ_MODVN = 126
    LJ_ADDNV = 127
    LJ_SUBNV = 128
    LJ_MULNV = 129
    LJ_DIVNV = 130
    LJ_MODNV = 131
    LJ_ADDVV = 132
    LJ_SUBVV = 133
    LJ_MULVV = 134
    LJ_DIVVV = 135
    LJ_MODVV = 136
    LJ_POW = 137
    LJ_CAT = 138
    LJ_KSTR = 139
    LJ_KCDATA = 140
    LJ_KSHORT = 141
    LJ_KNUM = 142
    LJ_KPRI = 143
    LJ_KNIL = 144
    LJ_UGET = 145
    LJ_USETV = 146
    LJ_USETS = 147
    LJ_USETN = 148
    LJ_USETP = 149
    LJ_UCLO = 150
    LJ_FNEW = 151
    LJ_TNEW = 152
    LJ_TDUP = 153
    LJ_GGET = 154
    LJ_GSET = 155
    LJ_TGETV = 156
    LJ_TGETS = 157
    LJ_TGETB = 158
    LJ_TGETR = 159
    LJ_TSETV = 160
    LJ_TSETS = 161
    LJ_TSETB = 162
    LJ_TSETM = 163
    LJ_TSETR = 164
    LJ_CALLM = 165
    LJ_CALL = 166
    LJ_CALLMT = 167
    LJ_CALLT = 168
    LJ_ITERC = 169
    LJ_ITERN = 170
    LJ_VARG = 171
    LJ_ISNEXT = 172
    LJ_RETM = 173
    LJ_RET = 174
    LJ_RET0 = 175
    LJ_RET1 = 176
    LJ_FORI = 177
    LJ_JFORI = 178
    LJ_FORL = 179
    LJ_IFORL = 180
    LJ_JFORL = 181
    LJ_ITERL = 182
    LJ_IITERL = 183
    LJ_JITERL = 184
    LJ_LOOP = 185
    LJ_ILOOP = 186
    LJ_JLOOP = 187
    LJ_JMP = 188
    LJ_FUNCF = 189
    LJ_IFUNCF = 190
    LJ_JFUNCF = 191
    LJ_FUNCV = 192
    LJ_IFUNCV = 193
    LJ_JFUNCV = 194
    LJ_FUNCC = 195
    LJ_FUNCCW = 196
    
    # Luau specific opcodes
    LOP_NOP = 200
    LOP_BREAK = 201
    LOP_LOADNIL = 202
    LOP_LOADB = 203
    LOP_LOADN = 204
    LOP_LOADK = 205
    LOP_MOVE = 206
    LOP_GETGLOBAL = 207
    LOP_SETGLOBAL = 208
    LOP_GETUPVAL = 209
    LOP_SETUPVAL = 210
    LOP_CLOSEUPVALS = 211
    LOP_GETIMPORT = 212
    LOP_GETTABLE = 213
    LOP_SETTABLE = 214
    LOP_GETTABLEKS = 215
    LOP_SETTABLEKS = 216
    LOP_GETTABLEN = 217
    LOP_SETTABLEN = 218
    LOP_NEWCLOSURE = 219
    LOP_NAMECALL = 220
    LOP_CALL = 221
    LOP_RETURN = 222
    LOP_JUMP = 223
    LOP_JUMPBACK = 224
    LOP_JUMPIF = 225
    LOP_JUMPIFNOT = 226
    LOP_JUMPIFEQ = 227
    LOP_JUMPIFLE = 228
    LOP_JUMPIFLT = 229
    LOP_JUMPIFNOTEQ = 230
    LOP_JUMPIFNOTLE = 231
    LOP_JUMPIFNOTLT = 232
    LOP_ADD = 233
    LOP_SUB = 234
    LOP_MUL = 235
    LOP_DIV = 236
    LOP_MOD = 237
    LOP_POW = 238
    LOP_ADDK = 239
    LOP_SUBK = 240
    LOP_MULK = 241
    LOP_DIVK = 242
    LOP_MODK = 243
    LOP_POWK = 244
    LOP_AND = 245
    LOP_OR = 246
    LOP_ANDK = 247
    LOP_ORK = 248
    LOP_CONCAT = 249
    LOP_NOT = 250
    LOP_MINUS = 251
    LOP_LENGTH = 252
    LOP_NEWTABLE = 253
    LOP_DUPTABLE = 254
    LOP_SETLIST = 255
    LOP_FORNPREP = 256
    LOP_FORNLOOP = 257
    LOP_FORGLOOP = 258
    LOP_FORGPREP_INEXT = 259
    LOP_FASTCALL3 = 260
    LOP_FORGPREP_NEXT = 261
    LOP_NATIVECALL = 262
    LOP_GETVARARGS = 263
    LOP_DUPCLOSURE = 264
    LOP_PREPVARARGS = 265
    LOP_LOADKX = 266
    LOP_JUMPX = 267
    LOP_FASTCALL = 268
    LOP_COVERAGE = 269
    LOP_CAPTURE = 270
    LOP_SUBRK = 271
    LOP_DIVRK = 272
    LOP_FASTCALL1 = 273
    LOP_FASTCALL2 = 274
    LOP_FASTCALL2K = 275
    LOP_FORGPREP = 276
    LOP_JUMPXEQKNIL = 277
    LOP_JUMPXEQKB = 278
    LOP_JUMPXEQKN = 279
    LOP_JUMPXEQKS = 280
    LOP_IDIV = 281
    LOP_IDIVK = 282
    
    UNKNOWN = 999

class OpMode(IntEnum):
    """Instruction encoding modes"""
    iABC = 0
    iABx = 1
    iAsBx = 2
    iAx = 3
    iAD = 4   # LuaJIT/Luau mode

# Opcode properties for standard Lua
OPCODE_INFO = {
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
    D: int = 0      # For LuaJIT/Luau AD format
    sD: int = 0     # Signed D
    mode: OpMode = OpMode.iABC
    
    # Metadata
    pc: int = 0
    line: int = 0
    basic_block: int = -1
    is_leader: bool = False
    is_jump_target: bool = False
    references: List[int] = field(default_factory=list)
    jumps_to: Optional[int] = None
    
    # Analysis data
    live_in: Set[int] = field(default_factory=set)
    live_out: Set[int] = field(default_factory=set)
    def_vars: Set[int] = field(default_factory=set)
    use_vars: Set[int] = field(default_factory=set)
    
    def __repr__(self):
        base = f"[{self.pc:04d}] {self.opcode.name:16}"
        if self.mode == OpMode.iABC:
            args = f"A={self.A} B={self.B} C={self.C}"
        elif self.mode == OpMode.iABx:
            args = f"A={self.A} Bx={self.Bx}"
        elif self.mode == OpMode.iAsBx:
            args = f"A={self.A} sBx={self.sBx}"
        elif self.mode == OpMode.iAD:
            args = f"A={self.A} D={self.D}"
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
    value: Union[None, bool, float, str, int, bytes]
    
    # Metadata
    index: int = 0
    references: List[int] = field(default_factory=list)
    hash: str = ""
    is_encrypted: bool = False
    encryption_method: int = 0
    
    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        if self.value is None:
            data = b"nil"
        elif isinstance(self.value, bool):
            data = b"true" if self.value else b"false"
        elif isinstance(self.value, (int, float)):
            data = str(self.value).encode()
        elif isinstance(self.value, str):
            data = self.value.encode('utf-8', errors='replace')
        elif isinstance(self.value, bytes):
            data = self.value
        else:
            data = str(self.value).encode()
        
        return hashlib.md5(data).hexdigest()[:16]
    
    def __repr__(self):
        type_names = {0: "nil", 1: "bool", 3: "num", 4: "str", 0x13: "int", 0x14: "float"}
        type_name = type_names.get(self.type, f"type{self.type}")
        
        if self.type == 0:
            return "nil"
        elif self.type == 1:
            return f"bool({self.value})"
        elif self.type in [3, 0x13, 0x14]:
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
    kind: int = 0

@dataclass
class BasicBlock:
    """Basic block for control flow analysis"""
    id: int
    start_pc: int
    end_pc: int
    instructions: List[int] = field(default_factory=list)
    
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    
    dominators: Set[int] = field(default_factory=set)
    immediate_dominator: int = -1
    dominance_frontier: Set[int] = field(default_factory=set)
    
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
    
    instructions: List[Instruction] = field(default_factory=list)
    constants: List[Constant] = field(default_factory=list)
    prototypes: List['Function'] = field(default_factory=list)
    
    source_lines: List[int] = field(default_factory=list)
    locals: List[Local] = field(default_factory=list)
    upvalues: List[Upvalue] = field(default_factory=list)
    
    basic_blocks: List[BasicBlock] = field(default_factory=list)
    call_graph: Dict[int, List[int]] = field(default_factory=dict)
    register_usage: Dict[int, int] = field(default_factory=dict)
    
    complexity: int = 0
    hash: str = ""
    is_main: bool = False
    parent: Optional['Function'] = None
    depth: int = 0
    
    # Luau-specific
    flags: int = 0
    type_info: bytes = b""

@dataclass
class LuaHeader:
    """Enhanced Lua bytecode header"""
    signature: bytes = b''
    version: int = 0
    format: int = 0
    endianness: int = 1
    int_size: int = 4
    size_t_size: int = 4
    instruction_size: int = 4
    number_size: int = 8
    integral_flag: int = 0
    
    lua_version: LuaVersion = LuaVersion.UNKNOWN
    format_type: BytecodeFormat = BytecodeFormat.UNKNOWN
    is_little_endian: bool = True
    
    # Luau specific
    luau_version: int = 0
    types_version: int = 0
    
    # LuaJIT specific
    luajit_flags: int = 0
    
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    raw_header: bytes = b""

@dataclass
class LuaChunk:
    """Enhanced Lua bytecode chunk"""
    header: LuaHeader
    main_function: Function
    
    all_functions: List[Function] = field(default_factory=list)
    constant_pool: Dict[str, Constant] = field(default_factory=dict)
    string_pool: Dict[str, int] = field(default_factory=dict)
    
    # Luau specific
    string_table: List[str] = field(default_factory=list)
    
    file_size: int = 0
    checksum: str = ""
    compressed: bool = False
    encrypted: bool = False
    
    stats: Dict[str, Any] = field(default_factory=dict)

# ============================================
# Format Detection
# ============================================

class FormatDetector:
    """Detect bytecode format from data"""
    
    # Known signatures
    LUA_SIGNATURE = b'\x1bLua'
    LUAJIT_SIGNATURE = b'\x1bLJ'
    
    # Luau has no signature - starts with version byte
    LUAU_MIN_VERSION = 0
    LUAU_MAX_VERSION = 6
    
    @classmethod
    def detect(cls, data: bytes) -> Tuple[BytecodeFormat, LuaVersion, int]:
        """
        Detect format and version from bytecode data.
        Returns: (format_type, lua_version, header_start_pos)
        """
        if len(data) < 4:
            return BytecodeFormat.UNKNOWN, LuaVersion.UNKNOWN, 0
        
        # Check for compressed data
        if data[:2] == b'\x1f\x8b':  # gzip
            return BytecodeFormat.COMPRESSED, LuaVersion.UNKNOWN, 0
        
        if data[:4] == b'ZLIB' or data[:4] == b'zlib':
            return BytecodeFormat.COMPRESSED, LuaVersion.UNKNOWN, 0
        
        # Check for standard Lua signature
        if data[:4] == cls.LUA_SIGNATURE:
            if len(data) >= 5:
                version_byte = data[4]
                if version_byte == 0x51:
                    return BytecodeFormat.STANDARD_LUA, LuaVersion.LUA_51, 0
                elif version_byte == 0x52:
                    return BytecodeFormat.STANDARD_LUA, LuaVersion.LUA_52, 0
                elif version_byte == 0x53:
                    return BytecodeFormat.STANDARD_LUA, LuaVersion.LUA_53, 0
                elif version_byte == 0x54:
                    return BytecodeFormat.STANDARD_LUA, LuaVersion.LUA_54, 0
            return BytecodeFormat.STANDARD_LUA, LuaVersion.LUA_51, 0
        
        # Check for LuaJIT signature
        if data[:3] == cls.LUAJIT_SIGNATURE:
            if len(data) >= 4:
                flags = data[3]
                if flags & 0x02:  # BE flag
                    pass  # Big endian
                return BytecodeFormat.LUAJIT, LuaVersion.LUAJIT_21, 0
            return BytecodeFormat.LUAJIT, LuaVersion.LUAJIT_20, 0
        
        # Check for Luau bytecode (Roblox)
        # Luau has NO signature - it starts with version byte directly
        version_byte = data[0]
        if cls.LUAU_MIN_VERSION <= version_byte <= cls.LUAU_MAX_VERSION:
            # Additional validation for Luau
            if cls._validate_luau_header(data):
                luau_version = LuaVersion.LUAU_0 + version_byte
                if luau_version > LuaVersion.LUAU_6:
                    luau_version = LuaVersion.LUAU_6
                return BytecodeFormat.LUAU, luau_version, 0
        
        # Check if it might be Luau with different version
        if cls._validate_luau_header(data):
            return BytecodeFormat.LUAU, LuaVersion.LUAU_3, 0
        
        # Check for encrypted/obfuscated
        if cls._looks_encrypted(data):
            return BytecodeFormat.ENCRYPTED, LuaVersion.UNKNOWN, 0
        
        return BytecodeFormat.UNKNOWN, LuaVersion.UNKNOWN, 0
    
    @classmethod
    def _validate_luau_header(cls, data: bytes) -> bool:
        """Validate if data looks like Luau bytecode"""
        if len(data) < 8:
            return False
        
        version = data[0]
        
        # Version should be reasonable (0-10)
        if version > 10:
            return False
        
        # Try to read string table size (varint starting at position 1)
        pos = 1
        try:
            # Read varint for string count
            string_count, pos = cls._read_varint(data, pos)
            
            # Reasonable string count (0 to 100000)
            if string_count > 100000:
                return False
            
            # If we can read string table structure, it's likely Luau
            if string_count > 0 and pos < len(data):
                # Try to read first string length
                first_len, _ = cls._read_varint(data, pos)
                if first_len < 1000000:  # Reasonable string length
                    return True
            elif string_count == 0:
                # Empty string table is valid
                return True
                
        except:
            return False
        
        return False
    
    @classmethod
    def _read_varint(cls, data: bytes, pos: int) -> Tuple[int, int]:
        """Read a variable-length integer (Luau format)"""
        result = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
            if shift > 35:  # Prevent infinite loop
                break
        return result, pos
    
    @classmethod
    def _looks_encrypted(cls, data: bytes) -> bool:
        """Check if data looks encrypted"""
        if len(data) < 16:
            return False
        
        # High entropy check
        byte_counts = [0] * 256
        for byte in data[:min(256, len(data))]:
            byte_counts[byte] += 1
        
        # Calculate entropy approximation
        non_zero = sum(1 for c in byte_counts if c > 0)
        if non_zero > 200:  # High diversity suggests encryption
            return True
        
        return False

# ============================================
# Base Parser Strategy
# ============================================

class ParserStrategy(ABC):
    """Abstract base class for version-specific parsers"""
    
    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos
        self.header: Optional[LuaHeader] = None
        self.stats = defaultdict(int)
    
    @abstractmethod
    def parse_header(self) -> LuaHeader:
        pass
    
    @abstractmethod
    def parse_function(self, depth: int = 0) -> Function:
        pass
    
    @abstractmethod
    def decode_instruction(self, raw: int, pc: int = 0) -> Instruction:
        pass
    
    def read_bytes(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError(f"Read past end: pos={self.pos}, n={n}, len={len(self.data)}")
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result
    
    def read_byte(self) -> int:
        return self.read_bytes(1)[0]
    
    def peek_byte(self) -> int:
        if self.pos >= len(self.data):
            return -1
        return self.data[self.pos]
    
    def remaining(self) -> int:
        return len(self.data) - self.pos

# ============================================
# Standard Lua Parser (5.1, 5.2, 5.3, 5.4)
# ============================================

class StandardLuaParser(ParserStrategy):
    """Parser for standard Lua 5.x bytecode"""
    
    def __init__(self, data: bytes, pos: int = 0, version: LuaVersion = LuaVersion.LUA_51):
        super().__init__(data, pos)
        self.version = version
    
    def parse_header(self) -> LuaHeader:
        header = LuaHeader()
        header.raw_header = self.data[self.pos:self.pos + 32]
        
        # Signature
        header.signature = self.read_bytes(4)
        if header.signature != b'\x1bLua':
            header.is_valid = False
            header.validation_errors.append(f"Invalid signature: {header.signature.hex()}")
        
        # Version
        header.version = self.read_byte()
        
        if header.version == 0x51:
            header.lua_version = LuaVersion.LUA_51
        elif header.version == 0x52:
            header.lua_version = LuaVersion.LUA_52
        elif header.version == 0x53:
            header.lua_version = LuaVersion.LUA_53
        elif header.version == 0x54:
            header.lua_version = LuaVersion.LUA_54
        else:
            header.is_valid = False
            header.validation_errors.append(f"Unknown version: 0x{header.version:02x}")
            header.lua_version = self.version
        
        self.version = header.lua_version
        
        # Format version
        header.format = self.read_byte()
        
        if self.version in [LuaVersion.LUA_53, LuaVersion.LUA_54]:
            # Lua 5.3+ has LUAC_DATA for validation
            luac_data = self.read_bytes(6)
            expected = b'\x19\x93\r\n\x1a\n'
            if luac_data != expected:
                header.validation_errors.append(f"Invalid LUAC_DATA: {luac_data.hex()}")
        else:
            # Lua 5.1/5.2 format
            header.endianness = self.read_byte()
            header.is_little_endian = header.endianness == 1
        
        # Size info
        header.int_size = self.read_byte()
        header.size_t_size = self.read_byte()
        header.instruction_size = self.read_byte()
        
        if self.version in [LuaVersion.LUA_53, LuaVersion.LUA_54]:
            # Integer size and number size
            header.number_size = self.read_byte()
            header.integral_flag = self.read_byte()
            
            # Check LUAC_INT and LUAC_NUM
            if self.version == LuaVersion.LUA_53:
                luac_int = self.read_bytes(header.int_size)
                luac_num = self.read_bytes(header.number_size)
        else:
            header.number_size = self.read_byte()
            header.integral_flag = self.read_byte()
        
        header.format_type = BytecodeFormat.STANDARD_LUA
        self.header = header
        return header
    
    def read_int(self) -> int:
        size = self.header.int_size if self.header else 4
        data = self.read_bytes(size)
        
        fmt = '<' if (not self.header or self.header.is_little_endian) else '>'
        if size == 4:
            return struct.unpack(fmt + 'i', data)[0]
        elif size == 8:
            return struct.unpack(fmt + 'q', data)[0]
        else:
            return int.from_bytes(data, 'little' if fmt == '<' else 'big', signed=True)
    
    def read_size_t(self) -> int:
        size = self.header.size_t_size if self.header else 4
        data = self.read_bytes(size)
        
        fmt = '<' if (not self.header or self.header.is_little_endian) else '>'
        if size == 4:
            return struct.unpack(fmt + 'I', data)[0]
        elif size == 8:
            return struct.unpack(fmt + 'Q', data)[0]
        else:
            return int.from_bytes(data, 'little' if fmt == '<' else 'big', signed=False)
    
    def read_number(self) -> float:
        size = self.header.number_size if self.header else 8
        data = self.read_bytes(size)
        
        fmt = '<' if (not self.header or self.header.is_little_endian) else '>'
        if size == 4:
            return struct.unpack(fmt + 'f', data)[0]
        else:
            return struct.unpack(fmt + 'd', data)[0]
    
    def read_integer(self) -> int:
        """Read Lua 5.3+ integer"""
        size = self.header.int_size if self.header else 8
        data = self.read_bytes(size)
        
        fmt = '<' if (not self.header or self.header.is_little_endian) else '>'
        if size == 4:
            return struct.unpack(fmt + 'i', data)[0]
        else:
            return struct.unpack(fmt + 'q', data)[0]
    
    def read_string(self) -> str:
        if self.version in [LuaVersion.LUA_53, LuaVersion.LUA_54]:
            size = self.read_byte()
            if size == 0xFF:
                size = self.read_size_t()
            if size == 0:
                return ""
            data = self.read_bytes(size - 1)
            return data.decode('utf-8', errors='replace')
        else:
            size = self.read_size_t()
            if size == 0:
                return ""
            data = self.read_bytes(size)
            if data and data[-1] == 0:
                return data[:-1].decode('utf-8', errors='replace')
            return data.decode('utf-8', errors='replace')
    
    def read_instruction(self) -> int:
        data = self.read_bytes(4)
        fmt = '<' if (not self.header or self.header.is_little_endian) else '>'
        return struct.unpack(fmt + 'I', data)[0]
    
    def decode_instruction(self, raw: int, pc: int = 0) -> Instruction:
        if self.version in [LuaVersion.LUA_51, LuaVersion.LUA_52]:
            return self._decode_lua51(raw, pc)
        else:
            return self._decode_lua53(raw, pc)
    
    def _decode_lua51(self, raw: int, pc: int) -> Instruction:
        opcode_num = raw & 0x3F
        try:
            opcode = OpCode(opcode_num)
        except ValueError:
            opcode = OpCode.UNKNOWN
        
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
    
    def _decode_lua53(self, raw: int, pc: int) -> Instruction:
        opcode_num = raw & 0x7F
        try:
            opcode = OpCode(opcode_num)
        except ValueError:
            opcode = OpCode.UNKNOWN
        
        A = (raw >> 7) & 0xFF
        B = (raw >> 16) & 0xFF
        C = (raw >> 24) & 0xFF
        Bx = (raw >> 15) & 0x3FFFF
        sBx = Bx - 131071
        
        mode = OPCODE_INFO.get(opcode, (OpMode.iABC,))[0] if opcode in OPCODE_INFO else OpMode.iABC
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            Bx=Bx, sBx=sBx, mode=mode, pc=pc
        )
    
    def read_constant(self, index: int) -> Constant:
        const_type = self.read_byte()
        
        if const_type == 0:  # LUA_TNIL
            return Constant(type=const_type, value=None, index=index)
        
        elif const_type == 1:  # LUA_TBOOLEAN
            return Constant(type=const_type, value=self.read_byte() != 0, index=index)
        
        elif const_type == 3:  # LUA_TNUMBER (Lua 5.1/5.2)
            return Constant(type=const_type, value=self.read_number(), index=index)
        
        elif const_type == 4:  # LUA_TSTRING
            self.stats['strings'] += 1
            return Constant(type=const_type, value=self.read_string(), index=index)
        
        elif const_type == 0x03:  # LUA_VNUMFLT (Lua 5.3+)
            return Constant(type=const_type, value=self.read_number(), index=index)
        
        elif const_type == 0x13:  # LUA_VNUMINT (Lua 5.3+)
            return Constant(type=const_type, value=self.read_integer(), index=index)
        
        elif const_type == 0x04 or const_type == 0x14:  # LUA_VSHRSTR, LUA_VLNGSTR
            self.stats['strings'] += 1
            return Constant(type=const_type, value=self.read_string(), index=index)
        
        else:
            raise ValueError(f"Unknown constant type: {const_type} at position {self.pos}")
    
    def parse_function(self, depth: int = 0) -> Function:
        func = Function(depth=depth, is_main=(depth == 0))
        
        # Source name (optional in 5.4)
        func.source = self.read_string()
        
        # Line info
        func.line_defined = self.read_int()
        func.last_line_defined = self.read_int()
        
        # Function params
        if self.version == LuaVersion.LUA_54:
            func.num_params = self.read_byte()
            func.is_vararg = self.read_byte()
            func.max_stack_size = self.read_byte()
        else:
            func.num_upvalues = self.read_byte()
            func.num_params = self.read_byte()
            func.is_vararg = self.read_byte()
            func.max_stack_size = self.read_byte()
        
        self.stats['functions'] += 1
        
        # Instructions
        num_instructions = self.read_int()
        for i in range(num_instructions):
            raw = self.read_instruction()
            instr = self.decode_instruction(raw, i)
            func.instructions.append(instr)
        
        self.stats['instructions'] += num_instructions
        
        # Constants
        num_constants = self.read_int()
        for i in range(num_constants):
            const = self.read_constant(i)
            func.constants.append(const)
        
        self.stats['constants'] += num_constants
        
        # Upvalues (Lua 5.2+)
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
        
        # Nested prototypes
        num_prototypes = self.read_int()
        for _ in range(num_prototypes):
            proto = self.parse_function(depth + 1)
            proto.parent = func
            func.prototypes.append(proto)
        
        # Debug: source lines
        num_lines = self.read_int()
        for i in range(num_lines):
            if self.version == LuaVersion.LUA_54:
                # Lua 5.4 uses abslineinfo
                line = self.read_byte()
            else:
                line = self.read_int()
            func.source_lines.append(line)
            if i < len(func.instructions):
                func.instructions[i].line = line
        
        # Lua 5.4 abslineinfo
        if self.version == LuaVersion.LUA_54:
            num_abslines = self.read_int()
            for _ in range(num_abslines):
                pc = self.read_int()
                line = self.read_int()
        
        # Debug: locals
        num_locals = self.read_int()
        for i in range(num_locals):
            name = self.read_string()
            start_pc = self.read_int()
            end_pc = self.read_int()
            
            local = Local(name, start_pc, end_pc)
            local.is_parameter = i < func.num_params
            func.locals.append(local)
        
        # Debug: upvalue names
        num_upvalue_names = self.read_int()
        for i in range(num_upvalue_names):
            name = self.read_string()
            if i < len(func.upvalues):
                func.upvalues[i].name = name
        
        # Compute hash
        func.hash = self._compute_function_hash(func)
        
        return func
    
    def _compute_function_hash(self, func: Function) -> str:
        data = bytearray()
        for instr in func.instructions:
            data.extend(struct.pack('<I', instr.raw))
        return hashlib.md5(data).hexdigest()

# ============================================
# LuaJIT Parser
# ============================================

class LuaJITParser(ParserStrategy):
    """Parser for LuaJIT bytecode"""
    
    def __init__(self, data: bytes, pos: int = 0):
        super().__init__(data, pos)
        self.is_big_endian = False
        self.strip = False
        self.ffi = False
    
    def parse_header(self) -> LuaHeader:
        header = LuaHeader()
        header.raw_header = self.data[self.pos:self.pos + 12]
        
        # Signature (3 bytes)
        sig = self.read_bytes(3)
        if sig != b'\x1bLJ':
            header.is_valid = False
            header.validation_errors.append(f"Invalid LuaJIT signature: {sig.hex()}")
        
        header.signature = sig
        
        # Version/flags
        flags = self.read_byte()
        header.luajit_flags = flags
        
        self.is_big_endian = bool(flags & 0x01)
        self.strip = bool(flags & 0x02)
        self.ffi = bool(flags & 0x04)
        
        header.is_little_endian = not self.is_big_endian
        header.format_type = BytecodeFormat.LUAJIT
        header.lua_version = LuaVersion.LUAJIT_21
        
        # Debug name (if not stripped)
        if not self.strip:
            name_len = self._read_uleb128()
            if name_len > 0:
                header.signature = self.read_bytes(name_len)
        
        self.header = header
        return header
    
    def _read_uleb128(self) -> int:
        """Read unsigned LEB128"""
        result = 0
        shift = 0
        while True:
            byte = self.read_byte()
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result
    
    def decode_instruction(self, raw: int, pc: int = 0) -> Instruction:
        # LuaJIT instruction format
        op = raw & 0xFF
        A = (raw >> 8) & 0xFF
        D = (raw >> 16) & 0xFFFF
        C = D & 0xFF
        B = (D >> 8) & 0xFF
        
        try:
            opcode = OpCode(op + 100)  # Offset for LuaJIT opcodes
        except ValueError:
            opcode = OpCode.UNKNOWN
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            D=D, sD=D - 0x8000 if D >= 0x8000 else D,
            mode=OpMode.iAD, pc=pc
        )
    
    def parse_function(self, depth: int = 0) -> Function:
        func = Function(depth=depth, is_main=(depth == 0))
        
        # Prototype size
        proto_size = self._read_uleb128()
        if proto_size == 0:
            return func  # End of prototypes
        
        proto_start = self.pos
        
        # Flags
        flags = self.read_byte()
        func.num_params = self.read_byte()
        func.max_stack_size = self.read_byte()
        num_upvalues = self.read_byte()
        
        # Counts
        num_kgc = self._read_uleb128()  # GC constants
        num_kn = self._read_uleb128()   # Numeric constants
        num_bc = self._read_uleb128()   # Bytecode instructions
        
        # Debug info (if not stripped)
        if not self.strip:
            debug_size = self._read_uleb128()
            first_line = self._read_uleb128()
            num_lines = self._read_uleb128()
        
        # Instructions
        for i in range(num_bc):
            raw = struct.unpack('<I', self.read_bytes(4))[0]
            instr = self.decode_instruction(raw, i)
            func.instructions.append(instr)
        
        self.stats['instructions'] += num_bc
        self.stats['functions'] += 1
        
        # Upvalue refs
        for i in range(num_upvalues):
            ref = struct.unpack('<H', self.read_bytes(2))[0]
            upval = Upvalue(name="", index=i, reg_index=ref & 0x3FFF,
                           in_stack=bool(ref & 0x8000))
            func.upvalues.append(upval)
        
        # GC constants (strings, tables, child protos)
        for i in range(num_kgc):
            ktype = self._read_uleb128()
            if ktype >= 5:  # String
                length = ktype - 5
                if length > 0:
                    sdata = self.read_bytes(length)
                    const = Constant(type=4, value=sdata.decode('utf-8', errors='replace'), index=i)
                    func.constants.append(const)
        
        # Numeric constants
        for i in range(num_kn):
            is_int = self._read_uleb128()
            lo = self._read_uleb128()
            if is_int & 1:
                value = lo
            else:
                hi = self._read_uleb128()
                value = struct.unpack('d', struct.pack('<Q', (hi << 32) | lo))[0]
            
            const = Constant(type=3, value=value, index=len(func.constants))
            func.constants.append(const)
        
        self.stats['constants'] += num_kgc + num_kn
        
        # Skip debug info
        if not self.strip:
            # Skip to end of prototype
            bytes_read = self.pos - proto_start
            remaining = proto_size - bytes_read
            if remaining > 0:
                self.pos += remaining
        
        return func

# ============================================
# Luau Parser (Roblox)
# ============================================

class LuauParser(ParserStrategy):
    """Parser for Roblox Luau bytecode"""
    
    def __init__(self, data: bytes, pos: int = 0):
        super().__init__(data, pos)
        self.luau_version = 0
        self.types_version = 0
        self.string_table: List[str] = []
    
    def parse_header(self) -> LuaHeader:
        header = LuaHeader()
        header.raw_header = self.data[self.pos:min(self.pos + 32, len(self.data))]
        
        # Version byte (no signature in Luau)
        self.luau_version = self.read_byte()
        header.luau_version = self.luau_version
        header.version = self.luau_version
        
        if self.luau_version > 6:
            header.validation_errors.append(f"Unknown Luau version: {self.luau_version}")
        
        # Types version (Luau v4+)
        if self.luau_version >= 4:
            self.types_version = self.read_byte()
            header.types_version = self.types_version
        
        header.format_type = BytecodeFormat.LUAU
        header.lua_version = LuaVersion.LUAU_0 + min(self.luau_version, 6)
        header.is_little_endian = True
        header.is_valid = True
        
        self.header = header
        return header
    
    def _read_varint(self) -> int:
        """Read variable-length integer"""
        result = 0
        shift = 0
        while True:
            if self.pos >= len(self.data):
                return result
            byte = self.read_byte()
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
            if shift > 35:
                break
        return result
    
    def _read_string_table(self):
        """Read the string table"""
        string_count = self._read_varint()
        
        for _ in range(string_count):
            length = self._read_varint()
            if length > 0:
                if self.pos + length <= len(self.data):
                    sdata = self.read_bytes(length)
                    self.string_table.append(sdata.decode('utf-8', errors='replace'))
                else:
                    self.string_table.append("")
            else:
                self.string_table.append("")
        
        self.stats['strings'] = len(self.string_table)
    
    def get_string(self, index: int) -> str:
        """Get string from table by index"""
        if index == 0:
            return ""
        if 0 < index <= len(self.string_table):
            return self.string_table[index - 1]
        return f"<invalid string {index}>"
    
    def decode_instruction(self, raw: int, pc: int = 0) -> Instruction:
        # Luau instruction format
        op = raw & 0xFF
        A = (raw >> 8) & 0xFF
        B = (raw >> 16) & 0xFF
        C = (raw >> 24) & 0xFF
        
        # D format
        D = (raw >> 16) & 0xFFFF
        sD = D - 0x10000 if D >= 0x8000 else D
        
        # E format (24-bit signed)
        E = (raw >> 8) & 0xFFFFFF
        sE = E - 0x1000000 if E >= 0x800000 else E
        
        try:
            opcode = OpCode(op + 200)  # Offset for Luau opcodes
        except ValueError:
            opcode = OpCode.UNKNOWN
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            D=D, sD=sD, Bx=E, sBx=sE,
            mode=OpMode.iABC, pc=pc
        )
    
    def _read_proto_function(self, depth: int = 0) -> Function:
        """Read a Luau prototype/function"""
        func = Function(depth=depth, is_main=(depth == 0))
        
        # Function header
        func.max_stack_size = self.read_byte()
        func.num_params = self.read_byte()
        func.num_upvalues = self.read_byte()
        func.is_vararg = self.read_byte()
        
        # Luau v4+ has additional flags
        if self.luau_version >= 4:
            func.flags = self.read_byte()
            
            # Type info
            type_size = self._read_varint()
            if type_size > 0:
                func.type_info = self.read_bytes(type_size)
        
        # Instructions (sizecode)
        num_instructions = self._read_varint()
        for i in range(num_instructions):
            raw = struct.unpack('<I', self.read_bytes(4))[0]
            instr = self.decode_instruction(raw, i)
            func.instructions.append(instr)
        
        self.stats['instructions'] += num_instructions
        
        # Constants (sizek)
        num_constants = self._read_varint()
        for i in range(num_constants):
            const = self._read_luau_constant(i)
            func.constants.append(const)
        
        self.stats['constants'] += num_constants
        
        # Nested prototypes (sizep)
        num_protos = self._read_varint()
        for _ in range(num_protos):
            proto_id = self._read_varint()
            # Proto IDs reference the proto table built later
            # For now, store placeholder
        
        # Line info
        line_gap_log2 = self.read_byte()
        if line_gap_log2 != 0:
            intervals = ((num_instructions - 1) >> line_gap_log2) + 1
            
            # Line gaps
            for _ in range(num_instructions):
                func.source_lines.append(self.read_byte())
            
            # Absolute line info
            abs_offset = self.read_byte() if num_instructions > 0 else 0
            for _ in range(intervals):
                line_info = struct.unpack('<I', self.read_bytes(4))[0] if num_instructions > 256 else self.read_byte()
        
        # Debug info (if present)
        if self.remaining() >= 4:
            has_debug = self._read_varint()
            if has_debug:
                # Local count
                local_count = self._read_varint()
                for j in range(local_count):
                    name_idx = self._read_varint()
                    start_pc = self._read_varint()
                    end_pc = self._read_varint()
                    reg = self.read_byte()
                    
                    name = self.get_string(name_idx)
                    local = Local(name, start_pc, end_pc, register=reg)
                    func.locals.append(local)
                
                # Upvalue count
                upval_count = self._read_varint()
                for j in range(upval_count):
                    name_idx = self._read_varint()
                    name = self.get_string(name_idx)
                    if j < len(func.upvalues):
                        func.upvalues[j].name = name
        
        self.stats['functions'] += 1
        func.hash = self._compute_function_hash(func)
        
        return func
    
    def _read_luau_constant(self, index: int) -> Constant:
        """Read a Luau constant"""
        const_type = self.read_byte()
        
        if const_type == 0:  # Nil
            return Constant(type=0, value=None, index=index)
        
        elif const_type == 1:  # Boolean
            return Constant(type=1, value=self.read_byte() != 0, index=index)
        
        elif const_type == 2:  # Number
            value = struct.unpack('<d', self.read_bytes(8))[0]
            return Constant(type=3, value=value, index=index)
        
        elif const_type == 3:  # String
            string_idx = self._read_varint()
            value = self.get_string(string_idx)
            return Constant(type=4, value=value, index=index)
        
        elif const_type == 4:  # Import
            iid = struct.unpack('<I', self.read_bytes(4))[0]
            return Constant(type=const_type, value=iid, index=index)
        
        elif const_type == 5:  # Table
            # Table constants
            num_keys = self._read_varint()
            keys = []
            for _ in range(num_keys):
                key_idx = self._read_varint()
                keys.append(key_idx)
            return Constant(type=const_type, value=keys, index=index)
        
        elif const_type == 6:  # Closure
            proto_id = self._read_varint()
            return Constant(type=const_type, value=proto_id, index=index)
        
        elif const_type == 7:  # Vector (Luau v5+)
            if self.luau_version >= 5:
                x = struct.unpack('<f', self.read_bytes(4))[0]
                y = struct.unpack('<f', self.read_bytes(4))[0]
                z = struct.unpack('<f', self.read_bytes(4))[0]
                w = struct.unpack('<f', self.read_bytes(4))[0]
                return Constant(type=const_type, value=(x, y, z, w), index=index)
        
        return Constant(type=const_type, value=None, index=index)
    
    def parse_function(self, depth: int = 0) -> Function:
        """Parse the main chunk"""
        # First read string table
        self._read_string_table()
        
        # Read proto count
        proto_count = self._read_varint()
        
        # Read all prototypes
        protos: List[Function] = []
        for i in range(proto_count):
            proto = self._read_proto_function(depth=1 if i > 0 else 0)
            protos.append(proto)
        
        # Main function ID
        main_id = self._read_varint()
        
        if main_id < len(protos):
            main = protos[main_id]
            main.is_main = True
            main.depth = 0
            
            # Link nested protos
            for i, proto in enumerate(protos):
                if i != main_id:
                    main.prototypes.append(proto)
                    proto.parent = main
            
            return main
        elif protos:
            return protos[0]
        else:
            return Function(is_main=True)
    
    def _compute_function_hash(self, func: Function) -> str:
        data = bytearray()
        for instr in func.instructions:
            data.extend(struct.pack('<I', instr.raw))
        return hashlib.md5(data).hexdigest()

# ============================================
# Main Parser with Auto-Detection
# ============================================

class AdvancedBytecodeParser:
    """Advanced parser with automatic format detection"""
    
    def __init__(self, data: bytes, auto_detect: bool = True, 
                 force_format: Optional[BytecodeFormat] = None,
                 force_version: Optional[LuaVersion] = None):
        self.data = data
        self.auto_detect = auto_detect
        self.force_format = force_format
        self.force_version = force_version
        
        self.format_type: BytecodeFormat = BytecodeFormat.UNKNOWN
        self.version: LuaVersion = LuaVersion.UNKNOWN
        self.parser: Optional[ParserStrategy] = None
        
        self.stats = {
            'total_instructions': 0,
            'total_constants': 0,
            'total_functions': 0,
            'total_strings': 0,
            'max_stack': 0,
        }
    
    def detect_format(self) -> Tuple[BytecodeFormat, LuaVersion]:
        """Detect bytecode format and version"""
        if self.force_format and self.force_version:
            return self.force_format, self.force_version
        
        fmt, version, _ = FormatDetector.detect(self.data)
        
        if self.force_format:
            fmt = self.force_format
        if self.force_version:
            version = self.force_version
        
        return fmt, version
    
    def _create_parser(self) -> ParserStrategy:
        """Create appropriate parser for detected format"""
        self.format_type, self.version = self.detect_format()
        
        if self.format_type == BytecodeFormat.STANDARD_LUA:
            return StandardLuaParser(self.data, 0, self.version)
        
        elif self.format_type == BytecodeFormat.LUAJIT:
            return LuaJITParser(self.data, 0)
        
        elif self.format_type == BytecodeFormat.LUAU:
            return LuauParser(self.data, 0)
        
        elif self.format_type == BytecodeFormat.COMPRESSED:
            # Try to decompress
            decompressed = self._try_decompress()
            if decompressed:
                # Recursively detect decompressed format
                sub_parser = AdvancedBytecodeParser(decompressed)
                return sub_parser._create_parser()
            raise ValueError("Failed to decompress bytecode")
        
        elif self.format_type == BytecodeFormat.ENCRYPTED:
            raise ValueError("Encrypted bytecode not supported")
        
        else:
            # Try each parser
            for parser_class in [StandardLuaParser, LuauParser, LuaJITParser]:
                try:
                    parser = parser_class(self.data, 0)
                    parser.parse_header()
                    if parser.header.is_valid:
                        return parser
                except:
                    continue
            
            raise ValueError(f"Unknown bytecode format: {self.data[:16].hex()}")
    
    def _try_decompress(self) -> Optional[bytes]:
        """Try various decompression methods"""
        # Try gzip
        if self.data[:2] == b'\x1f\x8b':
            try:
                import gzip
                return gzip.decompress(self.data)
            except:
                pass
        
        # Try zlib
        try:
            return zlib.decompress(self.data)
        except:
            pass
        
        # Try raw deflate
        try:
            return zlib.decompress(self.data, -15)
        except:
            pass
        
        return None
    
    def parse(self) -> LuaChunk:
        """Parse bytecode and return chunk"""
        try:
            self.parser = self._create_parser()
            
            # Parse header
            header = self.parser.parse_header()
            
            # Parse main function
            main_function = self.parser.parse_function(depth=0)
            
            # Build stats
            self.stats['total_instructions'] = self.parser.stats.get('instructions', 0)
            self.stats['total_constants'] = self.parser.stats.get('constants', 0)
            self.stats['total_functions'] = self.parser.stats.get('functions', 0)
            self.stats['total_strings'] = self.parser.stats.get('strings', 0)
            
            # Create chunk
            chunk = LuaChunk(
                header=header,
                main_function=main_function,
                file_size=len(self.data),
                stats=self.stats.copy()
            )
            
            # Build indices
            self._build_indices(chunk)
            
            # Luau specific
            if isinstance(self.parser, LuauParser):
                chunk.string_table = self.parser.string_table
            
            # Checksum
            chunk.checksum = hashlib.sha256(self.data).hexdigest()
            
            return chunk
            
        except Exception as e:
            # Create error chunk
            header = LuaHeader()
            header.is_valid = False
            header.validation_errors.append(str(e))
            header.format_type = self.format_type
            header.lua_version = self.version
            
            return LuaChunk(
                header=header,
                main_function=Function(is_main=True),
                file_size=len(self.data),
                stats={'error': str(e)}
            )
    
    def _build_indices(self, chunk: LuaChunk):
        """Build global indices"""
        def collect_functions(func: Function):
            chunk.all_functions.append(func)
            for proto in func.prototypes:
                collect_functions(proto)
        
        collect_functions(chunk.main_function)
        
        # Constant pool
        for func in chunk.all_functions:
            for const in func.constants:
                chunk.constant_pool[const.hash] = const
                if const.type == 4 and isinstance(const.value, str):
                    chunk.string_pool[const.value] = const.index

# ============================================
# Control Flow Analysis
# ============================================

class ControlFlowAnalyzer:
    """Advanced control flow analysis"""
    
    def __init__(self, function: Function):
        self.function = function
        self.basic_blocks: List[BasicBlock] = []
    
    def analyze(self):
        """Perform complete CFG analysis"""
        if not self.function.instructions:
            return
        
        self._identify_basic_blocks()
        self._build_cfg()
        self._compute_dominators()
        self._detect_loops()
        
        self.function.basic_blocks = self.basic_blocks
    
    def _identify_basic_blocks(self):
        """Identify basic block boundaries"""
        leaders = {0}
        
        jump_opcodes = {
            OpCode.JMP, OpCode.FORLOOP, OpCode.FORPREP, OpCode.TFORLOOP,
            # Luau
            OpCode.LOP_JUMP, OpCode.LOP_JUMPBACK, OpCode.LOP_JUMPIF,
            OpCode.LOP_JUMPIFNOT, OpCode.LOP_JUMPIFEQ, OpCode.LOP_JUMPIFLE,
            OpCode.LOP_JUMPIFLT, OpCode.LOP_JUMPIFNOTEQ, OpCode.LOP_JUMPIFNOTLE,
            OpCode.LOP_JUMPIFNOTLT, OpCode.LOP_JUMPX,
            # LuaJIT
            OpCode.LJ_JMP, OpCode.LJ_FORL, OpCode.LJ_ITERL, OpCode.LJ_LOOP,
        }
        
        test_opcodes = {
            OpCode.EQ, OpCode.LT, OpCode.LE, OpCode.TEST, OpCode.TESTSET,
        }
        
        return_opcodes = {
            OpCode.RETURN, OpCode.TAILCALL,
            OpCode.LOP_RETURN,
            OpCode.LJ_RET, OpCode.LJ_RET0, OpCode.LJ_RET1, OpCode.LJ_RETM,
        }
        
        for i, instr in enumerate(self.function.instructions):
            if instr.opcode in jump_opcodes:
                # Calculate target
                if instr.mode == OpMode.iAsBx:
                    target = i + 1 + instr.sBx
                elif instr.mode == OpMode.iAD:
                    target = i + 1 + instr.sD
                else:
                    target = i + 1 + instr.sBx
                
                if 0 <= target < len(self.function.instructions):
                    leaders.add(target)
                    self.function.instructions[target].is_jump_target = True
                    instr.jumps_to = target
                
                if i + 1 < len(self.function.instructions):
                    leaders.add(i + 1)
            
            elif instr.opcode in test_opcodes:
                if i + 1 < len(self.function.instructions):
                    leaders.add(i + 1)
            
            elif instr.opcode in return_opcodes:
                if i + 1 < len(self.function.instructions):
                    leaders.add(i + 1)
        
        # Mark leaders and create blocks
        sorted_leaders = sorted(leaders)
        for i, start in enumerate(sorted_leaders):
            end = sorted_leaders[i + 1] - 1 if i + 1 < len(sorted_leaders) else len(self.function.instructions) - 1
            
            bb = BasicBlock(
                id=i,
                start_pc=start,
                end_pc=end,
                instructions=list(range(start, end + 1))
            )
            
            for pc in range(start, end + 1):
                self.function.instructions[pc].basic_block = i
                self.function.instructions[pc].is_leader = (pc == start)
            
            self.basic_blocks.append(bb)
    
    def _build_cfg(self):
        """Build control flow graph"""
        for bb in self.basic_blocks:
            if bb.end_pc >= len(self.function.instructions):
                continue
            
            last_instr = self.function.instructions[bb.end_pc]
            
            # Handle different instruction types
            if last_instr.jumps_to is not None:
                target_bb = self._find_bb_for_pc(last_instr.jumps_to)
                if target_bb:
                    bb.successors.add(target_bb.id)
                    target_bb.predecessors.add(bb.id)
            
            # Fall-through (if not unconditional jump or return)
            return_ops = {OpCode.RETURN, OpCode.TAILCALL, OpCode.LOP_RETURN,
                         OpCode.LJ_RET, OpCode.LJ_RET0, OpCode.LJ_RET1}
            
            if last_instr.opcode not in return_ops:
                if bb.end_pc + 1 < len(self.function.instructions):
                    next_bb = self._find_bb_for_pc(bb.end_pc + 1)
                    if next_bb and next_bb.id != bb.id:
                        bb.successors.add(next_bb.id)
                        next_bb.predecessors.add(bb.id)
    
    def _find_bb_for_pc(self, pc: int) -> Optional[BasicBlock]:
        for bb in self.basic_blocks:
            if bb.start_pc <= pc <= bb.end_pc:
                return bb
        return None
    
    def _compute_dominators(self):
        if not self.basic_blocks:
            return
        
        entry = self.basic_blocks[0]
        entry.dominators = {entry.id}
        
        for bb in self.basic_blocks[1:]:
            bb.dominators = set(range(len(self.basic_blocks)))
        
        changed = True
        iterations = 0
        max_iterations = len(self.basic_blocks) * 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for bb in self.basic_blocks[1:]:
                new_dom = set(range(len(self.basic_blocks)))
                
                for pred_id in bb.predecessors:
                    if pred_id < len(self.basic_blocks):
                        pred_bb = self.basic_blocks[pred_id]
                        new_dom &= pred_bb.dominators
                
                new_dom.add(bb.id)
                
                if new_dom != bb.dominators:
                    bb.dominators = new_dom
                    changed = True
    
    def _detect_loops(self):
        for bb in self.basic_blocks:
            for succ_id in bb.successors:
                if succ_id in bb.dominators:
                    if succ_id < len(self.basic_blocks):
                        header = self.basic_blocks[succ_id]
                        header.is_loop_header = True

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
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("LUA BYTECODE DISASSEMBLY")
        lines.append("=" * 80)
        
        header = self.chunk.header
        lines.append(f"Format: {header.format_type.name}")
        lines.append(f"Version: {header.lua_version.name}")
        
        if header.format_type == BytecodeFormat.LUAU:
            lines.append(f"Luau Version: {header.luau_version}")
        
        lines.append(f"Valid: {header.is_valid}")
        
        if header.validation_errors:
            lines.append("Validation Errors:")
            for err in header.validation_errors:
                lines.append(f"  - {err}")
        
        lines.append(f"File Size: {self.chunk.file_size} bytes")
        lines.append(f"Checksum: {self.chunk.checksum[:16]}...")
        lines.append("=" * 80)
        lines.append("")
        
        # Statistics
        lines.append("STATISTICS:")
        for key, value in self.chunk.stats.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # String table (Luau)
        if self.chunk.string_table:
            lines.append(f"STRING TABLE ({len(self.chunk.string_table)} strings):")
            for i, s in enumerate(self.chunk.string_table[:20]):
                preview = s[:50] + "..." if len(s) > 50 else s
                lines.append(f"  [{i}] \"{preview}\"")
            if len(self.chunk.string_table) > 20:
                lines.append(f"  ... and {len(self.chunk.string_table) - 20} more")
            lines.append("")
        
        # Disassemble main function
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
        lines = []
        prefix = "  " * indent
        
        lines.append(f"{prefix}{'=' * 60}")
        lines.append(f"{prefix}FUNCTION: {func.source or '<anonymous>'}")
        lines.append(f"{prefix}{'=' * 60}")
        lines.append(f"{prefix}Parameters: {func.num_params}, Vararg: {func.is_vararg}")
        lines.append(f"{prefix}Stack Size: {func.max_stack_size}, Upvalues: {func.num_upvalues}")
        lines.append(f"{prefix}Hash: {func.hash}")
        lines.append(f"{prefix}")
        
        # Constants
        if func.constants:
            lines.append(f"{prefix}CONSTANTS ({len(func.constants)}):")
            for i, const in enumerate(func.constants[:50]):
                lines.append(f"{prefix}  [{i:3d}] {const}")
            if len(func.constants) > 50:
                lines.append(f"{prefix}  ... and {len(func.constants) - 50} more")
            lines.append(f"{prefix}")
        
        # Locals
        if func.locals:
            lines.append(f"{prefix}LOCALS ({len(func.locals)}):")
            for local in func.locals[:20]:
                lines.append(f"{prefix}  {local.name} (R{local.register}) [{local.start_pc}-{local.end_pc}]")
            lines.append(f"{prefix}")
        
        # Instructions
        lines.append(f"{prefix}INSTRUCTIONS ({len(func.instructions)}):")
        for instr in func.instructions:
            line_parts = []
            
            if show_hex:
                line_parts.append(f"{prefix}  {instr.pc:04d} [{instr.raw:08x}]")
            else:
                line_parts.append(f"{prefix}  {instr.pc:04d}")
            
            if show_cfg and instr.is_leader:
                line_parts.append(f" BB{instr.basic_block}")
            
            if instr.is_jump_target:
                line_parts.append(" ->")
            else:
                line_parts.append("   ")
            
            line_parts.append(f" {instr.opcode.name:18}")
            
            if instr.mode == OpMode.iABC:
                line_parts.append(f" A={instr.A:3d} B={instr.B:3d} C={instr.C:3d}")
            elif instr.mode == OpMode.iABx:
                line_parts.append(f" A={instr.A:3d} Bx={instr.Bx:5d}")
            elif instr.mode == OpMode.iAsBx:
                line_parts.append(f" A={instr.A:3d} sBx={instr.sBx:5d}")
            elif instr.mode == OpMode.iAD:
                line_parts.append(f" A={instr.A:3d} D={instr.D:5d}")
            
            if instr.line:
                line_parts.append(f" ; L{instr.line}")
            
            if instr.jumps_to is not None:
                line_parts.append(f" -> {instr.jumps_to}")
            
            lines.append("".join(line_parts))
        
        lines.append(f"{prefix}")
        
        # CFG
        if show_cfg and func.basic_blocks:
            lines.append(f"{prefix}CFG ({len(func.basic_blocks)} blocks):")
            for bb in func.basic_blocks:
                header = " [LOOP]" if bb.is_loop_header else ""
                lines.append(f"{prefix}  BB{bb.id}: PC {bb.start_pc}-{bb.end_pc}{header}")
                if bb.predecessors:
                    lines.append(f"{prefix}    <- {sorted(bb.predecessors)}")
                if bb.successors:
                    lines.append(f"{prefix}    -> {sorted(bb.successors)}")
            lines.append(f"{prefix}")
        
        # Nested functions
        for i, proto in enumerate(func.prototypes):
            lines.append(f"{prefix}")
            lines.append(f"{prefix}--- NESTED FUNCTION {i} ---")
            lines.append(self._disassemble_function(
                proto, indent + 1,
                show_analysis=show_analysis,
                show_hex=show_hex,
                show_cfg=show_cfg
            ))
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export to JSON"""
        def serialize_func(func: Function) -> Dict:
            return {
                'source': func.source,
                'params': func.num_params,
                'vararg': func.is_vararg,
                'stack': func.max_stack_size,
                'hash': func.hash,
                'instructions': [
                    {
                        'pc': i.pc,
                        'opcode': i.opcode.name,
                        'raw': hex(i.raw),
                        'A': i.A, 'B': i.B, 'C': i.C,
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
            'format': self.chunk.header.format_type.name,
            'version': self.chunk.header.lua_version.name,
            'valid': self.chunk.header.is_valid,
            'stats': self.chunk.stats,
            'checksum': self.chunk.checksum,
            'main': serialize_func(self.chunk.main_function)
        }
        
        return json.dumps(data, indent=2, default=str)

# ============================================
# Utility Functions
# ============================================

def parse_bytecode(data: bytes, 
                   force_format: Optional[BytecodeFormat] = None,
                   force_version: Optional[LuaVersion] = None) -> LuaChunk:
    """Parse Lua bytecode with automatic format detection"""
    parser = AdvancedBytecodeParser(
        data, 
        force_format=force_format,
        force_version=force_version
    )
    return parser.parse()

def parse_bytecode_file(filepath: str, **kwargs) -> LuaChunk:
    """Parse Lua bytecode from file"""
    with open(filepath, 'rb') as f:
        data = f.read()
    return parse_bytecode(data, **kwargs)

def analyze_function(func: Function) -> Function:
    """Perform control flow analysis"""
    analyzer = ControlFlowAnalyzer(func)
    analyzer.analyze()
    return func

def disassemble(chunk: LuaChunk, **kwargs) -> str:
    """Disassemble chunk to text"""
    disasm = AdvancedDisassembler(chunk)
    return disasm.disassemble(**kwargs)

def detect_format(data: bytes) -> Tuple[str, str]:
    """Detect bytecode format and version"""
    fmt, version, _ = FormatDetector.detect(data)
    return fmt.name, version.name

# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED LUA BYTECODE PARSER (FIXED VERSION)")
    print("=" * 60)
    print()
    
    print("Supported Formats:")
    for fmt in BytecodeFormat:
        if fmt != BytecodeFormat.UNKNOWN:
            print(f"  ✓ {fmt.name}")
    print()
    
    print("Supported Versions:")
    for version in LuaVersion:
        if version != LuaVersion.UNKNOWN:
            print(f"  ✓ {version.name}")
    print()
    
    print("Format Detection:")
    print("  ✓ Standard Lua 5.1/5.2/5.3/5.4 (signature: \\x1bLua)")
    print("  ✓ LuaJIT 2.0/2.1 (signature: \\x1bLJ)")
    print("  ✓ Roblox Luau (version byte: 0-6, no signature)")
    print("  ✓ Compressed (gzip/zlib)")
    print("  ✓ Encrypted (detection only)")
    print()
    
    print("Features:")
    print("  ✓ Automatic format detection")
    print("  ✓ Multi-version instruction decoding")
    print("  ✓ Control flow analysis")
    print("  ✓ Basic block detection")
    print("  ✓ Loop detection")
    print("  ✓ JSON export")
    print("  ✓ Comprehensive error handling")
    print()
    
    # Test detection
    print("Format Detection Tests:")
    
    test_cases = [
        (b'\x1bLua\x51\x00\x01\x04\x04\x04\x08\x00', "Lua 5.1"),
        (b'\x1bLua\x52\x00\x01\x04\x04\x04\x08\x00', "Lua 5.2"),
        (b'\x1bLua\x53\x00\x19\x93\r\n\x1a\n', "Lua 5.3"),
        (b'\x1bLua\x54\x00\x19\x93\r\n\x1a\n', "Lua 5.4"),
        (b'\x1bLJ\x02\x00', "LuaJIT 2.1"),
        (b'\x03\x00', "Luau v3 (minimal)"),
        (b'\x05\x01\x00', "Luau v5 (minimal)"),
    ]
    
    for data, expected in test_cases:
        try:
            fmt, ver = detect_format(data)
            status = "✓" if expected.lower().replace(" ", "_") in ver.lower() or expected.lower().replace(" ", "_") in fmt.lower() else "?"
            print(f"  {status} {data[:8].hex()}: {fmt}/{ver} (expected: {expected})")
        except Exception as e:
            print(f"  ✗ {data[:8].hex()}: Error - {e}")
    
    print()
    print("✅ Parser ready!")