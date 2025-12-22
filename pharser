# ============================================
# File: lua_parser.py
# Lua 5.1 Bytecode Parser
# ============================================

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Union, BinaryIO
from enum import IntEnum

# ============================================
# Lua 5.1 Opcodes
# ============================================

class OpCode(IntEnum):
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

# Instruction types
class OpMode(IntEnum):
    iABC = 0   # A, B, C operands
    iABx = 1   # A, Bx operands
    iAsBx = 2  # A, sBx operands (signed)

# Opcode modes mapping
OPCODE_MODES = {
    OpCode.MOVE: OpMode.iABC,
    OpCode.LOADK: OpMode.iABx,
    OpCode.LOADBOOL: OpMode.iABC,
    OpCode.LOADNIL: OpMode.iABC,
    OpCode.GETUPVAL: OpMode.iABC,
    OpCode.GETGLOBAL: OpMode.iABx,
    OpCode.GETTABLE: OpMode.iABC,
    OpCode.SETGLOBAL: OpMode.iABx,
    OpCode.SETUPVAL: OpMode.iABC,
    OpCode.SETTABLE: OpMode.iABC,
    OpCode.NEWTABLE: OpMode.iABC,
    OpCode.SELF: OpMode.iABC,
    OpCode.ADD: OpMode.iABC,
    OpCode.SUB: OpMode.iABC,
    OpCode.MUL: OpMode.iABC,
    OpCode.DIV: OpMode.iABC,
    OpCode.MOD: OpMode.iABC,
    OpCode.POW: OpMode.iABC,
    OpCode.UNM: OpMode.iABC,
    OpCode.NOT: OpMode.iABC,
    OpCode.LEN: OpMode.iABC,
    OpCode.CONCAT: OpMode.iABC,
    OpCode.JMP: OpMode.iAsBx,
    OpCode.EQ: OpMode.iABC,
    OpCode.LT: OpMode.iABC,
    OpCode.LE: OpMode.iABC,
    OpCode.TEST: OpMode.iABC,
    OpCode.TESTSET: OpMode.iABC,
    OpCode.CALL: OpMode.iABC,
    OpCode.TAILCALL: OpMode.iABC,
    OpCode.RETURN: OpMode.iABC,
    OpCode.FORLOOP: OpMode.iAsBx,
    OpCode.FORPREP: OpMode.iAsBx,
    OpCode.TFORLOOP: OpMode.iABC,
    OpCode.SETLIST: OpMode.iABC,
    OpCode.CLOSE: OpMode.iABC,
    OpCode.CLOSURE: OpMode.iABx,
    OpCode.VARARG: OpMode.iABC,
}

# ============================================
# Data Classes
# ============================================

@dataclass
class Instruction:
    """Represents a single Lua instruction"""
    raw: int
    opcode: OpCode
    A: int
    B: int = 0
    C: int = 0
    Bx: int = 0
    sBx: int = 0
    mode: OpMode = OpMode.iABC
    line: int = 0
    
    def __repr__(self):
        if self.mode == OpMode.iABC:
            return f"{self.opcode.name:12} A={self.A} B={self.B} C={self.C}"
        elif self.mode == OpMode.iABx:
            return f"{self.opcode.name:12} A={self.A} Bx={self.Bx}"
        else:
            return f"{self.opcode.name:12} A={self.A} sBx={self.sBx}"

@dataclass
class Local:
    """Local variable info"""
    name: str
    start_pc: int
    end_pc: int

@dataclass
class Upvalue:
    """Upvalue info"""
    name: str

@dataclass
class Constant:
    """Constant value"""
    type: int  # 0=nil, 1=bool, 3=number, 4=string
    value: Union[None, bool, float, str]
    
    def __repr__(self):
        if self.type == 0:
            return "nil"
        elif self.type == 1:
            return f"bool({self.value})"
        elif self.type == 3:
            return f"num({self.value})"
        elif self.type == 4:
            return f'str("{self.value}")'
        return f"unknown({self.value})"

@dataclass
class Function:
    """Lua function prototype"""
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

@dataclass
class LuaHeader:
    """Lua bytecode header"""
    signature: bytes = b'\x1bLua'
    version: int = 0x51
    format: int = 0
    endianness: int = 1
    int_size: int = 4
    size_t_size: int = 4
    instruction_size: int = 4
    number_size: int = 8
    integral_flag: int = 0

@dataclass
class LuaChunk:
    """Complete Lua bytecode chunk"""
    header: LuaHeader
    main_function: Function

# ============================================
# Bytecode Parser
# ============================================

class LuaBytecodeParser:
    """Parser for Lua 5.1 bytecode"""
    
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.header: Optional[LuaHeader] = None
        
    def read_bytes(self, n: int) -> bytes:
        """Read n bytes from buffer"""
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result
    
    def read_byte(self) -> int:
        """Read single byte"""
        result = self.data[self.pos]
        self.pos += 1
        return result
    
    def read_int(self) -> int:
        """Read integer based on header int_size"""
        size = self.header.int_size if self.header else 4
        fmt = '<i' if size == 4 else '<q'
        data = self.read_bytes(size)
        return struct.unpack(fmt, data)[0]
    
    def read_size_t(self) -> int:
        """Read size_t based on header"""
        size = self.header.size_t_size if self.header else 4
        fmt = '<I' if size == 4 else '<Q'
        data = self.read_bytes(size)
        return struct.unpack(fmt, data)[0]
    
    def read_number(self) -> float:
        """Read Lua number"""
        size = self.header.number_size if self.header else 8
        data = self.read_bytes(size)
        if size == 4:
            return struct.unpack('<f', data)[0]
        return struct.unpack('<d', data)[0]
    
    def read_string(self) -> str:
        """Read Lua string"""
        size = self.read_size_t()
        if size == 0:
            return ""
        data = self.read_bytes(size)
        # Remove trailing null byte
        return data[:-1].decode('utf-8', errors='replace')
    
    def read_instruction(self) -> int:
        """Read 32-bit instruction"""
        data = self.read_bytes(4)
        return struct.unpack('<I', data)[0]
    
    def decode_instruction(self, raw: int) -> Instruction:
        """Decode raw instruction into components"""
        # Extract fields
        opcode = OpCode(raw & 0x3F)  # bits 0-5
        A = (raw >> 6) & 0xFF        # bits 6-13
        C = (raw >> 14) & 0x1FF      # bits 14-22
        B = (raw >> 23) & 0x1FF      # bits 23-31
        Bx = (raw >> 14) & 0x3FFFF   # bits 14-31 (18 bits)
        sBx = Bx - 131071            # signed Bx
        
        mode = OPCODE_MODES.get(opcode, OpMode.iABC)
        
        return Instruction(
            raw=raw,
            opcode=opcode,
            A=A, B=B, C=C,
            Bx=Bx, sBx=sBx,
            mode=mode
        )
    
    def read_header(self) -> LuaHeader:
        """Parse bytecode header"""
        header = LuaHeader()
        
        # Signature (4 bytes)
        header.signature = self.read_bytes(4)
        if header.signature != b'\x1bLua':
            raise ValueError(f"Invalid Lua signature: {header.signature}")
        
        # Version
        header.version = self.read_byte()
        if header.version != 0x51:
            raise ValueError(f"Unsupported Lua version: {hex(header.version)}")
        
        # Format
        header.format = self.read_byte()
        
        # Endianness
        header.endianness = self.read_byte()
        
        # Sizes
        header.int_size = self.read_byte()
        header.size_t_size = self.read_byte()
        header.instruction_size = self.read_byte()
        header.number_size = self.read_byte()
        header.integral_flag = self.read_byte()
        
        self.header = header
        return header
    
    def read_constant(self) -> Constant:
        """Read a constant value"""
        const_type = self.read_byte()
        
        if const_type == 0:  # nil
            return Constant(type=0, value=None)
        elif const_type == 1:  # boolean
            value = self.read_byte() != 0
            return Constant(type=1, value=value)
        elif const_type == 3:  # number
            value = self.read_number()
            return Constant(type=3, value=value)
        elif const_type == 4:  # string
            value = self.read_string()
            return Constant(type=4, value=value)
        else:
            raise ValueError(f"Unknown constant type: {const_type}")
    
    def read_function(self) -> Function:
        """Read function prototype"""
        func = Function()
        
        # Source name
        func.source = self.read_string()
        
        # Line info
        func.line_defined = self.read_int()
        func.last_line_defined = self.read_int()
        
        # Sizes
        func.num_upvalues = self.read_byte()
        func.num_params = self.read_byte()
        func.is_vararg = self.read_byte()
        func.max_stack_size = self.read_byte()
        
        # Instructions
        num_instructions = self.read_int()
        for _ in range(num_instructions):
            raw = self.read_instruction()
            instr = self.decode_instruction(raw)
            func.instructions.append(instr)
        
        # Constants
        num_constants = self.read_int()
        for _ in range(num_constants):
            const = self.read_constant()
            func.constants.append(const)
        
        # Nested prototypes
        num_prototypes = self.read_int()
        for _ in range(num_prototypes):
            proto = self.read_function()
            func.prototypes.append(proto)
        
        # Source line positions (debug info)
        num_lines = self.read_int()
        for i in range(num_lines):
            line = self.read_int()
            func.source_lines.append(line)
            if i < len(func.instructions):
                func.instructions[i].line = line
        
        # Locals (debug info)
        num_locals = self.read_int()
        for _ in range(num_locals):
            name = self.read_string()
            start_pc = self.read_int()
            end_pc = self.read_int()
            func.locals.append(Local(name, start_pc, end_pc))
        
        # Upvalues (debug info)
        num_upvalues = self.read_int()
        for _ in range(num_upvalues):
            name = self.read_string()
            func.upvalues.append(Upvalue(name))
        
        return func
    
    def parse(self) -> LuaChunk:
        """Parse complete Lua bytecode"""
        header = self.read_header()
        main_function = self.read_function()
        return LuaChunk(header=header, main_function=main_function)

# ============================================
# Bytecode Writer (for reconstruction)
# ============================================

class LuaBytecodeWriter:
    """Writer for Lua 5.1 bytecode"""
    
    def __init__(self):
        self.data = bytearray()
        self.header: Optional[LuaHeader] = None
    
    def write_bytes(self, data: bytes):
        """Write raw bytes"""
        self.data.extend(data)
    
    def write_byte(self, value: int):
        """Write single byte"""
        self.data.append(value & 0xFF)
    
    def write_int(self, value: int):
        """Write integer"""
        size = self.header.int_size if self.header else 4
        fmt = '<i' if size == 4 else '<q'
        self.data.extend(struct.pack(fmt, value))
    
    def write_size_t(self, value: int):
        """Write size_t"""
        size = self.header.size_t_size if self.header else 4
        fmt = '<I' if size == 4 else '<Q'
        self.data.extend(struct.pack(fmt, value))
    
    def write_number(self, value: float):
        """Write Lua number"""
        size = self.header.number_size if self.header else 8
        if size == 4:
            self.data.extend(struct.pack('<f', value))
        else:
            self.data.extend(struct.pack('<d', value))
    
    def write_string(self, value: str):
        """Write Lua string"""
        if not value:
            self.write_size_t(0)
            return
        encoded = value.encode('utf-8') + b'\x00'
        self.write_size_t(len(encoded))
        self.data.extend(encoded)
    
    def write_instruction(self, instr: Instruction):
        """Write instruction"""
        self.data.extend(struct.pack('<I', instr.raw))
    
    def encode_instruction(self, opcode: OpCode, A: int, B: int = 0, 
                          C: int = 0, Bx: int = 0, sBx: int = 0) -> int:
        """Encode instruction fields to raw value"""
        mode = OPCODE_MODES.get(opcode, OpMode.iABC)
        
        raw = opcode & 0x3F
        raw |= (A & 0xFF) << 6
        
        if mode == OpMode.iABC:
            raw |= (C & 0x1FF) << 14
            raw |= (B & 0x1FF) << 23
        elif mode == OpMode.iABx:
            raw |= (Bx & 0x3FFFF) << 14
        else:  # iAsBx
            unsigned_sBx = sBx + 131071
            raw |= (unsigned_sBx & 0x3FFFF) << 14
        
        return raw
    
    def write_header(self, header: LuaHeader):
        """Write bytecode header"""
        self.header = header
        self.write_bytes(header.signature)
        self.write_byte(header.version)
        self.write_byte(header.format)
        self.write_byte(header.endianness)
        self.write_byte(header.int_size)
        self.write_byte(header.size_t_size)
        self.write_byte(header.instruction_size)
        self.write_byte(header.number_size)
        self.write_byte(header.integral_flag)
    
    def write_constant(self, const: Constant):
        """Write constant value"""
        self.write_byte(const.type)
        
        if const.type == 0:  # nil
            pass
        elif const.type == 1:  # boolean
            self.write_byte(1 if const.value else 0)
        elif const.type == 3:  # number
            self.write_number(const.value)
        elif const.type == 4:  # string
            self.write_string(const.value)
    
    def write_function(self, func: Function):
        """Write function prototype"""
        # Source name
        self.write_string(func.source)
        
        # Line info
        self.write_int(func.line_defined)
        self.write_int(func.last_line_defined)
        
        # Sizes
        self.write_byte(func.num_upvalues)
        self.write_byte(func.num_params)
        self.write_byte(func.is_vararg)
        self.write_byte(func.max_stack_size)
        
        # Instructions
        self.write_int(len(func.instructions))
        for instr in func.instructions:
            self.write_instruction(instr)
        
        # Constants
        self.write_int(len(func.constants))
        for const in func.constants:
            self.write_constant(const)
        
        # Prototypes
        self.write_int(len(func.prototypes))
        for proto in func.prototypes:
            self.write_function(proto)
        
        # Source lines
        self.write_int(len(func.source_lines))
        for line in func.source_lines:
            self.write_int(line)
        
        # Locals
        self.write_int(len(func.locals))
        for local in func.locals:
            self.write_string(local.name)
            self.write_int(local.start_pc)
            self.write_int(local.end_pc)
        
        # Upvalues
        self.write_int(len(func.upvalues))
        for upvalue in func.upvalues:
            self.write_string(upvalue.name)
    
    def write_chunk(self, chunk: LuaChunk) -> bytes:
        """Write complete chunk"""
        self.write_header(chunk.header)
        self.write_function(chunk.main_function)
        return bytes(self.data)

# ============================================
# Utility Functions
# ============================================

def parse_bytecode(data: bytes) -> LuaChunk:
    """Parse Lua bytecode from bytes"""
    parser = LuaBytecodeParser(data)
    return parser.parse()

def parse_bytecode_file(filepath: str) -> LuaChunk:
    """Parse Lua bytecode from file"""
    with open(filepath, 'rb') as f:
        data = f.read()
    return parse_bytecode(data)

def write_bytecode(chunk: LuaChunk) -> bytes:
    """Write Lua chunk to bytes"""
    writer = LuaBytecodeWriter()
    return writer.write_chunk(chunk)

def write_bytecode_file(chunk: LuaChunk, filepath: str):
    """Write Lua chunk to file"""
    data = write_bytecode(chunk)
    with open(filepath, 'wb') as f:
        f.write(data)

def disassemble_function(func: Function, indent: int = 0) -> str:
    """Disassemble function to readable format"""
    lines = []
    prefix = "  " * indent
    
    lines.append(f"{prefix}; Function: {func.source or '<main>'}")
    lines.append(f"{prefix}; Lines: {func.line_defined}-{func.last_line_defined}")
    lines.append(f"{prefix}; Params: {func.num_params}, Vararg: {func.is_vararg}")
    lines.append(f"{prefix}; Stack: {func.max_stack_size}, Upvalues: {func.num_upvalues}")
    lines.append(f"{prefix};")
    
    # Constants
    lines.append(f"{prefix}; Constants ({len(func.constants)}):")
    for i, const in enumerate(func.constants):
        lines.append(f"{prefix};   [{i}] = {const}")
    lines.append(f"{prefix};")
    
    # Locals
    if func.locals:
        lines.append(f"{prefix}; Locals ({len(func.locals)}):")
        for local in func.locals:
            lines.append(f"{prefix};   {local.name} ({local.start_pc}-{local.end_pc})")
        lines.append(f"{prefix};")
    
    # Instructions
    lines.append(f"{prefix}; Instructions ({len(func.instructions)}):")
    for i, instr in enumerate(func.instructions):
        line_info = f"[{instr.line}]" if instr.line else ""
        lines.append(f"{prefix}  {i:04d} {line_info:6} {instr}")
    
    # Nested functions
    for i, proto in enumerate(func.prototypes):
        lines.append(f"{prefix};")
        lines.append(f"{prefix}; --- Nested Function {i} ---")
        lines.append(disassemble_function(proto, indent + 1))
    
    return "\n".join(lines)

def disassemble(chunk: LuaChunk) -> str:
    """Disassemble complete chunk"""
    lines = [
        "; Lua 5.1 Bytecode Disassembly",
        f"; Version: {hex(chunk.header.version)}",
        f"; Endian: {'little' if chunk.header.endianness else 'big'}",
        ";",
    ]
    lines.append(disassemble_function(chunk.main_function))
    return "\n".join(lines)

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    # Create sample bytecode manually for testing
    # This represents: local x = 1; return x
    
    print("=== Lua 5.1 Bytecode Parser ===")
    print()
    
    # Demo: Create a simple function manually
    func = Function(
        source="@test.lua",
        line_defined=0,
        last_line_defined=0,
        num_upvalues=0,
        num_params=0,
        is_vararg=2,
        max_stack_size=2,
    )
    
    # Add constants
    func.constants.append(Constant(type=3, value=1.0))  # number 1
    func.constants.append(Constant(type=4, value="Hello"))  # string
    
    # Create writer to encode instructions
    writer = LuaBytecodeWriter()
    writer.header = LuaHeader()
    
    # LOADK A=0 Bx=0 (load constant 0 into register 0)
    raw1 = writer.encode_instruction(OpCode.LOADK, A=0, Bx=0)
    func.instructions.append(Instruction(
        raw=raw1, opcode=OpCode.LOADK, A=0, Bx=0, mode=OpMode.iABx
    ))
    
    # RETURN A=0 B=2 (return 1 value from register 0)
    raw2 = writer.encode_instruction(OpCode.RETURN, A=0, B=2, C=0)
    func.instructions.append(Instruction(
        raw=raw2, opcode=OpCode.RETURN, A=0, B=2, C=0, mode=OpMode.iABC
    ))
    
    # Create chunk
    chunk = LuaChunk(
        header=LuaHeader(),
        main_function=func
    )
    
    # Write to bytes
    bytecode = write_bytecode(chunk)
    print(f"Generated bytecode: {len(bytecode)} bytes")
    print(f"Hex: {bytecode.hex()}")
    print()
    
    # Parse back
    parsed = parse_bytecode(bytecode)
    print("Parsed successfully!")
    print()
    
    # Disassemble
    print("=== Disassembly ===")
    print(disassemble(parsed))
