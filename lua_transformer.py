# ============================================
# File: lua_transformer.py
# Advanced Lua 5.1 Bytecode Transformer
# ============================================

import random
import hashlib
import struct
import base64
import string
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Set, Any
from enum import IntEnum, auto
from abc import ABC, abstractmethod

# Import from parser
from lua_parser import (
    OpCode, OpMode, OPCODE_MODES,
    Instruction, Constant, Function, Local, Upvalue,
    LuaHeader, LuaChunk,
    LuaBytecodeParser, LuaBytecodeWriter,
    parse_bytecode, write_bytecode, disassemble
)

# ============================================
# Configuration & Settings
# ============================================

@dataclass
class TransformConfig:
    """Configuration for bytecode transformation"""
    
    # Opcode transformation
    shuffle_opcodes: bool = True
    opcode_seed: int = 0  # 0 = random
    
    # Constant encryption
    encrypt_strings: bool = True
    encrypt_numbers: bool = True
    string_encryption_key: bytes = b''  # empty = generate
    number_xor_key: int = 0  # 0 = generate
    
    # Control flow obfuscation
    add_junk_code: bool = True
    junk_code_ratio: float = 0.3  # 30% junk
    flatten_control_flow: bool = True
    add_opaque_predicates: bool = True
    
    # Dead code injection
    inject_dead_code: bool = True
    dead_code_blocks: int = 5
    
    # Instruction substitution
    substitute_instructions: bool = True
    
    # Anti-analysis
    add_anti_debug: bool = True
    add_timing_checks: bool = True
    
    # Debug info stripping
    strip_debug_info: bool = True
    strip_line_info: bool = True
    strip_local_names: bool = True
    strip_upvalue_names: bool = True
    
    # Watermarking
    add_watermark: bool = False
    watermark_data: bytes = b''
    
    # VM customization
    custom_vm_id: str = ""  # unique ID for this VM
    
    def generate_random_values(self):
        """Generate random values for empty fields"""
        if self.opcode_seed == 0:
            self.opcode_seed = random.randint(1, 2**32 - 1)
        if not self.string_encryption_key:
            self.string_encryption_key = bytes([random.randint(0, 255) for _ in range(32)])
        if self.number_xor_key == 0:
            self.number_xor_key = random.randint(1, 2**64 - 1)
        if not self.custom_vm_id:
            self.custom_vm_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

# ============================================
# Opcode Mapping System
# ============================================

class OpcodeMapper:
    """Handles opcode shuffling and mapping"""
    
    def __init__(self, seed: int):
        self.seed = seed
        self.original_to_custom: Dict[int, int] = {}
        self.custom_to_original: Dict[int, int] = {}
        self._generate_mapping()
    
    def _generate_mapping(self):
        """Generate deterministic opcode mapping based on seed"""
        random.seed(self.seed)
        
        # Get all opcodes (0-37 for Lua 5.1)
        opcodes = list(range(38))
        shuffled = opcodes.copy()
        random.shuffle(shuffled)
        
        # Create bidirectional mapping
        for original, custom in zip(opcodes, shuffled):
            self.original_to_custom[original] = custom
            self.custom_to_original[custom] = original
        
        # Reset random state
        random.seed()
    
    def to_custom(self, opcode: int) -> int:
        """Convert original opcode to custom"""
        return self.original_to_custom.get(opcode, opcode)
    
    def to_original(self, opcode: int) -> int:
        """Convert custom opcode back to original"""
        return self.custom_to_original.get(opcode, opcode)
    
    def get_mapping_table(self) -> Dict[str, int]:
        """Get mapping as named dictionary for VM generation"""
        result = {}
        for orig, custom in self.original_to_custom.items():
            try:
                name = OpCode(orig).name
                result[name] = custom
            except ValueError:
                result[f"OP_{orig}"] = custom
        return result
    
    def generate_lua_decoder(self) -> str:
        """Generate Lua code for opcode decoding in VM"""
        lines = ["local OP = {"]
        for orig, custom in sorted(self.original_to_custom.items()):
            try:
                name = OpCode(orig).name
                lines.append(f"  [{custom}] = {orig}, -- {name}")
            except ValueError:
                lines.append(f"  [{custom}] = {orig},")
        lines.append("}")
        return "\n".join(lines)

# ============================================
# Constant Encryption System
# ============================================

class ConstantEncryptor:
    """Encrypts constants in bytecode"""
    
    def __init__(self, string_key: bytes, number_key: int):
        self.string_key = string_key
        self.number_key = number_key
        self._expand_key()
    
    def _expand_key(self):
        """Expand string key for longer strings"""
        self.expanded_key = self.string_key
        while len(self.expanded_key) < 1024:
            h = hashlib.sha256(self.expanded_key).digest()
            self.expanded_key += h
    
    def encrypt_string(self, s: str) -> Tuple[bytes, int]:
        """
        Encrypt string using XOR + custom encoding
        Returns: (encrypted_bytes, method_id)
        """
        data = s.encode('utf-8')
        
        # XOR encryption
        encrypted = bytearray()
        for i, b in enumerate(data):
            key_byte = self.expanded_key[i % len(self.expanded_key)]
            encrypted.append(b ^ key_byte)
        
        # Add length prefix and checksum
        length = len(data)
        checksum = sum(data) & 0xFFFF
        
        result = struct.pack('<HH', length, checksum) + bytes(encrypted)
        return result, 1  # method 1 = XOR
    
    def decrypt_string(self, data: bytes, method: int) -> str:
        """Decrypt string (for verification)"""
        if method == 1:
            length, checksum = struct.unpack('<HH', data[:4])
            encrypted = data[4:4+length]
            
            decrypted = bytearray()
            for i, b in enumerate(encrypted):
                key_byte = self.expanded_key[i % len(self.expanded_key)]
                decrypted.append(b ^ key_byte)
            
            return bytes(decrypted).decode('utf-8')
        
        raise ValueError(f"Unknown encryption method: {method}")
    
    def encrypt_number(self, n: float) -> Tuple[int, int, int]:
        """
        Encrypt number using XOR and bit manipulation
        Returns: (part1, part2, method_id)
        """
        # Pack as double
        packed = struct.pack('<d', n)
        value = struct.unpack('<Q', packed)[0]
        
        # XOR with key
        encrypted = value ^ self.number_key
        
        # Split into two parts
        part1 = encrypted & 0xFFFFFFFF
        part2 = (encrypted >> 32) & 0xFFFFFFFF
        
        return part1, part2, 1  # method 1 = XOR split
    
    def decrypt_number(self, part1: int, part2: int, method: int) -> float:
        """Decrypt number (for verification)"""
        if method == 1:
            encrypted = part1 | (part2 << 32)
            value = encrypted ^ self.number_key
            packed = struct.pack('<Q', value)
            return struct.unpack('<d', packed)[0]
        
        raise ValueError(f"Unknown encryption method: {method}")
    
    def generate_lua_decryptor(self) -> str:
        """Generate Lua decryption functions for VM"""
        # Convert key to Lua table
        key_table = ",".join(str(b) for b in self.string_key[:32])
        
        lua_code = f'''
local __KEY = {{{key_table}}}
local __NUMKEY = {self.number_key}

local function __expand_key(key, len)
    local expanded = {{}}
    for i = 1, #key do expanded[i] = key[i] end
    local idx = #key
    while idx < len do
        local h = 0
        for i = 1, #expanded do
            h = (h * 31 + expanded[i]) % 256
        end
        idx = idx + 1
        expanded[idx] = h
    end
    return expanded
end

local __EXPANDED = __expand_key(__KEY, 1024)

local function __decrypt_string(data)
    local len = data[1] + data[2] * 256
    local result = {{}}
    for i = 1, len do
        local b = data[4 + i]
        local k = __EXPANDED[((i - 1) % #__EXPANDED) + 1]
        result[i] = string.char(bit32 and bit32.bxor(b, k) or ((b + 256 - k) % 256))
    end
    return table.concat(result)
end

local function __decrypt_number(p1, p2)
    local v = p1 + p2 * 4294967296
    v = bit32 and bit32.bxor(v, __NUMKEY) or v
    -- IEEE 754 conversion would go here
    return v
end
'''
        return lua_code

# ============================================
# Instruction Transformer
# ============================================

class InstructionTransformer:
    """Transforms individual instructions"""
    
    def __init__(self, opcode_mapper: OpcodeMapper):
        self.mapper = opcode_mapper
        self.writer = LuaBytecodeWriter()
        self.writer.header = LuaHeader()
    
    def transform_instruction(self, instr: Instruction) -> Instruction:
        """Transform single instruction with new opcode"""
        new_opcode = self.mapper.to_custom(instr.opcode)
        
        # Re-encode with new opcode
        new_raw = self._encode_with_new_opcode(instr, new_opcode)
        
        return Instruction(
            raw=new_raw,
            opcode=OpCode(instr.opcode),  # Keep original for reference
            A=instr.A,
            B=instr.B,
            C=instr.C,
            Bx=instr.Bx,
            sBx=instr.sBx,
            mode=instr.mode,
            line=instr.line
        )
    
    def _encode_with_new_opcode(self, instr: Instruction, new_opcode: int) -> int:
        """Encode instruction with new opcode value"""
        raw = new_opcode & 0x3F
        raw |= (instr.A & 0xFF) << 6
        
        if instr.mode == OpMode.iABC:
            raw |= (instr.C & 0x1FF) << 14
            raw |= (instr.B & 0x1FF) << 23
        elif instr.mode == OpMode.iABx:
            raw |= (instr.Bx & 0x3FFFF) << 14
        else:  # iAsBx
            unsigned_sBx = instr.sBx + 131071
            raw |= (unsigned_sBx & 0x3FFFF) << 14
        
        return raw

# ============================================
# Junk Code Generator
# ============================================

class JunkCodeGenerator:
    """Generates meaningless but valid instructions"""
    
    def __init__(self, max_register: int = 250):
        self.max_reg = max_register
        self.junk_patterns = [
            self._junk_move_chain,
            self._junk_arithmetic,
            self._junk_load_nil,
            self._junk_test_jump,
            self._junk_table_ops,
        ]
    
    def generate(self, count: int, avoid_registers: Set[int] = None) -> List[Instruction]:
        """Generate junk instructions"""
        if avoid_registers is None:
            avoid_registers = set()
        
        # Find safe registers for junk (high numbered)
        safe_start = max(avoid_registers) + 10 if avoid_registers else 200
        safe_regs = list(range(safe_start, min(safe_start + 20, self.max_reg)))
        
        if not safe_regs:
            safe_regs = [250, 251, 252, 253, 254]
        
        instructions = []
        for _ in range(count):
            pattern = random.choice(self.junk_patterns)
            junk = pattern(safe_regs)
            instructions.extend(junk)
        
        return instructions[:count]  # Trim to exact count
    
    def _encode(self, opcode: OpCode, A: int, B: int = 0, C: int = 0, 
                Bx: int = 0, sBx: int = 0) -> Instruction:
        """Encode a new instruction"""
        mode = OPCODE_MODES.get(opcode, OpMode.iABC)
        
        raw = opcode & 0x3F
        raw |= (A & 0xFF) << 6
        
        if mode == OpMode.iABC:
            raw |= (C & 0x1FF) << 14
            raw |= (B & 0x1FF) << 23
        elif mode == OpMode.iABx:
            raw |= (Bx & 0x3FFFF) << 14
        else:
            unsigned_sBx = sBx + 131071
            raw |= (unsigned_sBx & 0x3FFFF) << 14
        
        return Instruction(
            raw=raw, opcode=opcode, A=A, B=B, C=C,
            Bx=Bx, sBx=sBx, mode=mode
        )
    
    def _junk_move_chain(self, regs: List[int]) -> List[Instruction]:
        """Generate MOVE chain that does nothing useful"""
        if len(regs) < 2:
            return []
        
        r1, r2 = random.sample(regs, 2)
        return [
            self._encode(OpCode.MOVE, A=r1, B=r2),
            self._encode(OpCode.MOVE, A=r2, B=r1),
        ]
    
    def _junk_arithmetic(self, regs: List[int]) -> List[Instruction]:
        """Generate arithmetic that cancels out"""
        if len(regs) < 2:
            return []
        
        r1, r2 = random.sample(regs, 2)
        return [
            self._encode(OpCode.ADD, A=r1, B=r1, C=r2),
            self._encode(OpCode.SUB, A=r1, B=r1, C=r2),
        ]
    
    def _junk_load_nil(self, regs: List[int]) -> List[Instruction]:
        """Load nil to unused register"""
        r = random.choice(regs)
        return [self._encode(OpCode.LOADNIL, A=r, B=r)]
    
    def _junk_test_jump(self, regs: List[int]) -> List[Instruction]:
        """Test + skip pattern (opaque predicate)"""
        r = random.choice(regs)
        return [
            self._encode(OpCode.LOADBOOL, A=r, B=1, C=0),  # Load true
            self._encode(OpCode.TEST, A=r, B=0, C=1),      # Test true
            self._encode(OpCode.JMP, A=0, sBx=1),          # Skip next
            self._encode(OpCode.LOADNIL, A=r, B=r),        # Never executed
        ]
    
    def _junk_table_ops(self, regs: List[int]) -> List[Instruction]:
        """Table operations on temp table"""
        if len(regs) < 2:
            return []
        
        r1, r2 = random.sample(regs, 2)
        return [
            self._encode(OpCode.NEWTABLE, A=r1, B=0, C=0),
        ]

# ============================================
# Control Flow Obfuscator
# ============================================

class ControlFlowObfuscator:
    """Obfuscates control flow patterns"""
    
    def __init__(self, config: TransformConfig):
        self.config = config
        self.junk_gen = JunkCodeGenerator()
    
    def obfuscate_function(self, func: Function) -> Function:
        """Apply control flow obfuscation to function"""
        new_func = copy.deepcopy(func)
        
        # Collect used registers
        used_regs = self._find_used_registers(new_func)
        
        # Insert junk code
        if self.config.add_junk_code:
            new_func = self._insert_junk_code(new_func, used_regs)
        
        # Add opaque predicates
        if self.config.add_opaque_predicates:
            new_func = self._add_opaque_predicates(new_func, used_regs)
        
        # Flatten control flow (basic)
        if self.config.flatten_control_flow:
            new_func = self._flatten_basic(new_func)
        
        return new_func
    
    def _find_used_registers(self, func: Function) -> Set[int]:
        """Find all registers used in function"""
        used = set()
        for instr in func.instructions:
            used.add(instr.A)
            if instr.mode == OpMode.iABC:
                if instr.B < 256:
                    used.add(instr.B)
                if instr.C < 256:
                    used.add(instr.C)
        return used
    
    def _insert_junk_code(self, func: Function, used_regs: Set[int]) -> Function:
        """Insert junk instructions throughout function"""
        new_instructions = []
        junk_count = int(len(func.instructions) * self.config.junk_code_ratio)
        
        # Determine insertion points (not at jumps/returns)
        safe_points = []
        for i, instr in enumerate(func.instructions):
            if instr.opcode not in [OpCode.JMP, OpCode.RETURN, OpCode.TAILCALL,
                                    OpCode.FORLOOP, OpCode.FORPREP, OpCode.TFORLOOP]:
                safe_points.append(i)
        
        # Select random insertion points
        if safe_points:
            insert_at = set(random.sample(safe_points, 
                                         min(junk_count, len(safe_points))))
        else:
            insert_at = set()
        
        # Build new instruction list
        jump_adjust = {}  # Track how jumps need adjustment
        offset = 0
        
        for i, instr in enumerate(func.instructions):
            jump_adjust[i] = offset
            
            if i in insert_at:
                # Insert 1-3 junk instructions
                junk = self.junk_gen.generate(random.randint(1, 3), used_regs)
                new_instructions.extend(junk)
                offset += len(junk)
            
            new_instructions.append(instr)
        
        # Adjust jump targets
        for instr in new_instructions:
            if instr.opcode == OpCode.JMP:
                # Adjust sBx based on insertions
                # This is simplified - full impl would need careful tracking
                pass
        
        func.instructions = new_instructions
        return func
    
    def _add_opaque_predicates(self, func: Function, used_regs: Set[int]) -> Function:
        """Add always-true/false conditions"""
        # Find safe register
        safe_reg = max(used_regs) + 5 if used_regs else 200
        if safe_reg > 250:
            safe_reg = 250
        
        new_instructions = []
        
        for i, instr in enumerate(func.instructions):
            # Randomly add opaque predicate before some instructions
            if random.random() < 0.1:  # 10% chance
                # Create always-true condition
                pred = self._create_opaque_predicate(safe_reg)
                new_instructions.extend(pred)
            
            new_instructions.append(instr)
        
        func.instructions = new_instructions
        return func
    
    def _create_opaque_predicate(self, reg: int) -> List[Instruction]:
        """Create opaque predicate (always evaluates same way)"""
        junk = self.junk_gen
        
        # Pattern: x^2 >= 0 is always true
        return [
            junk._encode(OpCode.LOADK, A=reg, Bx=0),  # Load 0
            junk._encode(OpCode.MUL, A=reg, B=reg, C=reg),  # 0*0=0
            # Result is always 0, which is falsy in Lua
        ]
    
    def _flatten_basic(self, func: Function) -> Function:
        """Basic control flow flattening"""
        # Full CFG flattening is complex - this is simplified
        # Just reorder some basic blocks if possible
        return func

# ============================================
# Dead Code Injector
# ============================================

class DeadCodeInjector:
    """Injects unreachable code blocks"""
    
    def __init__(self):
        self.junk_gen = JunkCodeGenerator()
    
    def inject(self, func: Function, num_blocks: int) -> Function:
        """Inject dead code blocks"""
        new_func = copy.deepcopy(func)
        
        used_regs = set()
        for instr in new_func.instructions:
            used_regs.add(instr.A)
        
        safe_reg = max(used_regs) + 10 if used_regs else 200
        
        for _ in range(num_blocks):
            # Create dead block
            block = self._create_dead_block(safe_reg)
            
            # Insert at random position (but after a JMP that skips it)
            if len(new_func.instructions) > 2:
                pos = random.randint(1, len(new_func.instructions) - 1)
                
                # Add jump over the dead code
                jmp = self.junk_gen._encode(OpCode.JMP, A=0, sBx=len(block))
                
                new_func.instructions.insert(pos, jmp)
                for i, instr in enumerate(block):
                    new_func.instructions.insert(pos + 1 + i, instr)
        
        return new_func
    
    def _create_dead_block(self, start_reg: int) -> List[Instruction]:
        """Create a block of dead code"""
        block = []
        size = random.randint(3, 8)
        
        for i in range(size):
            reg = start_reg + (i % 5)
            op = random.choice([
                OpCode.MOVE, OpCode.LOADNIL, OpCode.LOADBOOL,
                OpCode.ADD, OpCode.SUB, OpCode.MUL
            ])
            
            if op == OpCode.MOVE:
                block.append(self.junk_gen._encode(op, A=reg, B=reg+1))
            elif op == OpCode.LOADNIL:
                block.append(self.junk_gen._encode(op, A=reg, B=reg))
            elif op == OpCode.LOADBOOL:
                block.append(self.junk_gen._encode(op, A=reg, B=random.randint(0,1), C=0))
            else:
                block.append(self.junk_gen._encode(op, A=reg, B=reg, C=reg+1))
        
        return block

# ============================================
# Instruction Substituter
# ============================================

class InstructionSubstituter:
    """Replaces instructions with equivalent sequences"""
    
    def __init__(self):
        self.substitutions: Dict[OpCode, Callable] = {
            OpCode.MOVE: self._sub_move,
            OpCode.ADD: self._sub_add,
            OpCode.SUB: self._sub_sub,
            OpCode.LOADNIL: self._sub_loadnil,
        }
    
    def substitute(self, func: Function) -> Function:
        """Apply instruction substitutions"""
        new_func = copy.deepcopy(func)
        new_instructions = []
        
        for instr in new_func.instructions:
            if instr.opcode in self.substitutions and random.random() < 0.3:
                # 30% chance to substitute
                substituted = self.substitutions[instr.opcode](instr)
                new_instructions.extend(substituted)
            else:
                new_instructions.append(instr)
        
        new_func.instructions = new_instructions
        return new_func
    
    def _encode(self, opcode: OpCode, A: int, B: int = 0, C: int = 0,
                Bx: int = 0, sBx: int = 0) -> Instruction:
        """Encode instruction"""
        mode = OPCODE_MODES.get(opcode, OpMode.iABC)
        
        raw = opcode & 0x3F
        raw |= (A & 0xFF) << 6
        
        if mode == OpMode.iABC:
            raw |= (C & 0x1FF) << 14
            raw |= (B & 0x1FF) << 23
        elif mode == OpMode.iABx:
            raw |= (Bx & 0x3FFFF) << 14
        else:
            unsigned_sBx = sBx + 131071
            raw |= (unsigned_sBx & 0x3FFFF) << 14
        
        return Instruction(raw=raw, opcode=opcode, A=A, B=B, C=C,
                          Bx=Bx, sBx=sBx, mode=mode)
    
    def _sub_move(self, instr: Instruction) -> List[Instruction]:
        """MOVE A B -> equivalent sequence"""
        # MOVE A B can become:
        # LOADNIL A A; SETTABLE temp A B; GETTABLE A temp A
        # Simplified: just return original for now
        return [instr]
    
    def _sub_add(self, instr: Instruction) -> List[Instruction]:
        """ADD A B C -> equivalent using SUB with negation"""
        # A = B + C can become A = B - (-C)
        # Would need UNM instruction - simplified here
        return [instr]
    
    def _sub_sub(self, instr: Instruction) -> List[Instruction]:
        """SUB A B C -> equivalent sequence"""
        return [instr]
    
    def _sub_loadnil(self, instr: Instruction) -> List[Instruction]:
        """LOADNIL can use LOADBOOL + clear pattern"""
        return [instr]

# ============================================
# Debug Info Stripper
# ============================================

class DebugStripper:
    """Removes debug information from bytecode"""
    
    @staticmethod
    def strip(func: Function, config: TransformConfig) -> Function:
        """Strip debug info from function"""
        new_func = copy.deepcopy(func)
        
        if config.strip_debug_info:
            new_func.source = ""
            new_func.line_defined = 0
            new_func.last_line_defined = 0
        
        if config.strip_line_info:
            new_func.source_lines = []
            for instr in new_func.instructions:
                instr.line = 0
        
        if config.strip_local_names:
            new_func.locals = []
        
        if config.strip_upvalue_names:
            new_func.upvalues = []
        
        # Recursively strip nested functions
        new_func.prototypes = [
            DebugStripper.strip(proto, config) 
            for proto in new_func.prototypes
        ]
        
        return new_func

# ============================================
# Watermark Embedder
# ============================================

class WatermarkEmbedder:
    """Embeds hidden watermarks in bytecode"""
    
    @staticmethod
    def embed(func: Function, data: bytes) -> Function:
        """Embed watermark data in function"""
        new_func = copy.deepcopy(func)
        
        # Encode watermark in constant pool as obfuscated string
        watermark_str = base64.b64encode(data).decode('ascii')
        
        # Add as a "dead" constant (not referenced by code)
        marker = Constant(type=4, value=f"__WM_{watermark_str}")
        new_func.constants.append(marker)
        
        return new_func
    
    @staticmethod
    def extract(func: Function) -> Optional[bytes]:
        """Extract watermark from function"""
        for const in func.constants:
            if const.type == 4 and isinstance(const.value, str):
                if const.value.startswith("__WM_"):
                    try:
                        encoded = const.value[5:]
                        return base64.b64decode(encoded)
                    except:
                        pass
        return None

# ============================================
# Main Transformer Class
# ============================================

class BytecodeTransformer:
    """Main bytecode transformation engine"""
    
    def __init__(self, config: TransformConfig = None):
        self.config = config or TransformConfig()
        self.config.generate_random_values()
        
        # Initialize components
        self.opcode_mapper = OpcodeMapper(self.config.opcode_seed)
        self.encryptor = ConstantEncryptor(
            self.config.string_encryption_key,
            self.config.number_xor_key
        )
        self.instruction_transformer = InstructionTransformer(self.opcode_mapper)
        self.cf_obfuscator = ControlFlowObfuscator(self.config)
        self.dead_code_injector = DeadCodeInjector()
        self.substituter = InstructionSubstituter()
        
        # Track transformation metadata
        self.metadata: Dict[str, Any] = {}
    
    def transform(self, chunk: LuaChunk) -> Tuple[LuaChunk, Dict[str, Any]]:
        """Transform entire Lua chunk"""
        new_chunk = copy.deepcopy(chunk)
        
        # Transform main function
        new_chunk.main_function = self._transform_function(new_chunk.main_function)
        
        # Collect metadata for VM generation
        self.metadata = {
            'opcode_seed': self.config.opcode_seed,
            'opcode_mapping': self.opcode_mapper.get_mapping_table(),
            'string_key': self.config.string_encryption_key,
            'number_key': self.config.number_xor_key,
            'vm_id': self.config.custom_vm_id,
            'encrypted_constants': [],  # Filled during transformation
        }
        
        return new_chunk, self.metadata
    
    def _transform_function(self, func: Function) -> Function:
        """Transform single function"""
        new_func = copy.deepcopy(func)
        
        # 1. Strip debug info first
        if self.config.strip_debug_info:
            new_func = DebugStripper.strip(new_func, self.config)
        
        # 2. Encrypt constants
        if self.config.encrypt_strings or self.config.encrypt_numbers:
            new_func = self._encrypt_constants(new_func)
        
        # 3. Apply instruction substitution
        if self.config.substitute_instructions:
            new_func = self.substituter.substitute(new_func)
        
        # 4. Control flow obfuscation
        new_func = self.cf_obfuscator.obfuscate_function(new_func)
        
        # 5. Inject dead code
        if self.config.inject_dead_code:
            new_func = self.dead_code_injector.inject(
                new_func, 
                self.config.dead_code_blocks
            )
        
        # 6. Transform opcodes (must be last for instructions)
        if self.config.shuffle_opcodes:
            new_func = self._transform_opcodes(new_func)
        
        # 7. Add watermark
        if self.config.add_watermark and self.config.watermark_data:
            new_func = WatermarkEmbedder.embed(new_func, self.config.watermark_data)
        
        # 8. Transform nested functions recursively
        new_func.prototypes = [
            self._transform_function(proto)
            for proto in new_func.prototypes
        ]
        
        return new_func
    
    def _encrypt_constants(self, func: Function) -> Function:
        """Encrypt constants in function"""
        new_constants = []
        
        for const in func.constants:
            if const.type == 4 and self.config.encrypt_strings:
                # Encrypt string
                encrypted, method = self.encryptor.encrypt_string(const.value)
                # Store as bytes in a special format
                encoded = base64.b64encode(encrypted).decode('ascii')
                new_const = Constant(type=4, value=f"__E{method}_{encoded}")
                new_constants.append(new_const)
                
                self.metadata.setdefault('encrypted_constants', []).append({
                    'type': 'string',
                    'method': method,
                    'original_length': len(const.value)
                })
                
            elif const.type == 3 and self.config.encrypt_numbers:
                # Encrypt number
                p1, p2, method = self.encryptor.encrypt_number(const.value)
                # Store as special string marker
                new_const = Constant(type=4, value=f"__N{method}_{p1}_{p2}")
                new_constants.append(new_const)
                
                self.metadata.setdefault('encrypted_constants', []).append({
                    'type': 'number',
                    'method': method,
                    'parts': [p1, p2]
                })
            else:
                new_constants.append(const)
        
        func.constants = new_constants
        return func
    
    def _transform_opcodes(self, func: Function) -> Function:
        """Transform instruction opcodes"""
        new_instructions = []
        
        for instr in func.instructions:
            new_instr = self.instruction_transformer.transform_instruction(instr)
            new_instructions.append(new_instr)
        
        func.instructions = new_instructions
        return func
    
    def get_vm_requirements(self) -> Dict[str, Any]:
        """Get requirements for VM generation"""
        return {
            'opcode_decoder': self.opcode_mapper.generate_lua_decoder(),
            'constant_decryptor': self.encryptor.generate_lua_decryptor(),
            'vm_id': self.config.custom_vm_id,
            'config': {
                'has_encrypted_strings': self.config.encrypt_strings,
                'has_encrypted_numbers': self.config.encrypt_numbers,
                'has_shuffled_opcodes': self.config.shuffle_opcodes,
            }
        }

# ============================================
# Transformation Pipeline
# ============================================

class TransformPipeline:
    """Pipeline for applying multiple transformations"""
    
    def __init__(self):
        self.stages: List[Tuple[str, Callable[[Function], Function]]] = []
    
    def add_stage(self, name: str, transformer: Callable[[Function], Function]):
        """Add transformation stage"""
        self.stages.append((name, transformer))
    
    def execute(self, func: Function, verbose: bool = False) -> Function:
        """Execute all transformation stages"""
        result = func
        
        for name, transformer in self.stages:
            if verbose:
                print(f"  Applying: {name}")
            result = transformer(result)
        
        return result

# ============================================
# Utility Functions
# ============================================

def transform_bytecode(data: bytes, config: TransformConfig = None) -> Tuple[bytes, Dict]:
    """Transform bytecode with given config"""
    # Parse
    chunk = parse_bytecode(data)
    
    # Transform
    transformer = BytecodeTransformer(config)
    transformed_chunk, metadata = transformer.transform(chunk)
    
    # Add VM requirements to metadata
    metadata['vm_requirements'] = transformer.get_vm_requirements()
    
    # Write back
    output = write_bytecode(transformed_chunk)
    
    return output, metadata

def transform_file(input_path: str, output_path: str, 
                   config: TransformConfig = None) -> Dict:
    """Transform bytecode file"""
    with open(input_path, 'rb') as f:
        data = f.read()
    
    transformed, metadata = transform_bytecode(data, config)
    
    with open(output_path, 'wb') as f:
        f.write(transformed)
    
    return metadata

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    print("=== Lua Bytecode Transformer ===\n")
    
    # Create test configuration
    config = TransformConfig(
        shuffle_opcodes=True,
        encrypt_strings=True,
        encrypt_numbers=True,
        add_junk_code=True,
        junk_code_ratio=0.2,
        inject_dead_code=True,
        dead_code_blocks=3,
        strip_debug_info=True,
        add_watermark=True,
        watermark_data=b"MyObfuscator_v1.0"
    )
    config.generate_random_values()
    
    print(f"Configuration:")
    print(f"  VM ID: {config.custom_vm_id}")
    print(f"  Opcode Seed: {config.opcode_seed}")
    print(f"  String Key: {config.string_encryption_key[:16].hex()}...")
    print(f"  Number Key: {config.number_xor_key}")
    print()
    
    # Create transformer
    transformer = BytecodeTransformer(config)
    
    # Show opcode mapping
    print("Opcode Mapping (first 10):")
    mapping = transformer.opcode_mapper.get_mapping_table()
    for i, (name, custom) in enumerate(list(mapping.items())[:10]):
        print(f"  {name}: {OpCode[name].value} -> {custom}")
    print()
    
    # Test encryption
    print("Encryption Test:")
    test_str = "Hello, World!"
    encrypted, method = transformer.encryptor.encrypt_string(test_str)
    decrypted = transformer.encryptor.decrypt_string(encrypted, method)
    print(f"  Original: '{test_str}'")
    print(f"  Encrypted: {encrypted.hex()[:32]}...")
    print(f"  Decrypted: '{decrypted}'")
    print(f"  Match: {test_str == decrypted}")
    print()
    
    test_num = 3.14159
    p1, p2, method = transformer.encryptor.encrypt_number(test_num)
    decrypted_num = transformer.encryptor.decrypt_number(p1, p2, method)
    print(f"  Original: {test_num}")
    print(f"  Encrypted: ({p1}, {p2})")
    print(f"  Decrypted: {decrypted_num}")
    print(f"  Match: {abs(test_num - decrypted_num) < 0.0001}")
    print()
    
    # Generate VM requirements
    vm_req = transformer.get_vm_requirements()
    print("VM Requirements Generated:")
    print(f"  Has opcode decoder: {len(vm_req['opcode_decoder'])} chars")
    print(f"  Has constant decryptor: {len(vm_req['constant_decryptor'])} chars")
    print()
    
    print("✅ Transformer initialized successfully!")
    print("\nTo transform bytecode:")
    print("  transformed, metadata = transform_bytecode(bytecode_data, config)")
