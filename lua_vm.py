# ============================================
# File: lua_vm_generator.py
# Advanced Custom Lua VM Generator
# Generates Lua code that executes transformed bytecode
# ============================================

import random
import string
import hashlib
import base64
import struct
import json
import zlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import IntEnum, auto
from abc import ABC, abstractmethod
import textwrap

# Import from previous modules
from lua_parser import (
    OpCode, OpMode, OPCODE_MODES,
    Instruction, Constant, Function, Local, Upvalue,
    LuaHeader, LuaChunk,
    LuaBytecodeParser, LuaBytecodeWriter,
    parse_bytecode, write_bytecode
)

from lua_transformer import (
    TransformConfig, OpcodeMapper, ConstantEncryptor,
    BytecodeTransformer
)

# ============================================
# VM Configuration
# ============================================

@dataclass
class VMConfig:
    """Configuration for VM generation"""
    
    # Naming obfuscation
    obfuscate_names: bool = True
    name_style: str = "random"  # random, underscore, hex, unicode
    min_name_length: int = 8
    max_name_length: int = 16
    
    # Code obfuscation
    obfuscate_strings: bool = True
    obfuscate_numbers: bool = True
    add_dummy_code: bool = True
    dummy_code_ratio: float = 0.2
    
    # Structure obfuscation
    flatten_vm_structure: bool = True
    use_goto_dispatch: bool = True  # Use goto for opcode dispatch (Lua 5.2+)
    use_tail_calls: bool = True
    
    # Anti-analysis
    add_environment_checks: bool = True
    add_timing_checks: bool = True
    add_integrity_checks: bool = True
    
    # Output format
    minify_output: bool = True
    add_comments: bool = False  # Debug only
    
    # Compatibility
    target_lua_version: str = "5.1"  # 5.1, 5.2, 5.3, luau
    use_bit_library: bool = True  # bit32 or bit
    
    # Performance
    use_local_caching: bool = True
    inline_simple_ops: bool = True

# ============================================
# Name Generator
# ============================================

class NameGenerator:
    """Generates obfuscated variable/function names"""
    
    def __init__(self, config: VMConfig, seed: int = None):
        self.config = config
        self.used_names: Set[str] = set()
        self.name_counter = 0
        
        if seed:
            random.seed(seed)
        
        # Reserved Lua keywords
        self.reserved = {
            'and', 'break', 'do', 'else', 'elseif', 'end',
            'false', 'for', 'function', 'goto', 'if', 'in',
            'local', 'nil', 'not', 'or', 'repeat', 'return',
            'then', 'true', 'until', 'while'
        }
        
        # Unicode characters for names (if using unicode style)
        self.unicode_chars = [
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ',
            'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'σ', 'τ',
            'φ', 'χ', 'ψ', 'ω', 'Σ', 'Ω', 'Δ', 'Φ'
        ]
    
    def generate(self, prefix: str = "") -> str:
        """Generate unique obfuscated name"""
        if not self.config.obfuscate_names:
            self.name_counter += 1
            return f"{prefix}_{self.name_counter}"
        
        for _ in range(100):  # Max attempts
            if self.config.name_style == "random":
                name = self._random_name()
            elif self.config.name_style == "underscore":
                name = self._underscore_name()
            elif self.config.name_style == "hex":
                name = self._hex_name()
            elif self.config.name_style == "unicode":
                name = self._unicode_name()
            else:
                name = self._random_name()
            
            if name not in self.used_names and name not in self.reserved:
                self.used_names.add(name)
                return name
        
        # Fallback
        self.name_counter += 1
        return f"__{self.name_counter}"
    
    def _random_name(self) -> str:
        """Generate random alphanumeric name"""
        length = random.randint(self.config.min_name_length, 
                               self.config.max_name_length)
        first = random.choice(string.ascii_letters + '_')
        rest = ''.join(random.choices(string.ascii_letters + string.digits + '_', 
                                      k=length-1))
        return first + rest
    
    def _underscore_name(self) -> str:
        """Generate underscore-heavy name like __ll1l1__"""
        length = random.randint(self.config.min_name_length,
                               self.config.max_name_length)
        chars = ['_', 'l', '1', 'I', 'i']
        first = random.choice(['_', 'l', 'I', 'i'])
        rest = ''.join(random.choices(chars, k=length-1))
        return first + rest
    
    def _hex_name(self) -> str:
        """Generate hex-like name"""
        length = random.randint(8, 12)
        return '_x' + ''.join(random.choices('0123456789abcdef', k=length))
    
    def _unicode_name(self) -> str:
        """Generate unicode name (for Lua 5.3+/LuaU)"""
        length = random.randint(4, 8)
        return ''.join(random.choices(self.unicode_chars, k=length))
    
    def generate_many(self, count: int, prefix: str = "") -> List[str]:
        """Generate multiple unique names"""
        return [self.generate(prefix) for _ in range(count)]

# ============================================
# Code Builder
# ============================================

class LuaCodeBuilder:
    """Helper for building Lua code"""
    
    def __init__(self, config: VMConfig):
        self.config = config
        self.lines: List[str] = []
        self.indent_level = 0
        self.indent_str = "  "
    
    def indent(self):
        """Increase indent"""
        self.indent_level += 1
        return self
    
    def dedent(self):
        """Decrease indent"""
        self.indent_level = max(0, self.indent_level - 1)
        return self
    
    def line(self, code: str = ""):
        """Add a line of code"""
        if code:
            self.lines.append(self.indent_str * self.indent_level + code)
        else:
            self.lines.append("")
        return self
    
    def comment(self, text: str):
        """Add comment (if enabled)"""
        if self.config.add_comments:
            self.line(f"-- {text}")
        return self
    
    def block(self, header: str, footer: str = "end"):
        """Context manager for code blocks"""
        return CodeBlock(self, header, footer)
    
    def build(self) -> str:
        """Build final code string"""
        code = "\n".join(self.lines)
        
        if self.config.minify_output:
            code = self._minify(code)
        
        return code
    
    def _minify(self, code: str) -> str:
        """Basic minification"""
        lines = []
        for line in code.split('\n'):
            # Remove comments
            if '--' in line:
                line = line[:line.index('--')]
            # Strip whitespace
            line = line.strip()
            if line:
                lines.append(line)
        
        # Join with minimal separators
        result = ' '
        for line in lines:
            if result[-1] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_':
                if line[0] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_':
                    result += ' '
            result += line
        
        return result.strip()


class CodeBlock:
    """Context manager for code blocks"""
    
    def __init__(self, builder: LuaCodeBuilder, header: str, footer: str):
        self.builder = builder
        self.header = header
        self.footer = footer
    
    def __enter__(self):
        self.builder.line(self.header)
        self.builder.indent()
        return self.builder
    
    def __exit__(self, *args):
        self.builder.dedent()
        self.builder.line(self.footer)

# ============================================
# String Obfuscator for VM Code
# ============================================

class VMStringObfuscator:
    """Obfuscates strings in VM source code"""
    
    def __init__(self, name_gen: NameGenerator):
        self.name_gen = name_gen
        self.string_table: Dict[str, str] = {}  # original -> var_name
        self.decoder_name = name_gen.generate("dec")
    
    def obfuscate(self, s: str) -> str:
        """Return obfuscated reference to string"""
        if s in self.string_table:
            return self.string_table[s]
        
        var_name = self.name_gen.generate("s")
        self.string_table[s] = var_name
        return var_name
    
    def generate_decoder(self) -> str:
        """Generate string decoder and table"""
        if not self.string_table:
            return ""
        
        lines = []
        
        # XOR decoder function
        lines.append(f"local function {self.decoder_name}(t)")
        lines.append("  local r = {}")
        lines.append("  for i = 1, #t do")
        lines.append("    r[i] = string.char((t[i] + 256 - i) % 256)")
        lines.append("  end")
        lines.append("  return table.concat(r)")
        lines.append("end")
        lines.append("")
        
        # String declarations
        for original, var_name in self.string_table.items():
            encoded = self._encode_string(original)
            lines.append(f"local {var_name} = {self.decoder_name}({encoded})")
        
        return "\n".join(lines)
    
    def _encode_string(self, s: str) -> str:
        """Encode string as byte table"""
        encoded = []
        for i, c in enumerate(s.encode('utf-8')):
            encoded.append((c + i + 1) % 256)
        return "{" + ",".join(str(b) for b in encoded) + "}"

# ============================================
# VM Instruction Handlers
# ============================================

class InstructionHandler:
    """Generates code for handling individual opcodes"""
    
    def __init__(self, name_gen: NameGenerator, config: VMConfig):
        self.name_gen = name_gen
        self.config = config
        
        # VM register names
        self.n = {
            'stack': name_gen.generate("stk"),
            'pc': name_gen.generate("pc"),
            'top': name_gen.generate("top"),
            'constants': name_gen.generate("K"),
            'upvalues': name_gen.generate("upvals"),
            'protos': name_gen.generate("protos"),
            'vararg': name_gen.generate("varg"),
            'env': name_gen.generate("env"),
            'instruction': name_gen.generate("inst"),
            'opcode': name_gen.generate("op"),
            'A': name_gen.generate("A"),
            'B': name_gen.generate("B"),
            'C': name_gen.generate("C"),
            'Bx': name_gen.generate("Bx"),
            'sBx': name_gen.generate("sBx"),
        }
    
    def generate_handler(self, opcode: OpCode, custom_opcode: int) -> str:
        """Generate handler code for opcode"""
        handlers = {
            OpCode.MOVE: self._handle_move,
            OpCode.LOADK: self._handle_loadk,
            OpCode.LOADBOOL: self._handle_loadbool,
            OpCode.LOADNIL: self._handle_loadnil,
            OpCode.GETUPVAL: self._handle_getupval,
            OpCode.GETGLOBAL: self._handle_getglobal,
            OpCode.GETTABLE: self._handle_gettable,
            OpCode.SETGLOBAL: self._handle_setglobal,
            OpCode.SETUPVAL: self._handle_setupval,
            OpCode.SETTABLE: self._handle_settable,
            OpCode.NEWTABLE: self._handle_newtable,
            OpCode.SELF: self._handle_self,
            OpCode.ADD: self._handle_add,
            OpCode.SUB: self._handle_sub,
            OpCode.MUL: self._handle_mul,
            OpCode.DIV: self._handle_div,
            OpCode.MOD: self._handle_mod,
            OpCode.POW: self._handle_pow,
            OpCode.UNM: self._handle_unm,
            OpCode.NOT: self._handle_not,
            OpCode.LEN: self._handle_len,
            OpCode.CONCAT: self._handle_concat,
            OpCode.JMP: self._handle_jmp,
            OpCode.EQ: self._handle_eq,
            OpCode.LT: self._handle_lt,
            OpCode.LE: self._handle_le,
            OpCode.TEST: self._handle_test,
            OpCode.TESTSET: self._handle_testset,
            OpCode.CALL: self._handle_call,
            OpCode.TAILCALL: self._handle_tailcall,
            OpCode.RETURN: self._handle_return,
            OpCode.FORLOOP: self._handle_forloop,
            OpCode.FORPREP: self._handle_forprep,
            OpCode.TFORLOOP: self._handle_tforloop,
            OpCode.SETLIST: self._handle_setlist,
            OpCode.CLOSE: self._handle_close,
            OpCode.CLOSURE: self._handle_closure,
            OpCode.VARARG: self._handle_vararg,
        }
        
        handler = handlers.get(opcode, self._handle_unknown)
        return handler()
    
    def _rk(self, reg: str) -> str:
        """Generate RK (register or constant) lookup"""
        n = self.n
        return f"({reg} >= 256 and {n['constants']}[{reg} - 255] or {n['stack']}[{reg}])"
    
    def _handle_move(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = {n['stack']}[{n['B']}]"
    
    def _handle_loadk(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = {n['constants']}[{n['Bx']} + 1]"
    
    def _handle_loadbool(self) -> str:
        n = self.n
        lines = [
            f"{n['stack']}[{n['A']}] = ({n['B']} ~= 0)",
            f"if {n['C']} ~= 0 then {n['pc']} = {n['pc']} + 1 end"
        ]
        return "\n".join(lines)
    
    def _handle_loadnil(self) -> str:
        n = self.n
        lines = [
            f"for i = {n['A']}, {n['B']} do",
            f"  {n['stack']}[i] = nil",
            f"end"
        ]
        return "\n".join(lines)
    
    def _handle_getupval(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = {n['upvalues']}[{n['B']} + 1]"
    
    def _handle_getglobal(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = {n['env']}[{n['constants']}[{n['Bx']} + 1]]"
    
    def _handle_gettable(self) -> str:
        n = self.n
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}] = {n['stack']}[{n['B']}][{rk_c}]"
    
    def _handle_setglobal(self) -> str:
        n = self.n
        return f"{n['env']}[{n['constants']}[{n['Bx']} + 1]] = {n['stack']}[{n['A']}]"
    
    def _handle_setupval(self) -> str:
        n = self.n
        return f"{n['upvalues']}[{n['B']} + 1] = {n['stack']}[{n['A']}]"
    
    def _handle_settable(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}][{rk_b}] = {rk_c}"
    
    def _handle_newtable(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = {{}}"
    
    def _handle_self(self) -> str:
        n = self.n
        rk_c = self._rk(n['C'])
        lines = [
            f"{n['stack']}[{n['A']} + 1] = {n['stack']}[{n['B']}]",
            f"{n['stack']}[{n['A']}] = {n['stack']}[{n['B']}][{rk_c}]"
        ]
        return "\n".join(lines)
    
    def _handle_add(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}] = {rk_b} + {rk_c}"
    
    def _handle_sub(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}] = {rk_b} - {rk_c}"
    
    def _handle_mul(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}] = {rk_b} * {rk_c}"
    
    def _handle_div(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}] = {rk_b} / {rk_c}"
    
    def _handle_mod(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}] = {rk_b} % {rk_c}"
    
    def _handle_pow(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"{n['stack']}[{n['A']}] = {rk_b} ^ {rk_c}"
    
    def _handle_unm(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = -{n['stack']}[{n['B']}]"
    
    def _handle_not(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = not {n['stack']}[{n['B']}]"
    
    def _handle_len(self) -> str:
        n = self.n
        return f"{n['stack']}[{n['A']}] = #{n['stack']}[{n['B']}]"
    
    def _handle_concat(self) -> str:
        n = self.n
        lines = [
            f"local _t = {{}}",
            f"for i = {n['B']}, {n['C']} do",
            f"  _t[#_t + 1] = {n['stack']}[i]",
            f"end",
            f"{n['stack']}[{n['A']}] = table.concat(_t)"
        ]
        return "\n".join(lines)
    
    def _handle_jmp(self) -> str:
        n = self.n
        return f"{n['pc']} = {n['pc']} + {n['sBx']}"
    
    def _handle_eq(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"if ({rk_b} == {rk_c}) ~= ({n['A']} ~= 0) then {n['pc']} = {n['pc']} + 1 end"
    
    def _handle_lt(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"if ({rk_b} < {rk_c}) ~= ({n['A']} ~= 0) then {n['pc']} = {n['pc']} + 1 end"
    
    def _handle_le(self) -> str:
        n = self.n
        rk_b = self._rk(n['B'])
        rk_c = self._rk(n['C'])
        return f"if ({rk_b} <= {rk_c}) ~= ({n['A']} ~= 0) then {n['pc']} = {n['pc']} + 1 end"
    
    def _handle_test(self) -> str:
        n = self.n
        return f"if (not not {n['stack']}[{n['A']}]) ~= ({n['C']} ~= 0) then {n['pc']} = {n['pc']} + 1 end"
    
    def _handle_testset(self) -> str:
        n = self.n
        lines = [
            f"if (not not {n['stack']}[{n['B']}]) == ({n['C']} ~= 0) then",
            f"  {n['stack']}[{n['A']}] = {n['stack']}[{n['B']}]",
            f"else",
            f"  {n['pc']} = {n['pc']} + 1",
            f"end"
        ]
        return "\n".join(lines)
    
    def _handle_call(self) -> str:
        n = self.n
        lines = [
            f"local _f = {n['stack']}[{n['A']}]",
            f"local _args = {{}}",
            f"local _nargs = {n['B']} - 1",
            f"if _nargs < 0 then _nargs = {n['top']} - {n['A']} end",
            f"for i = 1, _nargs do",
            f"  _args[i] = {n['stack']}[{n['A']} + i]",
            f"end",
            f"local _results = {{_f(unpack(_args))}}",
            f"local _nres = {n['C']} - 1",
            f"if _nres < 0 then",
            f"  for i = 1, #_results do",
            f"    {n['stack']}[{n['A']} + i - 1] = _results[i]",
            f"  end",
            f"  {n['top']} = {n['A']} + #_results - 1",
            f"else",
            f"  for i = 1, _nres do",
            f"    {n['stack']}[{n['A']} + i - 1] = _results[i]",
            f"  end",
            f"end"
        ]
        return "\n".join(lines)
    
    def _handle_tailcall(self) -> str:
        n = self.n
        lines = [
            f"local _f = {n['stack']}[{n['A']}]",
            f"local _args = {{}}",
            f"local _nargs = {n['B']} - 1",
            f"if _nargs < 0 then _nargs = {n['top']} - {n['A']} end",
            f"for i = 1, _nargs do",
            f"  _args[i] = {n['stack']}[{n['A']} + i]",
            f"end",
            f"return _f(unpack(_args))"
        ]
        return "\n".join(lines)
    
    def _handle_return(self) -> str:
        n = self.n
        lines = [
            f"local _results = {{}}",
            f"local _nres = {n['B']} - 1",
            f"if _nres < 0 then _nres = {n['top']} - {n['A']} + 1 end",
            f"for i = 0, _nres - 1 do",
            f"  _results[i + 1] = {n['stack']}[{n['A']} + i]",
            f"end",
            f"return unpack(_results)"
        ]
        return "\n".join(lines)
    
    def _handle_forloop(self) -> str:
        n = self.n
        lines = [
            f"local _step = {n['stack']}[{n['A']} + 2]",
            f"local _idx = {n['stack']}[{n['A']}] + _step",
            f"{n['stack']}[{n['A']}] = _idx",
            f"local _limit = {n['stack']}[{n['A']} + 1]",
            f"if (_step > 0 and _idx <= _limit) or (_step <= 0 and _idx >= _limit) then",
            f"  {n['pc']} = {n['pc']} + {n['sBx']}",
            f"  {n['stack']}[{n['A']} + 3] = _idx",
            f"end"
        ]
        return "\n".join(lines)
    
    def _handle_forprep(self) -> str:
        n = self.n
        lines = [
            f"{n['stack']}[{n['A']}] = {n['stack']}[{n['A']}] - {n['stack']}[{n['A']} + 2]",
            f"{n['pc']} = {n['pc']} + {n['sBx']}"
        ]
        return "\n".join(lines)
    
    def _handle_tforloop(self) -> str:
        n = self.n
        lines = [
            f"local _f = {n['stack']}[{n['A']}]",
            f"local _s = {n['stack']}[{n['A']} + 1]",
            f"local _var = {n['stack']}[{n['A']} + 2]",
            f"local _results = {{_f(_s, _var)}}",
            f"for i = 1, {n['C']} do",
            f"  {n['stack']}[{n['A']} + 2 + i] = _results[i]",
            f"end",
            f"if {n['stack']}[{n['A']} + 3] ~= nil then",
            f"  {n['stack']}[{n['A']} + 2] = {n['stack']}[{n['A']} + 3]",
            f"else",
            f"  {n['pc']} = {n['pc']} + 1",
            f"end"
        ]
        return "\n".join(lines)
    
    def _handle_setlist(self) -> str:
        n = self.n
        lines = [
            f"local _tbl = {n['stack']}[{n['A']}]",
            f"local _n = {n['B']}",
            f"if _n == 0 then _n = {n['top']} - {n['A']} end",
            f"local _off = ({n['C']} - 1) * 50",
            f"for i = 1, _n do",
            f"  _tbl[_off + i] = {n['stack']}[{n['A']} + i]",
            f"end"
        ]
        return "\n".join(lines)
    
    def _handle_close(self) -> str:
        # Close upvalues - simplified for this implementation
        return "-- close upvalues (simplified)"
    
    def _handle_closure(self) -> str:
        n = self.n
        lines = [
            f"local _proto = {n['protos']}[{n['Bx']} + 1]",
            f"local _upvals = {{}}",
            f"for i = 1, _proto.nups do",
            f"  local _inst = {n['stack']}._code[{n['pc']} + i]",
            f"  local _pseudo_op = _inst % 64",
            f"  local _pseudo_b = math.floor(_inst / 8388608) % 512",
            f"  if _pseudo_op == 0 then",  # MOVE
            f"    _upvals[i] = {n['stack']}[_pseudo_b]",
            f"  else",  # GETUPVAL
            f"    _upvals[i] = {n['upvalues']}[_pseudo_b + 1]",
            f"  end",
            f"end",
            f"{n['pc']} = {n['pc']} + _proto.nups",
            f"{n['stack']}[{n['A']}] = _proto.wrap(_upvals, {n['env']})"
        ]
        return "\n".join(lines)
    
    def _handle_vararg(self) -> str:
        n = self.n
        lines = [
            f"local _n = {n['B']} - 1",
            f"if _n < 0 then",
            f"  _n = #{n['vararg']}",
            f"  {n['top']} = {n['A']} + _n - 1",
            f"end",
            f"for i = 1, _n do",
            f"  {n['stack']}[{n['A']} + i - 1] = {n['vararg']}[i]",
            f"end"
        ]
        return "\n".join(lines)
    
    def _handle_unknown(self) -> str:
        return "error('Unknown opcode')"

# ============================================
# VM Core Generator
# ============================================

class VMCoreGenerator:
    """Generates the core VM execution loop"""
    
    def __init__(self, name_gen: NameGenerator, config: VMConfig, 
                 opcode_mapper: OpcodeMapper):
        self.name_gen = name_gen
        self.config = config
        self.opcode_mapper = opcode_mapper
        self.handler = InstructionHandler(name_gen, config)
        self.n = self.handler.n  # Share names
    
    def generate(self) -> str:
        """Generate complete VM core"""
        builder = LuaCodeBuilder(self.config)
        
        # Generate bit operations helper
        self._generate_bit_ops(builder)
        
        # Generate instruction decoder
        self._generate_decoder(builder)
        
        # Generate main execution function
        self._generate_executor(builder)
        
        # Generate wrapper/entry point
        self._generate_wrapper(builder)
        
        return builder.build()
    
    def _generate_bit_ops(self, builder: LuaCodeBuilder):
        """Generate bit operation helpers"""
        builder.comment("Bit operations")
        
        if self.config.target_lua_version == "5.1":
            # Lua 5.1 doesn't have built-in bit ops
            builder.line("local bit = bit32 or bit or require('bit')")
        else:
            builder.line("local bit = bit32 or bit")
        
        band = self.name_gen.generate("band")
        rshift = self.name_gen.generate("rshift")
        
        builder.line(f"local {band} = bit.band")
        builder.line(f"local {rshift} = bit.rshift")
        builder.line()
        
        self.n['band'] = band
        self.n['rshift'] = rshift
    
    def _generate_decoder(self, builder: LuaCodeBuilder):
        """Generate instruction decoder function"""
        builder.comment("Instruction decoder")
        
        decode_name = self.name_gen.generate("decode")
        self.n['decode'] = decode_name
        
        n = self.n
        
        builder.line(f"local function {decode_name}(inst)")
        builder.indent()
        builder.line(f"local op = {n['band']}(inst, 63)")
        builder.line(f"local a = {n['band']}({n['rshift']}(inst, 6), 255)")
        builder.line(f"local c = {n['band']}({n['rshift']}(inst, 14), 511)")
        builder.line(f"local b = {n['band']}({n['rshift']}(inst, 23), 511)")
        builder.line(f"local bx = {n['band']}({n['rshift']}(inst, 14), 262143)")
        builder.line(f"local sbx = bx - 131071")
        builder.line("return op, a, b, c, bx, sbx")
        builder.dedent()
        builder.line("end")
        builder.line()
    
    def _generate_executor(self, builder: LuaCodeBuilder):
        """Generate main execution function"""
        builder.comment("VM Executor")
        
        exec_name = self.name_gen.generate("exec")
        self.n['exec'] = exec_name
        n = self.n
        
        builder.line(f"local function {exec_name}(code, constants, protos, upvalues, env, vararg)")
        builder.indent()
        
        # Initialize VM state
        builder.line(f"local {n['stack']} = {{}}")
        builder.line(f"{n['stack']}._code = code")
        builder.line(f"local {n['pc']} = 1")
        builder.line(f"local {n['top']} = -1")
        builder.line(f"local {n['constants']} = constants")
        builder.line(f"local {n['upvalues']} = upvalues")
        builder.line(f"local {n['protos']} = protos")
        builder.line(f"local {n['env']} = env")
        builder.line(f"local {n['vararg']} = vararg or {{}}")
        builder.line()
        
        # Main execution loop
        builder.line("while true do")
        builder.indent()
        builder.line(f"local {n['instruction']} = code[{n['pc']}]")
        builder.line(f"{n['pc']} = {n['pc']} + 1")
        builder.line()
        builder.line(f"local {n['opcode']}, {n['A']}, {n['B']}, {n['C']}, {n['Bx']}, {n['sBx']} = {n['decode']}({n['instruction']})")
        builder.line()
        
        # Generate opcode dispatch
        self._generate_dispatch(builder)
        
        builder.dedent()
        builder.line("end")  # while
        
        builder.dedent()
        builder.line("end")  # function
        builder.line()
    
    def _generate_dispatch(self, builder: LuaCodeBuilder):
        """Generate opcode dispatch switch"""
        n = self.n
        
        # Get reverse mapping (custom -> original)
        reverse_map = self.opcode_mapper.custom_to_original
        
        # Generate dispatch using if-elseif chain
        first = True
        for custom_op in sorted(reverse_map.keys()):
            original_op = reverse_map[custom_op]
            
            try:
                opcode = OpCode(original_op)
            except ValueError:
                continue
            
            keyword = "if" if first else "elseif"
            first = False
            
            builder.line(f"{keyword} {n['opcode']} == {custom_op} then")
            builder.indent()
            builder.comment(f"Original: {opcode.name}")
            
            # Generate handler code
            handler_code = self.handler.generate_handler(opcode, custom_op)
            for line in handler_code.split('\n'):
                builder.line(line)
            
            builder.dedent()
        
        builder.line("else")
        builder.indent()
        builder.line("error('Invalid opcode: ' .. tostring(" + n['opcode'] + "))")
        builder.dedent()
        builder.line("end")
    
    def _generate_wrapper(self, builder: LuaCodeBuilder):
        """Generate wrapper function that creates executable"""
        builder.comment("Wrapper to create executable function")
        
        wrap_name = self.name_gen.generate("wrap")
        self.n['wrap'] = wrap_name
        n = self.n
        
        builder.line(f"local function {wrap_name}(bytecode)")
        builder.indent()
        
        builder.line("local code = bytecode.code")
        builder.line("local constants = bytecode.constants")
        builder.line("local protos = bytecode.protos or {}")
        builder.line("local numparams = bytecode.numparams or 0")
        builder.line("local is_vararg = bytecode.is_vararg or 0")
        builder.line()
        
        builder.line("-- Process nested prototypes")
        builder.line("for i, proto in ipairs(protos) do")
        builder.indent()
        builder.line(f"proto.wrap = function(upvals, env) return {wrap_name}(proto)(upvals, env) end")
        builder.dedent()
        builder.line("end")
        builder.line()
        
        builder.line("return function(upvalues, env)")
        builder.indent()
        builder.line("upvalues = upvalues or {}")
        builder.line("env = env or _G or getfenv and getfenv() or _ENV")
        builder.line()
        builder.line("return function(...)")
        builder.indent()
        builder.line("local args = {...}")
        builder.line("local vararg = {}")
        builder.line()
        builder.line("-- Handle varargs")
        builder.line("if is_vararg >= 2 then")
        builder.indent()
        builder.line("for i = numparams + 1, #args do")
        builder.indent()
        builder.line("vararg[#vararg + 1] = args[i]")
        builder.dedent()
        builder.line("end")
        builder.dedent()
        builder.line("end")
        builder.line()
        builder.line(f"return {n['exec']}(code, constants, protos, upvalues, env, vararg)")
        builder.dedent()
        builder.line("end")
        builder.dedent()
        builder.line("end")
        
        builder.dedent()
        builder.line("end")
        builder.line()
        
        builder.line(f"return {wrap_name}")

# ============================================
# Constant Decoder Generator
# ============================================

class ConstantDecoderGenerator:
    """Generates constant decryption code for VM"""
    
    def __init__(self, name_gen: NameGenerator, encryptor: ConstantEncryptor):
        self.name_gen = name_gen
        self.encryptor = encryptor
    
    def generate(self) -> str:
        """Generate constant decoder"""
        lines = []
        
        # String key as table
        key_bytes = list(self.encryptor.string_key[:32])
        key_str = ",".join(str(b) for b in key_bytes)
        
        dec_str = self.name_gen.generate("decstr")
        dec_num = self.name_gen.generate("decnum")
        proc_const = self.name_gen.generate("procconst")
        
        lines.append(f"-- Constant decoder")
        lines.append(f"local _SKEY = {{{key_str}}}")
        lines.append(f"local _NKEY = {self.encryptor.number_key}")
        lines.append("")
        
        # Key expansion
        lines.append("local function _expand(key, len)")
        lines.append("  local exp = {}")
        lines.append("  for i = 1, #key do exp[i] = key[i] end")
        lines.append("  while #exp < len do")
        lines.append("    local h = 0")
        lines.append("    for i = 1, #exp do h = (h * 31 + exp[i]) % 256 end")
        lines.append("    exp[#exp + 1] = h")
        lines.append("  end")
        lines.append("  return exp")
        lines.append("end")
        lines.append("local _EXPKEY = _expand(_SKEY, 1024)")
        lines.append("")
        
        # String decoder
        lines.append(f"local function {dec_str}(data)")
        lines.append("  if type(data) ~= 'string' then return data end")
        lines.append("  if data:sub(1, 3) ~= '__E' then return data end")
        lines.append("  local method = tonumber(data:sub(4, 4))")
        lines.append("  local encoded = data:sub(6)")
        lines.append("  -- Base64 decode")
        lines.append("  local b64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'")
        lines.append("  local decoded = {}")
        lines.append("  local val, bits = 0, 0")
        lines.append("  for i = 1, #encoded do")
        lines.append("    local c = encoded:sub(i, i)")
        lines.append("    if c ~= '=' then")
        lines.append("      local idx = b64:find(c, 1, true)")
        lines.append("      if idx then")
        lines.append("        val = val * 64 + (idx - 1)")
        lines.append("        bits = bits + 6")
        lines.append("        if bits >= 8 then")
        lines.append("          bits = bits - 8")
        lines.append("          decoded[#decoded + 1] = math.floor(val / (2 ^ bits)) % 256")
        lines.append("          val = val % (2 ^ bits)")
        lines.append("        end")
        lines.append("      end")
        lines.append("    end")
        lines.append("  end")
        lines.append("  -- Skip header (4 bytes)")
        lines.append("  local len = decoded[1] + decoded[2] * 256")
        lines.append("  local result = {}")
        lines.append("  for i = 1, len do")
        lines.append("    local b = decoded[4 + i]")
        lines.append("    local k = _EXPKEY[((i - 1) % #_EXPKEY) + 1]")
        lines.append("    result[i] = string.char((b + 256 - k) % 256)")
        lines.append("  end")
        lines.append("  return table.concat(result)")
        lines.append("end")
        lines.append("")
        
        # Number decoder
        lines.append(f"local function {dec_num}(data)")
        lines.append("  if type(data) ~= 'string' then return data end")
        lines.append("  if data:sub(1, 3) ~= '__N' then return data end")
        lines.append("  local parts = {}")
        lines.append("  for p in data:sub(4):gmatch('[^_]+') do parts[#parts + 1] = p end")
        lines.append("  -- Reconstruct number (simplified)")
        lines.append("  local p1 = tonumber(parts[2]) or 0")
        lines.append("  local p2 = tonumber(parts[3]) or 0")
        lines.append("  -- This is simplified - full IEEE754 would be more complex")
        lines.append("  return p1 + p2 * 4294967296")
        lines.append("end")
        lines.append("")
        
        # Process all constants
        lines.append(f"local function {proc_const}(constants)")
        lines.append("  local processed = {}")
        lines.append("  for i, c in ipairs(constants) do")
        lines.append(f"    c = {dec_str}(c)")
        lines.append(f"    c = {dec_num}(c)")
        lines.append("    processed[i] = c")
        lines.append("  end")
        lines.append("  return processed")
        lines.append("end")
        lines.append("")
        
        return "\n".join(lines)

# ============================================
# Anti-Tamper Generator
# ============================================

class AntiTamperGenerator:
    """Generates anti-tampering checks for VM"""
    
    def __init__(self, name_gen: NameGenerator, config: VMConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate(self) -> str:
        """Generate anti-tamper code"""
        lines = []
        
        if self.config.add_environment_checks:
            lines.extend(self._generate_env_checks())
        
        if self.config.add_timing_checks:
            lines.extend(self._generate_timing_checks())
        
        if self.config.add_integrity_checks:
            lines.extend(self._generate_integrity_checks())
        
        return "\n".join(lines)
    
    def _generate_env_checks(self) -> List[str]:
        """Generate environment detection checks"""
        check_name = self.name_gen.generate("envchk")
        
        return [
            f"-- Environment checks",
            f"local function {check_name}()",
            f"  local suspicious = false",
            f"  -- Check for debug library tampering",
            f"  if debug and debug.sethook then",
            f"    local info = debug.getinfo(1, 'S')",
            f"    -- Additional checks could go here",
            f"  end",
            f"  -- Check for common analysis tools",
            f"  local env = _G or _ENV or {{}}",
            f"  local bad = {{'decoda_output', 'mobdebug', '_TEST'}}",
            f"  for _, name in ipairs(bad) do",
            f"    if env[name] then suspicious = true end",
            f"  end",
            f"  return not suspicious",
            f"end",
            f"if not {check_name}() then error('Execution not allowed') end",
            "",
        ]
    
    def _generate_timing_checks(self) -> List[str]:
        """Generate timing-based anti-debug checks"""
        check_name = self.name_gen.generate("timechk")
        
        return [
            f"-- Timing checks",
            f"local function {check_name}()",
            f"  local t1 = os.clock and os.clock() or 0",
            f"  local sum = 0",
            f"  for i = 1, 1000 do sum = sum + i end",
            f"  local t2 = os.clock and os.clock() or 0",
            f"  -- If execution is too slow, debugger might be attached",
            f"  return (t2 - t1) < 1",
            f"end",
            "",
        ]
    
    def _generate_integrity_checks(self) -> List[str]:
        """Generate code integrity checks"""
        check_name = self.name_gen.generate("intchk")
        
        return [
            f"-- Integrity checks",
            f"local {check_name} = function(code)",
            f"  local hash = 0",
            f"  for i = 1, #code do",
            f"    hash = (hash * 31 + code[i]) % 2147483647",
            f"  end",
            f"  return hash",
            f"end",
            "",
        ]

# ============================================
# Bytecode Serializer for VM
# ============================================

class BytecodeSerializer:
    """Serializes bytecode into format for VM consumption"""
    
    def __init__(self, name_gen: NameGenerator):
        self.name_gen = name_gen
    
    def serialize_function(self, func: Function, depth: int = 0) -> str:
        """Serialize function to Lua table literal"""
        lines = []
        indent = "  " * depth
        
        lines.append(f"{indent}{{")
        
        # Code (instructions as raw integers)
        code_values = [str(instr.raw) for instr in func.instructions]
        lines.append(f"{indent}  code = {{{','.join(code_values)}}},")
        
        # Constants
        const_strs = []
        for const in func.constants:
            if const.type == 0:  # nil
                const_strs.append("nil")
            elif const.type == 1:  # bool
                const_strs.append("true" if const.value else "false")
            elif const.type == 3:  # number
                const_strs.append(str(const.value))
            elif const.type == 4:  # string
                # Escape string properly
                escaped = self._escape_string(const.value)
                const_strs.append(f'"{escaped}"')
        
        lines.append(f"{indent}  constants = {{{','.join(const_strs)}}},")
        
        # Nested prototypes
        if func.prototypes:
            lines.append(f"{indent}  protos = {{")
            for proto in func.prototypes:
                lines.append(self.serialize_function(proto, depth + 2) + ",")
            lines.append(f"{indent}  }},")
        else:
            lines.append(f"{indent}  protos = {{}},")
        
        # Function metadata
        lines.append(f"{indent}  numparams = {func.num_params},")
        lines.append(f"{indent}  is_vararg = {func.is_vararg},")
        lines.append(f"{indent}  maxstack = {func.max_stack_size},")
        lines.append(f"{indent}  nups = {func.num_upvalues},")
        
        lines.append(f"{indent}}}")
        
        return "\n".join(lines)
    
    def _escape_string(self, s: str) -> str:
        """Escape string for Lua"""
        result = []
        for c in s:
            if c == '\\':
                result.append('\\\\')
            elif c == '"':
                result.append('\\"')
            elif c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            elif ord(c) < 32 or ord(c) > 126:
                result.append(f'\\{ord(c):03d}')
            else:
                result.append(c)
        return ''.join(result)

# ============================================
# Main VM Generator
# ============================================

class LuaVMGenerator:
    """Main VM generator that combines all components"""
    
    def __init__(self, transform_config: TransformConfig, vm_config: VMConfig = None):
        self.transform_config = transform_config
        self.vm_config = vm_config or VMConfig()
        
        # Initialize components
        seed = transform_config.opcode_seed
        self.name_gen = NameGenerator(self.vm_config, seed)
        self.opcode_mapper = OpcodeMapper(seed)
        self.encryptor = ConstantEncryptor(
            transform_config.string_encryption_key,
            transform_config.number_xor_key
        )
        
        # Generators
        self.core_gen = VMCoreGenerator(self.name_gen, self.vm_config, self.opcode_mapper)
        self.const_dec_gen = ConstantDecoderGenerator(self.name_gen, self.encryptor)
        self.anti_tamper_gen = AntiTamperGenerator(self.name_gen, self.vm_config)
        self.serializer = BytecodeSerializer(self.name_gen)
    
    def generate_vm(self) -> str:
        """Generate complete VM code"""
        sections = []
        
        # Header
        sections.append(self._generate_header())
        
        # Anti-tamper checks
        if self.vm_config.add_environment_checks or \
           self.vm_config.add_timing_checks or \
           self.vm_config.add_integrity_checks:
            sections.append(self.anti_tamper_gen.generate())
        
        # Constant decoder (if encryption enabled)
        if self.transform_config.encrypt_strings or self.transform_config.encrypt_numbers:
            sections.append(self.const_dec_gen.generate())
        
        # VM Core
        sections.append(self.core_gen.generate())
        
        return "\n".join(sections)
    
    def generate_loader(self, chunk: LuaChunk) -> str:
        """Generate complete loader with embedded bytecode"""
        sections = []
        
        # VM code
        sections.append(self.generate_vm())
        
        # Serialized bytecode
        sections.append("-- Bytecode")
        sections.append("local bytecode = " + self.serializer.serialize_function(chunk.main_function))
        sections.append("")
        
        # Execution
        wrap_name = self.core_gen.n.get('wrap', 'wrap')
        sections.append(f"-- Execute")
        sections.append(f"local vm = {wrap_name}(bytecode)")
        sections.append(f"return vm({{}}, _G or _ENV)()")
        
        return "\n".join(sections)
    
    def _generate_header(self) -> str:
        """Generate VM header with metadata"""
        return f"""-- Generated Lua VM
-- VM ID: {self.transform_config.custom_vm_id}
-- Do not modify

"""

# ============================================
# High-Level API
# ============================================

def generate_vm(transform_config: TransformConfig, 
                vm_config: VMConfig = None) -> str:
    """Generate VM code only"""
    generator = LuaVMGenerator(transform_config, vm_config)
    return generator.generate_vm()


def generate_protected_script(chunk: LuaChunk,
                             transform_config: TransformConfig,
                             vm_config: VMConfig = None) -> str:
    """Generate complete protected script with VM and bytecode"""
    # Transform the chunk first
    transformer = BytecodeTransformer(transform_config)
    transformed_chunk, metadata = transformer.transform(chunk)
    
    # Generate VM with embedded bytecode
    generator = LuaVMGenerator(transform_config, vm_config)
    return generator.generate_loader(transformed_chunk)


def protect_bytecode(bytecode_data: bytes,
                    transform_config: TransformConfig = None,
                    vm_config: VMConfig = None) -> str:
    """High-level API: protect bytecode and generate executable Lua"""
    # Parse bytecode
    chunk = parse_bytecode(bytecode_data)
    
    # Use defaults if not provided
    if transform_config is None:
        transform_config = TransformConfig()
        transform_config.generate_random_values()
    
    # Generate protected script
    return generate_protected_script(chunk, transform_config, vm_config)

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    print("=== Lua VM Generator ===\n")
    
    # Create test configuration
    transform_config = TransformConfig(
        shuffle_opcodes=True,
        encrypt_strings=True,
        encrypt_numbers=True,
    )
    transform_config.generate_random_values()
    
    vm_config = VMConfig(
        obfuscate_names=True,
        name_style="underscore",
        add_environment_checks=True,
        minify_output=False,
        add_comments=True,
    )
    
    print(f"Transform Config:")
    print(f"  VM ID: {transform_config.custom_vm_id}")
    print(f"  Opcode Seed: {transform_config.opcode_seed}")
    print()
    
    print(f"VM Config:")
    print(f"  Name Style: {vm_config.name_style}")
    print(f"  Environment Checks: {vm_config.add_environment_checks}")
    print()
    
    # Generate VM
    generator = LuaVMGenerator(transform_config, vm_config)
    vm_code = generator.generate_vm()
    
    print(f"Generated VM Code: {len(vm_code)} characters")
    print()
    print("=== VM Code Preview (first 2000 chars) ===")
    print(vm_code[:2000])
    print("...")
    print()
    
    # Test name generation
    print("=== Name Generation Samples ===")
    name_gen = NameGenerator(vm_config)
    for i in range(10):
        print(f"  {name_gen.generate()}")
    print()
    
    print("✅ VM Generator initialized successfully!")
