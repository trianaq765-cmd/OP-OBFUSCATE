# ============================================
# File: luraph_style.py
# Luraph-Style Output Generator
# ============================================

import random
import string
from typing import List, Dict, Any

class LuraphStyleGenerator:
    """Generate Luraph-style obfuscated output"""
    
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        
        # Single letter variables (Luraph style)
        self.vars = list('KUcRDZNPTMLWYXVHGJQSOFIEB')
        self.var_index = 0
    
    def get_var(self) -> str:
        """Get next single letter variable"""
        var = self.vars[self.var_index % len(self.vars)]
        self.var_index += 1
        return var
    
    def binary_literal(self, num: int) -> str:
        """Convert number to binary literal (0B101, 0X1)"""
        formats = [
            lambda n: f"0B{bin(n)[2:]}",
            lambda n: f"0X{hex(n)[2:].upper()}",
            lambda n: f"0x{hex(n)[2:]}",
            lambda n: str(n),
        ]
        return random.choice(formats)(num)
    
    def generate_table_wrapper(self, vm_code: str, constants: List[Any]) -> str:
        """Wrap code in Luraph-style table return"""
        
        var_names = [self.get_var() for _ in range(15)]
        
        output = f"""-- This file was protected using Lua Obfuscator (Luraph-Style)
return({{
    {var_names[0]}=coroutine.yield,
    {var_names[1]}=string.byte,
    {var_names[2]}=string.char,
    {var_names[3]}=string.sub,
    {var_names[4]}=table.concat,
    {var_names[5]}=bit32.bor,
    {var_names[6]}=bit32.band,
    {var_names[7]}=bit32.bnot,
    {var_names[8]}=bit32.rshift,
    {var_names[9]}=bit32.lshift,
    {var_names[10]}=function({var_names[11]},{var_names[12]})
        {self.luraphify_code(vm_code, var_names)}
    end,
}})
"""
        return output
    
    def luraphify_code(self, code: str, var_names: List[str]) -> str:
        """Convert code to Luraph style"""
        K, U, c, R = var_names[11:15]
        
        luraph_code = f"""        {K}[{self.binary_literal(21)}]=(function({U},{c},{R})
            local {var_names[0]}={{{K}[{self.binary_literal(21)}]}};
            if not({c}>{U})then else return;end;
            local {var_names[1]}=({U}-{c}+{self.binary_literal(1)});
            
            local function _d(i)
                local o=(i&{self.binary_literal(63)});
                local a=((i>>{self.binary_literal(6)})&{self.binary_literal(255)});
                local b=((i>>{self.binary_literal(23)})&{self.binary_literal(511)});
                local c=((i>>{self.binary_literal(14)})&{self.binary_literal(511)});
                return o,a,b,c;
            end;
            
            local S,pc={{}},{self.binary_literal(1)};
            while true do
                local i={R}[pc];pc=pc+{self.binary_literal(1)};
                local op,A,B,C=_d(i);
                
                if op=={self.binary_literal(0)} then S[A]=S[B]
                elseif op=={self.binary_literal(1)} then S[A]={U}[B+{self.binary_literal(1)}]
                elseif op=={self.binary_literal(2)} then S[A]=S[B]+S[C]
                elseif op=={self.binary_literal(8)} then 
                    local r={{}};
                    for j={self.binary_literal(0)},B-{self.binary_literal(1)} do 
                        r[j+{self.binary_literal(1)}]=S[A+j]
                    end;
                    return table.unpack(r);
                end;
            end;
        end);
        
        return {K}[{self.binary_literal(21)}]({U},{c},{R});
"""
        return luraph_code

def apply_luraph_style(vm_code: str, constants: List[Any], seed: int = None) -> str:
    """Apply Luraph-style formatting to output"""
    generator = LuraphStyleGenerator(seed)
    output = generator.generate_table_wrapper(vm_code, constants)
    return minify_luraph(output)

def minify_luraph(code: str) -> str:
    """Minify code Luraph-style"""
    lines = []
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('--'):
            while '  ' in stripped:
                stripped = stripped.replace('  ', ' ')
            lines.append(stripped)
    
    result = ''
    for line in lines:
        if line.endswith(('then', 'do', 'else')):
            result += line
        elif line in ('{', '}', '(', ')'):
            result += line
        else:
            result += line + ';'
    
    return result
