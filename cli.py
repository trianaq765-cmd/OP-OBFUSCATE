# ============================================
# File: cli.py
# Command Line Interface untuk Lua Obfuscator
# Part 1: Basic CLI
# ============================================

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Import semua tools kita
try:
    from lua_parser import parse_bytecode_file, disassemble
    from lua_transformer import TransformConfig, BytecodeTransformer
    from lua_vm_generator import VMConfig, LuaVMGenerator, generate_protected_script
    from lua_encryption import EncryptionConfig, EncryptionManager
    from lua_antitamper import AntiTamperConfig, AntiTamperGenerator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the same directory:")
    print("  - lua_parser.py")
    print("  - lua_transformer.py")
    print("  - lua_vm_generator.py")
    print("  - lua_encryption.py")
    print("  - lua_antitamper.py")
    sys.exit(1)

# ============================================
# CLI Class
# ============================================

class LuaObfuscatorCLI:
    """Command Line Interface untuk Lua Obfuscator"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Membuat argument parser"""
        parser = argparse.ArgumentParser(
            description=f'Lua Obfuscator v{self.version} - Advanced Lua Protection Tool',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Parse bytecode
  python cli.py parse input.luac
  
  # Obfuscate with default settings
  python cli.py obfuscate input.luac -o output.lua
  
  # Obfuscate with high protection
  python cli.py obfuscate input.luac -o output.lua --level high
  
  # Show statistics
  python cli.py stats input.luac
  
For more information, visit: https://github.com/yourusername/lua-obfuscator
            """
        )
        
        parser.add_argument(
            '--version',
            action='version',
            version=f'Lua Obfuscator v{self.version}'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Parse command
        parse_parser = subparsers.add_parser('parse', help='Parse and analyze Lua bytecode')
        parse_parser.add_argument('input', help='Input bytecode file (.luac)')
        parse_parser.add_argument('-o', '--output', help='Output file for disassembly')
        parse_parser.add_argument('--json', action='store_true', help='Output in JSON format')
        parse_parser.add_argument('--cfg', action='store_true', help='Show control flow graph')
        parse_parser.add_argument('--hex', action='store_true', help='Show hexadecimal values')
        
        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show bytecode statistics')
        stats_parser.add_argument('input', help='Input bytecode file (.luac)')
        
        # Obfuscate command
        obf_parser = subparsers.add_parser('obfuscate', help='Obfuscate Lua bytecode')
        obf_parser.add_argument('input', help='Input bytecode file (.luac)')
        obf_parser.add_argument('-o', '--output', required=True, help='Output Lua file')
        obf_parser.add_argument('-l', '--level', 
                               choices=['low', 'medium', 'high', 'extreme'],
                               default='medium',
                               help='Protection level (default: medium)')
        obf_parser.add_argument('-c', '--config', help='Configuration file (JSON)')
        obf_parser.add_argument('--no-encrypt', action='store_true', 
                               help='Disable encryption')
        obf_parser.add_argument('--no-antitamper', action='store_true',
                               help='Disable anti-tamper')
        obf_parser.add_argument('--watermark', help='Add watermark text')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Generate configuration file')
        config_parser.add_argument('output', help='Output configuration file')
        config_parser.add_argument('-l', '--level',
                                  choices=['low', 'medium', 'high', 'extreme'],
                                  default='medium',
                                  help='Protection level template')
        
        return parser
    
    def run(self, args=None):
        """Jalankan CLI"""
        args = self.parser.parse_args(args)
        
        if not args.command:
            self.parser.print_help()
            return 0
        
        try:
            if args.command == 'parse':
                return self.cmd_parse(args)
            elif args.command == 'stats':
                return self.cmd_stats(args)
            elif args.command == 'obfuscate':
                return self.cmd_obfuscate(args)
            elif args.command == 'config':
                return self.cmd_config(args)
            else:
                self.parser.print_help()
                return 1
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    
    # ========================================
    # Parse Command
    # ========================================
    
    def cmd_parse(self, args) -> int:
        """Parse bytecode command"""
        print(f"Parsing: {args.input}")
        
        # Check file exists
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            return 1
        
        # Parse
        try:
            chunk = parse_bytecode_file(args.input)
        except Exception as e:
            print(f"Error parsing bytecode: {e}", file=sys.stderr)
            return 1
        
        print(f"✓ Successfully parsed")
        print(f"  Version: {hex(chunk.header.version)}")
        print(f"  Functions: {len(chunk.all_functions)}")
        print(f"  Instructions: {chunk.stats.get('total_instructions', 0)}")
        print(f"  Constants: {chunk.stats.get('total_constants', 0)}")
        print()
        
        # Disassemble
        output = disassemble(
            chunk,
            show_analysis=True,
            show_hex=args.hex,
            show_cfg=args.cfg
        )
        
        # Output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"✓ Disassembly written to: {args.output}")
        else:
            print(output)
        
        return 0
    
    # ========================================
    # Stats Command
    # ========================================
    
    def cmd_stats(self, args) -> int:
        """Show statistics command"""
        print(f"Analyzing: {args.input}")
        
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            return 1
        
        # Parse
        try:
            chunk = parse_bytecode_file(args.input)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        # Display statistics
        print("\n" + "=" * 60)
        print("BYTECODE STATISTICS")
        print("=" * 60)
        
        print(f"\nFile Information:")
        print(f"  Path: {args.input}")
        print(f"  Size: {chunk.file_size} bytes")
        print(f"  Checksum: {chunk.checksum}")
        
        print(f"\nHeader:")
        print(f"  Version: {hex(chunk.header.version)}")
        print(f"  Format: {chunk.header.format_type.name}")
        print(f"  Endianness: {'Little' if chunk.header.is_little_endian else 'Big'}")
        
        print(f"\nCode Statistics:")
        print(f"  Total Functions: {len(chunk.all_functions)}")
        print(f"  Total Instructions: {chunk.stats.get('total_instructions', 0)}")
        print(f"  Total Constants: {chunk.stats.get('total_constants', 0)}")
        print(f"  Total Strings: {chunk.stats.get('total_strings', 0)}")
        print(f"  Max Stack Size: {chunk.stats.get('max_stack', 0)}")
        
        # Opcode distribution
        opcode_count = {}
        for func in chunk.all_functions:
            for instr in func.instructions:
                opname = instr.opcode.name
                opcode_count[opname] = opcode_count.get(opname, 0) + 1
        
        print(f"\nTop 10 Opcodes:")
        for opname, count in sorted(opcode_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / chunk.stats.get('total_instructions', 1)) * 100
            print(f"  {opname:15} {count:6d} ({percentage:5.2f}%)")
        
        print()
        return 0
    
    # ========================================
    # Obfuscate Command
    # ========================================
    
    def cmd_obfuscate(self, args) -> int:
        """Obfuscate bytecode command"""
        print(f"Obfuscating: {args.input}")
        print(f"Protection Level: {args.level.upper()}")
        print()
        
        # Check input
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            return 1
        
        # Load or create config
        if args.config and os.path.exists(args.config):
            print(f"Loading config from: {args.config}")
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            configs = self._config_from_dict(config_dict)
        else:
            print(f"Using built-in {args.level} protection preset")
            configs = self._get_preset_config(args.level)
        
        # Apply CLI overrides
        if args.no_encrypt:
            configs['encryption'].encrypt_strings = False
            configs['encryption'].encrypt_numbers = False
            configs['encryption'].encrypt_bytecode = False
        
        if args.no_antitamper:
            configs['antitamper'].enable_debugger_detect = False
            configs['antitamper'].enable_integrity_check = False
        
        if args.watermark:
            configs['transform'].add_watermark = True
            configs['transform'].watermark_data = args.watermark.encode('utf-8')
        
        # Process
        start_time = time.time()
        
        try:
            result = self._obfuscate_file(args.input, args.output, configs)
        except Exception as e:
            print(f"\n✗ Obfuscation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
        
        elapsed = time.time() - start_time
        
        # Success
        print(f"\n{'=' * 60}")
        print(f"✓ Obfuscation completed successfully!")
        print(f"{'=' * 60}")
        print(f"Input:  {args.input} ({result['input_size']} bytes)")
        print(f"Output: {args.output} ({result['output_size']} bytes)")
        print(f"Ratio:  {result['output_size'] / result['input_size']:.2f}x")
        print(f"Time:   {elapsed:.2f}s")
        print()
        
        return 0
    
    def _obfuscate_file(self, input_path: str, output_path: str, 
                       configs: Dict[str, Any]) -> Dict[str, Any]:
        """Obfuscate single file"""
        print("[1/5] Parsing bytecode...")
        
        # Parse bytecode
        with open(input_path, 'rb') as f:
            bytecode = f.read()
        
        chunk = parse_bytecode_file(input_path)
        
        print(f"      ✓ Parsed {len(chunk.all_functions)} functions")
        print(f"      ✓ {chunk.stats.get('total_instructions', 0)} instructions")
        
        # Transform
        print("[2/5] Transforming bytecode...")
        transformer = BytecodeTransformer(configs['transform'])
        transformed_chunk, metadata = transformer.transform(chunk)
        print(f"      ✓ Applied {configs['transform'].num_layers if configs['transform'].use_layered_encryption else 1} transformation layers")
        
        # Generate VM
        print("[3/5] Generating custom VM...")
        generator = LuaVMGenerator(configs['transform'], configs['vm'])
        vm_code = generator.generate_vm()
        print(f"      ✓ Generated VM ({len(vm_code)} chars)")
        
        # Add anti-tamper
        print("[4/5] Adding anti-tamper protection...")
        antitamper = AntiTamperGenerator(configs['antitamper'])
        protection_code = antitamper.generate_all_protections()
        print(f"      ✓ Added protection ({len(protection_code)} chars)")
        
        # Combine
        print("[5/5] Combining components...")
        final_code = self._combine_output(
            protection_code,
            vm_code,
            generator.serializer.serialize_function(transformed_chunk.main_function)
        )
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_code)
        
        return {
            'input_size': len(bytecode),
            'output_size': len(final_code.encode('utf-8')),
            'functions': len(chunk.all_functions),
            'instructions': chunk.stats.get('total_instructions', 0),
        }
    
    def _combine_output(self, protection: str, vm: str, bytecode: str) -> str:
        """Combine all components into final output"""
        return f"""-- Protected by Lua Obfuscator v{self.version}
-- https://github.com/yourusername/lua-obfuscator
-- DO NOT MODIFY

{protection}

{vm}

-- Bytecode
local bytecode = {bytecode}

-- Execute
local wrap = _WRAP or wrap
if wrap then
    local vm = wrap(bytecode)
    return vm({{}}, _G or _ENV)()
else
    error("VM initialization failed")
end
"""
    
    # ========================================
    # Config Command
    # ========================================
    
    def cmd_config(self, args) -> int:
        """Generate config file command"""
        print(f"Generating {args.level} configuration...")
        
        configs = self._get_preset_config(args.level)
        config_dict = self._config_to_dict(configs)
        
        with open(args.output, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration written to: {args.output}")
        print()
        print("You can now edit this file and use it with:")
        print(f"  python cli.py obfuscate input.luac -o output.lua -c {args.output}")
        
        return 0
    
    # ========================================
    # Config Presets
    # ========================================
    
    def _get_preset_config(self, level: str) -> Dict[str, Any]:
        """Get preset configuration by level"""
        
        if level == 'low':
            return {
                'transform': TransformConfig(
                    shuffle_opcodes=True,
                    encrypt_strings=False,
                    encrypt_numbers=False,
                    add_junk_code=False,
                    strip_debug_info=True,
                ),
                'vm': VMConfig(
                    obfuscate_names=False,
                    minify_output=True,
                ),
                'encryption': EncryptionConfig(
                    encrypt_strings=False,
                    encrypt_numbers=False,
                    encrypt_bytecode=False,
                ),
                'antitamper': AntiTamperConfig(
                    enable_debugger_detect=False,
                    enable_timing_check=False,
                    enable_integrity_check=True,
                ),
            }
        
        elif level == 'medium':
            return {
                'transform': TransformConfig(
                    shuffle_opcodes=True,
                    encrypt_strings=True,
                    encrypt_numbers=True,
                    add_junk_code=True,
                    junk_code_ratio=0.2,
                    strip_debug_info=True,
                ),
                'vm': VMConfig(
                    obfuscate_names=True,
                    name_style='underscore',
                    minify_output=True,
                ),
                'encryption': EncryptionConfig(
                    encrypt_strings=True,
                    encrypt_numbers=True,
                    encrypt_bytecode=True,
                    use_layered_encryption=False,
                ),
                'antitamper': AntiTamperConfig(
                    enable_debugger_detect=True,
                    enable_timing_check=True,
                    enable_integrity_check=True,
                    enable_anti_hook=True,
                ),
            }
        
        elif level == 'high':
            return {
                'transform': TransformConfig(
                    shuffle_opcodes=True,
                    encrypt_strings=True,
                    encrypt_numbers=True,
                    add_junk_code=True,
                    junk_code_ratio=0.3,
                    inject_dead_code=True,
                    dead_code_blocks=5,
                    substitute_instructions=True,
                    strip_debug_info=True,
                ),
                'vm': VMConfig(
                    obfuscate_names=True,
                    name_style='hex',
                    add_environment_checks=True,
                    add_timing_checks=True,
                    minify_output=True,
                ),
                'encryption': EncryptionConfig(
                    encrypt_strings=True,
                    encrypt_numbers=True,
                    encrypt_bytecode=True,
                    use_layered_encryption=True,
                    num_layers=2,
                    compress_before_encrypt=True,
                ),
                'antitamper': AntiTamperConfig(
                    enable_debugger_detect=True,
                    enable_timing_check=True,
                    enable_integrity_check=True,
                    enable_environment_detect=True,
                    enable_anti_hook=True,
                    enable_watermark=True,
                    enable_code_flow=True,
                ),
            }
        
        else:  # extreme
            return {
                'transform': TransformConfig(
                    shuffle_opcodes=True,
                    encrypt_strings=True,
                    encrypt_numbers=True,
                    add_junk_code=True,
                    junk_code_ratio=0.5,
                    flatten_control_flow=True,
                    add_opaque_predicates=True,
                    inject_dead_code=True,
                    dead_code_blocks=10,
                    substitute_instructions=True,
                    strip_debug_info=True,
                ),
                'vm': VMConfig(
                    obfuscate_names=True,
                    name_style='unicode',
                    add_environment_checks=True,
                    add_timing_checks=True,
                    add_integrity_checks=True,
                    minify_output=True,
                ),
                'encryption': EncryptionConfig(
                    encrypt_strings=True,
                    encrypt_numbers=True,
                    encrypt_bytecode=True,
                    use_layered_encryption=True,
                    num_layers=3,
                    compress_before_encrypt=True,
                    add_integrity_check=True,
                    add_junk_bytes=True,
                ),
                'antitamper': AntiTamperConfig(
                    enable_debugger_detect=True,
                    enable_timing_check=True,
                    enable_integrity_check=True,
                    enable_environment_detect=True,
                    enable_vm_detect=True,
                    enable_sandbox_detect=True,
                    enable_anti_hook=True,
                    enable_watermark=True,
                    enable_code_flow=True,
                    decoy_code_ratio=0.3,
                    randomize_check_order=True,
                ),
            }
    
    def _config_to_dict(self, configs: Dict[str, Any]) -> Dict:
        """Convert config objects to dict for JSON"""
        result = {}
        
        for name, config in configs.items():
            config_dict = {}
            for key, value in vars(config).items():
                if not key.startswith('_'):
                    if isinstance(value, bytes):
                        config_dict[key] = value.hex()
                    elif isinstance(value, list):
                        config_dict[key] = [v.value if hasattr(v, 'value') else v for v in value]
                    elif hasattr(value, 'value'):  # Enum
                        config_dict[key] = value.value
                    else:
                        config_dict[key] = value
            
            result[name] = config_dict
        
        return result
    
    def _config_from_dict(self, config_dict: Dict) -> Dict[str, Any]:
        """Convert dict to config objects"""
        # Simplified - would need full implementation
        return self._get_preset_config('medium')

# ============================================
# Main Entry Point
# ============================================

def main():
    """Main entry point"""
    cli = LuaObfuscatorCLI()
    sys.exit(cli.run())

if __name__ == '__main__':
    main()
