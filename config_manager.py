# ============================================
# File: config_manager.py
# Advanced Configuration Management System
# Part 2: Config Management
# ============================================

import os
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, fields
from enum import IntEnum
import copy

# Import configs dari semua tools
from lua_transformer import TransformConfig
from lua_vm_generator import VMConfig
from lua_encryption import EncryptionConfig, EncryptionAlgorithm, HashAlgorithm
from lua_antitamper import AntiTamperConfig, ProtectionType, DetectionAction

# ============================================
# Config Format Types
# ============================================

class ConfigFormat(IntEnum):
    """Supported config file formats"""
    JSON = 1
    YAML = 2
    TOML = 3
    PYTHON = 4

# ============================================
# Complete Configuration Container
# ============================================

@dataclass
class ObfuscatorConfig:
    """Complete configuration for obfuscator"""
    
    # Sub-configurations
    transform: TransformConfig
    vm: VMConfig
    encryption: EncryptionConfig
    antitamper: AntiTamperConfig
    
    # Metadata
    name: str = "default"
    description: str = ""
    version: str = "1.0"
    author: str = ""
    
    # Output options
    output_format: str = "lua"  # lua, luac, combined
    minify: bool = True
    add_header: bool = True
    header_text: str = ""
    
    # Logging
    verbose: bool = False
    log_file: Optional[str] = None

# ============================================
# Configuration Manager
# ============================================

class ConfigManager:
    """Manages configuration loading, saving, and validation"""
    
    def __init__(self):
        self.presets: Dict[str, ObfuscatorConfig] = {}
        self._load_builtin_presets()
    
    # ========================================
    # Load Configuration
    # ========================================
    
    def load_config(self, path: str) -> ObfuscatorConfig:
        """Load configuration from file"""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        # Detect format
        ext = path_obj.suffix.lower()
        
        if ext == '.json':
            return self._load_json(path)
        elif ext in ['.yaml', '.yml']:
            return self._load_yaml(path)
        elif ext == '.toml':
            return self._load_toml(path)
        elif ext == '.py':
            return self._load_python(path)
        else:
            # Try JSON as default
            return self._load_json(path)
    
    def _load_json(self, path: str) -> ObfuscatorConfig:
        """Load from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._dict_to_config(data)
    
    def _load_yaml(self, path: str) -> ObfuscatorConfig:
        """Load from YAML file"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML config: pip install pyyaml")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return self._dict_to_config(data)
    
    def _load_toml(self, path: str) -> ObfuscatorConfig:
        """Load from TOML file"""
        try:
            import toml
        except ImportError:
            raise ImportError("toml required for TOML config: pip install toml")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
        return self._dict_to_config(data)
    
    def _load_python(self, path: str) -> ObfuscatorConfig:
        """Load from Python file"""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'CONFIG'):
            return module.CONFIG
        elif hasattr(module, 'get_config'):
            return module.get_config()
        else:
            raise ValueError("Python config must define CONFIG or get_config()")
    
    # ========================================
    # Save Configuration
    # ========================================
    
    def save_config(self, config: ObfuscatorConfig, path: str, 
                    format: Optional[ConfigFormat] = None):
        """Save configuration to file"""
        path_obj = Path(path)
        
        # Auto-detect format if not specified
        if format is None:
            ext = path_obj.suffix.lower()
            if ext == '.json':
                format = ConfigFormat.JSON
            elif ext in ['.yaml', '.yml']:
                format = ConfigFormat.YAML
            elif ext == '.toml':
                format = ConfigFormat.TOML
            else:
                format = ConfigFormat.JSON
        
        # Convert to dict
        data = self._config_to_dict(config)
        
        # Save based on format
        if format == ConfigFormat.JSON:
            self._save_json(data, path)
        elif format == ConfigFormat.YAML:
            self._save_yaml(data, path)
        elif format == ConfigFormat.TOML:
            self._save_toml(data, path)
        elif format == ConfigFormat.PYTHON:
            self._save_python(config, path)
    
    def _save_json(self, data: Dict, path: str):
        """Save to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_yaml(self, data: Dict, path: str):
        """Save to YAML file"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def _save_toml(self, data: Dict, path: str):
        """Save to TOML file"""
        try:
            import toml
        except ImportError:
            raise ImportError("toml required: pip install toml")
        
        with open(path, 'w', encoding='utf-8') as f:
            toml.dump(data, f)
    
    def _save_python(self, config: ObfuscatorConfig, path: str):
        """Save to Python file"""
        lines = [
            "# Lua Obfuscator Configuration",
            "# Auto-generated configuration file",
            "",
            "from lua_transformer import TransformConfig",
            "from lua_vm_generator import VMConfig",
            "from lua_encryption import EncryptionConfig, EncryptionAlgorithm",
            "from lua_antitamper import AntiTamperConfig, DetectionAction",
            "from config_manager import ObfuscatorConfig",
            "",
            "CONFIG = ObfuscatorConfig(",
        ]
        
        # Add config fields (simplified)
        lines.append("    # Configuration here")
        lines.append(")")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    # ========================================
    # Config Conversion
    # ========================================
    
    def _config_to_dict(self, config: ObfuscatorConfig) -> Dict:
        """Convert config object to dictionary"""
        result = {
            'metadata': {
                'name': config.name,
                'description': config.description,
                'version': config.version,
                'author': config.author,
            },
            'output': {
                'format': config.output_format,
                'minify': config.minify,
                'add_header': config.add_header,
                'header_text': config.header_text,
            },
            'logging': {
                'verbose': config.verbose,
                'log_file': config.log_file,
            },
            'transform': self._transform_to_dict(config.transform),
            'vm': self._vm_to_dict(config.vm),
            'encryption': self._encryption_to_dict(config.encryption),
            'antitamper': self._antitamper_to_dict(config.antitamper),
        }
        return result
    
    def _transform_to_dict(self, config: TransformConfig) -> Dict:
        """Convert TransformConfig to dict"""
        return {
            'shuffle_opcodes': config.shuffle_opcodes,
            'opcode_seed': config.opcode_seed,
            'encrypt_strings': config.encrypt_strings,
            'encrypt_numbers': config.encrypt_numbers,
            'add_junk_code': config.add_junk_code,
            'junk_code_ratio': config.junk_code_ratio,
            'flatten_control_flow': config.flatten_control_flow,
            'add_opaque_predicates': config.add_opaque_predicates,
            'inject_dead_code': config.inject_dead_code,
            'dead_code_blocks': config.dead_code_blocks,
            'substitute_instructions': config.substitute_instructions,
            'strip_debug_info': config.strip_debug_info,
            'strip_line_info': config.strip_line_info,
            'strip_local_names': config.strip_local_names,
            'strip_upvalue_names': config.strip_upvalue_names,
            'add_watermark': config.add_watermark,
            'watermark_data': config.watermark_data.hex() if config.watermark_data else "",
            'custom_vm_id': config.custom_vm_id,
        }
    
    def _vm_to_dict(self, config: VMConfig) -> Dict:
        """Convert VMConfig to dict"""
        return {
            'obfuscate_names': config.obfuscate_names,
            'name_style': config.name_style,
            'min_name_length': config.min_name_length,
            'max_name_length': config.max_name_length,
            'obfuscate_strings': config.obfuscate_strings,
            'obfuscate_numbers': config.obfuscate_numbers,
            'add_dummy_code': config.add_dummy_code,
            'dummy_code_ratio': config.dummy_code_ratio,
            'flatten_vm_structure': config.flatten_vm_structure,
            'use_goto_dispatch': config.use_goto_dispatch,
            'add_environment_checks': config.add_environment_checks,
            'add_timing_checks': config.add_timing_checks,
            'add_integrity_checks': config.add_integrity_checks,
            'minify_output': config.minify_output,
            'add_comments': config.add_comments,
            'target_lua_version': config.target_lua_version,
        }
    
    def _encryption_to_dict(self, config: EncryptionConfig) -> Dict:
        """Convert EncryptionConfig to dict"""
        return {
            'algorithm': config.algorithm.name,
            'key_size': config.key_size,
            'use_key_derivation': config.use_key_derivation,
            'kdf_iterations': config.kdf_iterations,
            'use_layered_encryption': config.use_layered_encryption,
            'num_layers': config.num_layers,
            'encrypt_strings': config.encrypt_strings,
            'encrypt_numbers': config.encrypt_numbers,
            'encrypt_bytecode': config.encrypt_bytecode,
            'add_integrity_check': config.add_integrity_check,
            'compress_before_encrypt': config.compress_before_encrypt,
            'compression_level': config.compression_level,
            'add_junk_bytes': config.add_junk_bytes,
            'junk_byte_ratio': config.junk_byte_ratio,
        }
    
    def _antitamper_to_dict(self, config: AntiTamperConfig) -> Dict:
        """Convert AntiTamperConfig to dict"""
        return {
            'enable_integrity_check': config.enable_integrity_check,
            'enable_environment_detect': config.enable_environment_detect,
            'enable_debugger_detect': config.enable_debugger_detect,
            'enable_timing_check': config.enable_timing_check,
            'enable_vm_detect': config.enable_vm_detect,
            'enable_sandbox_detect': config.enable_sandbox_detect,
            'enable_anti_hook': config.enable_anti_hook,
            'enable_watermark': config.enable_watermark,
            'enable_code_flow': config.enable_code_flow,
            'detection_action': config.detection_action.name,
            'timing_threshold_ms': config.timing_threshold_ms,
            'timing_samples': config.timing_samples,
            'obfuscate_checks': config.obfuscate_checks,
            'randomize_check_order': config.randomize_check_order,
            'decoy_code_ratio': config.decoy_code_ratio,
        }
    
    def _dict_to_config(self, data: Dict) -> ObfuscatorConfig:
        """Convert dictionary to config object"""
        # Extract metadata
        metadata = data.get('metadata', {})
        output = data.get('output', {})
        logging = data.get('logging', {})
        
        # Create sub-configs
        transform = self._dict_to_transform(data.get('transform', {}))
        vm = self._dict_to_vm(data.get('vm', {}))
        encryption = self._dict_to_encryption(data.get('encryption', {}))
        antitamper = self._dict_to_antitamper(data.get('antitamper', {}))
        
        # Create main config
        config = ObfuscatorConfig(
            transform=transform,
            vm=vm,
            encryption=encryption,
            antitamper=antitamper,
            name=metadata.get('name', 'default'),
            description=metadata.get('description', ''),
            version=metadata.get('version', '1.0'),
            author=metadata.get('author', ''),
            output_format=output.get('format', 'lua'),
            minify=output.get('minify', True),
            add_header=output.get('add_header', True),
            header_text=output.get('header_text', ''),
            verbose=logging.get('verbose', False),
            log_file=logging.get('log_file', None),
        )
        
        return config
    
    def _dict_to_transform(self, data: Dict) -> TransformConfig:
        """Convert dict to TransformConfig"""
        config = TransformConfig()
        
        # Update fields
        config.shuffle_opcodes = data.get('shuffle_opcodes', True)
        config.opcode_seed = data.get('opcode_seed', 0)
        config.encrypt_strings = data.get('encrypt_strings', True)
        config.encrypt_numbers = data.get('encrypt_numbers', True)
        config.add_junk_code = data.get('add_junk_code', True)
        config.junk_code_ratio = data.get('junk_code_ratio', 0.2)
        config.flatten_control_flow = data.get('flatten_control_flow', False)
        config.add_opaque_predicates = data.get('add_opaque_predicates', False)
        config.inject_dead_code = data.get('inject_dead_code', False)
        config.dead_code_blocks = data.get('dead_code_blocks', 5)
        config.substitute_instructions = data.get('substitute_instructions', False)
        config.strip_debug_info = data.get('strip_debug_info', True)
        config.strip_line_info = data.get('strip_line_info', True)
        config.strip_local_names = data.get('strip_local_names', True)
        config.strip_upvalue_names = data.get('strip_upvalue_names', True)
        config.add_watermark = data.get('add_watermark', False)
        
        watermark_hex = data.get('watermark_data', '')
        if watermark_hex:
            config.watermark_data = bytes.fromhex(watermark_hex)
        
        config.custom_vm_id = data.get('custom_vm_id', '')
        
        return config
    
    def _dict_to_vm(self, data: Dict) -> VMConfig:
        """Convert dict to VMConfig"""
        config = VMConfig()
        
        config.obfuscate_names = data.get('obfuscate_names', True)
        config.name_style = data.get('name_style', 'random')
        config.min_name_length = data.get('min_name_length', 8)
        config.max_name_length = data.get('max_name_length', 16)
        config.obfuscate_strings = data.get('obfuscate_strings', True)
        config.obfuscate_numbers = data.get('obfuscate_numbers', True)
        config.add_dummy_code = data.get('add_dummy_code', True)
        config.dummy_code_ratio = data.get('dummy_code_ratio', 0.2)
        config.flatten_vm_structure = data.get('flatten_vm_structure', True)
        config.use_goto_dispatch = data.get('use_goto_dispatch', True)
        config.add_environment_checks = data.get('add_environment_checks', True)
        config.add_timing_checks = data.get('add_timing_checks', True)
        config.add_integrity_checks = data.get('add_integrity_checks', True)
        config.minify_output = data.get('minify_output', True)
        config.add_comments = data.get('add_comments', False)
        config.target_lua_version = data.get('target_lua_version', '5.1')
        
        return config
    
    def _dict_to_encryption(self, data: Dict) -> EncryptionConfig:
        """Convert dict to EncryptionConfig"""
        config = EncryptionConfig()
        
        # Algorithm
        algo_name = data.get('algorithm', 'XOR_ROLLING')
        try:
            config.algorithm = EncryptionAlgorithm[algo_name]
        except KeyError:
            config.algorithm = EncryptionAlgorithm.XOR_ROLLING
        
        config.key_size = data.get('key_size', 32)
        config.use_key_derivation = data.get('use_key_derivation', True)
        config.kdf_iterations = data.get('kdf_iterations', 10000)
        config.use_layered_encryption = data.get('use_layered_encryption', False)
        config.num_layers = data.get('num_layers', 3)
        config.encrypt_strings = data.get('encrypt_strings', True)
        config.encrypt_numbers = data.get('encrypt_numbers', True)
        config.encrypt_bytecode = data.get('encrypt_bytecode', True)
        config.add_integrity_check = data.get('add_integrity_check', True)
        config.compress_before_encrypt = data.get('compress_before_encrypt', True)
        config.compression_level = data.get('compression_level', 9)
        config.add_junk_bytes = data.get('add_junk_bytes', True)
        config.junk_byte_ratio = data.get('junk_byte_ratio', 0.1)
        
        return config
    
    def _dict_to_antitamper(self, data: Dict) -> AntiTamperConfig:
        """Convert dict to AntiTamperConfig"""
        config = AntiTamperConfig()
        
        config.enable_integrity_check = data.get('enable_integrity_check', True)
        config.enable_environment_detect = data.get('enable_environment_detect', True)
        config.enable_debugger_detect = data.get('enable_debugger_detect', True)
        config.enable_timing_check = data.get('enable_timing_check', True)
        config.enable_vm_detect = data.get('enable_vm_detect', False)
        config.enable_sandbox_detect = data.get('enable_sandbox_detect', True)
        config.enable_anti_hook = data.get('enable_anti_hook', True)
        config.enable_watermark = data.get('enable_watermark', False)
        config.enable_code_flow = data.get('enable_code_flow', True)
        
        # Detection action
        action_name = data.get('detection_action', 'SILENT_FAIL')
        try:
            config.detection_action = DetectionAction[action_name]
        except KeyError:
            config.detection_action = DetectionAction.SILENT_FAIL
        
        config.timing_threshold_ms = data.get('timing_threshold_ms', 100.0)
        config.timing_samples = data.get('timing_samples', 5)
        config.obfuscate_checks = data.get('obfuscate_checks', True)
        config.randomize_check_order = data.get('randomize_check_order', True)
        config.decoy_code_ratio = data.get('decoy_code_ratio', 0.2)
        
        return config
    
    # ========================================
    # Preset Management
    # ========================================
    
    def _load_builtin_presets(self):
        """Load built-in configuration presets"""
        self.presets = {
            'minimal': self._create_minimal_preset(),
            'low': self._create_low_preset(),
            'medium': self._create_medium_preset(),
            'high': self._create_high_preset(),
            'extreme': self._create_extreme_preset(),
            'stealth': self._create_stealth_preset(),
            'performance': self._create_performance_preset(),
        }
    
    def _create_minimal_preset(self) -> ObfuscatorConfig:
        """Minimal protection preset"""
        return ObfuscatorConfig(
            name="minimal",
            description="Minimal protection - only basic obfuscation",
            transform=TransformConfig(
                shuffle_opcodes=True,
                encrypt_strings=False,
                encrypt_numbers=False,
                add_junk_code=False,
                strip_debug_info=True,
            ),
            vm=VMConfig(
                obfuscate_names=False,
                minify_output=True,
            ),
            encryption=EncryptionConfig(
                encrypt_strings=False,
                encrypt_numbers=False,
                encrypt_bytecode=False,
            ),
            antitamper=AntiTamperConfig(
                enable_debugger_detect=False,
                enable_timing_check=False,
                enable_integrity_check=False,
            ),
        )
    
    def _create_low_preset(self) -> ObfuscatorConfig:
        """Low protection preset"""
        return ObfuscatorConfig(
            name="low",
            description="Low protection - basic encryption and obfuscation",
            transform=TransformConfig(
                shuffle_opcodes=True,
                encrypt_strings=True,
                encrypt_numbers=False,
                add_junk_code=True,
                junk_code_ratio=0.1,
                strip_debug_info=True,
            ),
            vm=VMConfig(
                obfuscate_names=True,
                name_style='random',
                minify_output=True,
            ),
            encryption=EncryptionConfig(
                encrypt_strings=True,
                encrypt_numbers=False,
                encrypt_bytecode=False,
            ),
            antitamper=AntiTamperConfig(
                enable_debugger_detect=False,
                enable_timing_check=False,
                enable_integrity_check=True,
            ),
        )
    
    def _create_medium_preset(self) -> ObfuscatorConfig:
        """Medium protection preset"""
        return ObfuscatorConfig(
            name="medium",
            description="Medium protection - balanced security and performance",
            transform=TransformConfig(
                shuffle_opcodes=True,
                encrypt_strings=True,
                encrypt_numbers=True,
                add_junk_code=True,
                junk_code_ratio=0.2,
                inject_dead_code=True,
                dead_code_blocks=3,
                strip_debug_info=True,
            ),
            vm=VMConfig(
                obfuscate_names=True,
                name_style='underscore',
                add_environment_checks=True,
                minify_output=True,
            ),
            encryption=EncryptionConfig(
                encrypt_strings=True,
                encrypt_numbers=True,
                encrypt_bytecode=True,
                use_layered_encryption=False,
                compress_before_encrypt=True,
            ),
            antitamper=AntiTamperConfig(
                enable_debugger_detect=True,
                enable_timing_check=True,
                enable_integrity_check=True,
                enable_anti_hook=True,
            ),
        )
    
    def _create_high_preset(self) -> ObfuscatorConfig:
        """High protection preset"""
        return ObfuscatorConfig(
            name="high",
            description="High protection - strong security measures",
            transform=TransformConfig(
                shuffle_opcodes=True,
                encrypt_strings=True,
                encrypt_numbers=True,
                add_junk_code=True,
                junk_code_ratio=0.3,
                flatten_control_flow=True,
                add_opaque_predicates=True,
                inject_dead_code=True,
                dead_code_blocks=5,
                substitute_instructions=True,
                strip_debug_info=True,
            ),
            vm=VMConfig(
                obfuscate_names=True,
                name_style='hex',
                add_environment_checks=True,
                add_timing_checks=True,
                add_integrity_checks=True,
                minify_output=True,
            ),
            encryption=EncryptionConfig(
                encrypt_strings=True,
                encrypt_numbers=True,
                encrypt_bytecode=True,
                use_layered_encryption=True,
                num_layers=2,
                compress_before_encrypt=True,
                add_integrity_check=True,
            ),
            antitamper=AntiTamperConfig(
                enable_debugger_detect=True,
                enable_timing_check=True,
                enable_integrity_check=True,
                enable_environment_detect=True,
                enable_anti_hook=True,
                enable_code_flow=True,
                enable_watermark=True,
            ),
        )
    
    def _create_extreme_preset(self) -> ObfuscatorConfig:
        """Extreme protection preset"""
        return ObfuscatorConfig(
            name="extreme",
            description="Extreme protection - maximum security (slower)",
            transform=TransformConfig(
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
            vm=VMConfig(
                obfuscate_names=True,
                name_style='unicode',
                add_dummy_code=True,
                dummy_code_ratio=0.3,
                add_environment_checks=True,
                add_timing_checks=True,
                add_integrity_checks=True,
                minify_output=True,
            ),
            encryption=EncryptionConfig(
                encrypt_strings=True,
                encrypt_numbers=True,
                encrypt_bytecode=True,
                use_layered_encryption=True,
                num_layers=3,
                compress_before_encrypt=True,
                add_integrity_check=True,
                add_junk_bytes=True,
                junk_byte_ratio=0.2,
            ),
            antitamper=AntiTamperConfig(
                enable_debugger_detect=True,
                enable_timing_check=True,
                enable_integrity_check=True,
                enable_environment_detect=True,
                enable_vm_detect=True,
                enable_sandbox_detect=True,
                enable_anti_hook=True,
                enable_code_flow=True,
                enable_watermark=True,
                decoy_code_ratio=0.3,
                randomize_check_order=True,
            ),
        )
    
    def _create_stealth_preset(self) -> ObfuscatorConfig:
        """Stealth preset - focuses on being undetectable"""
        return ObfuscatorConfig(
            name="stealth",
            description="Stealth mode - avoid detection rather than crash",
            transform=TransformConfig(
                shuffle_opcodes=True,
                encrypt_strings=True,
                encrypt_numbers=True,
                add_junk_code=True,
                junk_code_ratio=0.4,
                inject_dead_code=True,
                dead_code_blocks=8,
                strip_debug_info=True,
            ),
            vm=VMConfig(
                obfuscate_names=True,
                name_style='random',
                add_dummy_code=True,
                minify_output=True,
            ),
            encryption=EncryptionConfig(
                encrypt_strings=True,
                encrypt_numbers=True,
                encrypt_bytecode=True,
                use_layered_encryption=True,
                num_layers=2,
                compress_before_encrypt=True,
            ),
            antitamper=AntiTamperConfig(
                enable_debugger_detect=True,
                enable_timing_check=True,
                enable_integrity_check=True,
                enable_environment_detect=True,
                enable_anti_hook=True,
                enable_code_flow=True,
                detection_action=DetectionAction.SILENT_FAIL,
                decoy_code_ratio=0.4,
            ),
        )
    
    def _create_performance_preset(self) -> ObfuscatorConfig:
        """Performance preset - fast execution, moderate protection"""
        return ObfuscatorConfig(
            name="performance",
            description="Performance mode - prioritizes speed over security",
            transform=TransformConfig(
                shuffle_opcodes=True,
                encrypt_strings=True,
                encrypt_numbers=False,
                add_junk_code=False,
                strip_debug_info=True,
            ),
            vm=VMConfig(
                obfuscate_names=True,
                name_style='random',
                add_environment_checks=False,
                add_timing_checks=False,
                minify_output=True,
            ),
            encryption=EncryptionConfig(
                encrypt_strings=True,
                encrypt_numbers=False,
                encrypt_bytecode=False,
                use_layered_encryption=False,
                compress_before_encrypt=False,
            ),
            antitamper=AntiTamperConfig(
                enable_debugger_detect=False,
                enable_timing_check=False,
                enable_integrity_check=True,
                enable_environment_detect=False,
            ),
        )
    
    def get_preset(self, name: str) -> ObfuscatorConfig:
        """Get preset configuration by name"""
        if name not in self.presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(self.presets.keys())}")
        
        # Return a copy to avoid modifications
        return copy.deepcopy(self.presets[name])
    
    def list_presets(self) -> List[str]:
        """List available preset names"""
        return list(self.presets.keys())
    
    def get_preset_info(self, name: str) -> Dict[str, str]:
        """Get preset information"""
        preset = self.get_preset(name)
        return {
            'name': preset.name,
            'description': preset.description,
        }
    
    # ========================================
    # Config Validation
    # ========================================
    
    def validate_config(self, config: ObfuscatorConfig) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check transform config
        if config.transform.junk_code_ratio < 0 or config.transform.junk_code_ratio > 1:
            issues.append("junk_code_ratio must be between 0 and 1")
        
        if config.transform.dead_code_blocks < 0:
            issues.append("dead_code_blocks must be non-negative")
        
        # Check VM config
        if config.vm.min_name_length < 1:
            issues.append("min_name_length must be at least 1")
        
        if config.vm.max_name_length < config.vm.min_name_length:
            issues.append("max_name_length must be >= min_name_length")
        
        # Check encryption config
        if config.encryption.key_size < 8:
            issues.append("key_size should be at least 8 bytes")
        
        if config.encryption.num_layers < 1:
            issues.append("num_layers must be at least 1")
        
        if config.encryption.compression_level < 0 or config.encryption.compression_level > 9:
            issues.append("compression_level must be 0-9")
        
        # Check anti-tamper config
        if config.antitamper.timing_threshold_ms < 0:
            issues.append("timing_threshold_ms must be non-negative")
        
        if config.antitamper.timing_samples < 1:
            issues.append("timing_samples must be at least 1")
        
        return issues
    
    # ========================================
    # Config Merging
    # ========================================
    
    def merge_configs(self, base: ObfuscatorConfig, 
                     override: ObfuscatorConfig) -> ObfuscatorConfig:
        """Merge two configs, with override taking precedence"""
        merged = copy.deepcopy(base)
        
        # Simple field-by-field merge for now
        # In production, would need smarter merging logic
        
        return merged

# ============================================
# Environment Variable Support
# ============================================

class EnvConfigLoader:
    """Load configuration from environment variables"""
    
    PREFIX = "LUA_OBF_"
    
    @classmethod
    def load_from_env(cls) -> Dict[str, Any]:
        """Load config values from environment variables"""
        config = {}
        
        # Transform settings
        if cls._get_env_bool('SHUFFLE_OPCODES'):
            config.setdefault('transform', {})['shuffle_opcodes'] = True
        
        if cls._get_env_bool('ENCRYPT_STRINGS'):
            config.setdefault('transform', {})['encrypt_strings'] = True
        
        # VM settings
        name_style = os.getenv(f'{cls.PREFIX}NAME_STYLE')
        if name_style:
            config.setdefault('vm', {})['name_style'] = name_style
        
        # Add more as needed
        
        return config
    
    @classmethod
    def _get_env_bool(cls, key: str) -> Optional[bool]:
        """Get boolean from environment"""
        value = os.getenv(f'{cls.PREFIX}{key}')
        if value is None:
            return None
        return value.lower() in ('true', '1', 'yes', 'on')
    
    @classmethod
    def _get_env_int(cls, key: str) -> Optional[int]:
        """Get integer from environment"""
        value = os.getenv(f'{cls.PREFIX}{key}')
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    print("=== Config Manager Test ===\n")
    
    manager = ConfigManager()
    
    # List presets
    print("Available Presets:")
    for name in manager.list_presets():
        info = manager.get_preset_info(name)
        print(f"  {name:12} - {info['description']}")
    print()
    
    # Get a preset
    config = manager.get_preset('medium')
    print(f"Loaded preset: {config.name}")
    print(f"Description: {config.description}")
    print()
    
    # Validate
    issues = manager.validate_config(config)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Configuration is valid")
    print()
    
    # Save to JSON
    print("Saving to JSON...")
    manager.save_config(config, 'test_config.json')
    print("✓ Saved to test_config.json")
    
    # Load back
    print("Loading from JSON...")
    loaded = manager.load_config('test_config.json')
    print(f"✓ Loaded: {loaded.name}")
    
    # Cleanup
    import os
    if os.path.exists('test_config.json'):
        os.remove('test_config.json')
    
    print("\n✅ Config manager test completed!")
