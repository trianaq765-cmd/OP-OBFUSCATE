# ============================================
# File: lua_encryption.py
# Advanced Encryption Layer for Lua Obfuscator
# Supports multiple encryption algorithms and layers
# ============================================

import os
import struct
import hashlib
import hmac
import base64
import zlib
import random
import secrets
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from enum import IntEnum, auto
from abc import ABC, abstractmethod
import json

# ============================================
# Encryption Algorithms Enum
# ============================================

class EncryptionAlgorithm(IntEnum):
    """Supported encryption algorithms"""
    XOR_SIMPLE = 1
    XOR_ROLLING = 2
    XOR_MULTIBYTE = 3
    RC4 = 4
    CHACHA20_SIMPLE = 5
    AES_CTR_SIMPLE = 6
    CUSTOM_FEISTEL = 7
    CUSTOM_SUBSTITUTION = 8
    LAYERED = 9

class HashAlgorithm(IntEnum):
    """Supported hash algorithms"""
    MD5 = 1
    SHA1 = 2
    SHA256 = 3
    SHA512 = 4
    CUSTOM = 5

# ============================================
# Encryption Configuration
# ============================================

@dataclass
class EncryptionConfig:
    """Configuration for encryption layer"""
    
    # Primary encryption
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.XOR_ROLLING
    key_size: int = 32
    
    # Key derivation
    use_key_derivation: bool = True
    kdf_iterations: int = 10000
    kdf_salt_size: int = 16
    
    # Layered encryption
    use_layered_encryption: bool = True
    num_layers: int = 3
    layer_algorithms: List[EncryptionAlgorithm] = field(default_factory=list)
    
    # String encryption
    encrypt_strings: bool = True
    string_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.XOR_ROLLING
    per_string_keys: bool = True  # Different key for each string
    
    # Number encryption
    encrypt_numbers: bool = True
    number_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.XOR_SIMPLE
    
    # Bytecode encryption
    encrypt_bytecode: bool = True
    bytecode_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RC4
    
    # Integrity
    add_integrity_check: bool = True
    integrity_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    
    # Compression
    compress_before_encrypt: bool = True
    compression_level: int = 9
    
    # Anti-analysis
    add_junk_bytes: bool = True
    junk_byte_ratio: float = 0.1
    randomize_structure: bool = True
    
    # Keys (auto-generated if empty)
    master_key: bytes = b''
    layer_keys: List[bytes] = field(default_factory=list)
    
    def generate_keys(self):
        """Generate all required keys"""
        if not self.master_key:
            self.master_key = secrets.token_bytes(self.key_size)
        
        if not self.layer_keys and self.use_layered_encryption:
            self.layer_keys = [
                secrets.token_bytes(self.key_size) 
                for _ in range(self.num_layers)
            ]
        
        if not self.layer_algorithms and self.use_layered_encryption:
            available = [
                EncryptionAlgorithm.XOR_ROLLING,
                EncryptionAlgorithm.XOR_MULTIBYTE,
                EncryptionAlgorithm.RC4,
                EncryptionAlgorithm.CUSTOM_SUBSTITUTION,
            ]
            self.layer_algorithms = [
                random.choice(available) 
                for _ in range(self.num_layers)
            ]

# ============================================
# Key Derivation Functions
# ============================================

class KeyDerivation:
    """Key derivation functions"""
    
    @staticmethod
    def pbkdf2(password: bytes, salt: bytes, iterations: int = 10000, 
               key_length: int = 32) -> bytes:
        """PBKDF2 key derivation"""
        return hashlib.pbkdf2_hmac('sha256', password, salt, iterations, key_length)
    
    @staticmethod
    def simple_kdf(password: bytes, salt: bytes, length: int = 32) -> bytes:
        """Simple key derivation using iterative hashing"""
        result = password + salt
        for _ in range(1000):
            result = hashlib.sha256(result).digest()
        
        # Extend to required length
        extended = result
        while len(extended) < length:
            extended += hashlib.sha256(extended + password).digest()
        
        return extended[:length]
    
    @staticmethod
    def derive_subkeys(master_key: bytes, num_keys: int, key_length: int = 32) -> List[bytes]:
        """Derive multiple subkeys from master key"""
        keys = []
        for i in range(num_keys):
            context = f"subkey_{i}".encode()
            derived = hashlib.sha256(master_key + context + struct.pack('<I', i)).digest()
            
            # Extend if needed
            while len(derived) < key_length:
                derived += hashlib.sha256(derived + master_key).digest()
            
            keys.append(derived[:key_length])
        
        return keys

# ============================================
# Base Cipher Interface
# ============================================

class BaseCipher(ABC):
    """Abstract base class for ciphers"""
    
    @abstractmethod
    def encrypt(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data"""
        pass
    
    @abstractmethod
    def decrypt(self, data: bytes, key: bytes) -> bytes:
        """Decrypt data"""
        pass
    
    @abstractmethod
    def generate_lua_decryptor(self, key: bytes) -> str:
        """Generate Lua code for decryption"""
        pass

# ============================================
# XOR Ciphers
# ============================================

class XORSimpleCipher(BaseCipher):
    """Simple XOR cipher with single byte key"""
    
    def encrypt(self, data: bytes, key: bytes) -> bytes:
        key_byte = key[0] if key else 0x55
        return bytes(b ^ key_byte for b in data)
    
    def decrypt(self, data: bytes, key: bytes) -> bytes:
        return self.encrypt(data, key)  # XOR is symmetric
    
    def generate_lua_decryptor(self, key: bytes) -> str:
        key_byte = key[0] if key else 0x55
        return f'''
local function xor_simple_decrypt(data, key)
    key = key or {key_byte}
    local result = {{}}
    for i = 1, #data do
        result[i] = string.char(bit32.bxor(data:byte(i), key))
    end
    return table.concat(result)
end
'''


class XORRollingCipher(BaseCipher):
    """Rolling XOR cipher with key feedback"""
    
    def encrypt(self, data: bytes, key: bytes) -> bytes:
        result = bytearray(len(data))
        key_len = len(key)
        prev = key[0]
        
        for i, b in enumerate(data):
            key_byte = key[i % key_len]
            encrypted = (b ^ key_byte ^ prev) & 0xFF
            result[i] = encrypted
            prev = encrypted
        
        return bytes(result)
    
    def decrypt(self, data: bytes, key: bytes) -> bytes:
        result = bytearray(len(data))
        key_len = len(key)
        prev = key[0]
        
        for i, b in enumerate(data):
            key_byte = key[i % key_len]
            decrypted = (b ^ key_byte ^ prev) & 0xFF
            result[i] = decrypted
            prev = b  # Use encrypted byte for next
        
        return bytes(result)
    
    def generate_lua_decryptor(self, key: bytes) -> str:
        key_table = ",".join(str(b) for b in key[:32])
        return f'''
local function xor_rolling_decrypt(data)
    local key = {{{key_table}}}
    local key_len = #key
    local result = {{}}
    local prev = key[1]
    
    for i = 1, #data do
        local b = data:byte(i)
        local key_byte = key[((i - 1) % key_len) + 1]
        local decrypted = bit32.bxor(bit32.bxor(b, key_byte), prev)
        result[i] = string.char(decrypted % 256)
        prev = b
    end
    
    return table.concat(result)
end
'''


class XORMultibyteCipher(BaseCipher):
    """Multi-byte XOR with position-dependent transformation"""
    
    def __init__(self):
        self.transforms = [
            lambda b, k, i: (b ^ k ^ (i & 0xFF)) & 0xFF,
            lambda b, k, i: (b ^ k ^ ((i * 7) & 0xFF)) & 0xFF,
            lambda b, k, i: (b ^ k ^ ((i >> 2) & 0xFF)) & 0xFF,
        ]
    
    def encrypt(self, data: bytes, key: bytes) -> bytes:
        result = bytearray(len(data))
        key_len = len(key)
        
        for i, b in enumerate(data):
            key_byte = key[i % key_len]
            transform = self.transforms[i % len(self.transforms)]
            result[i] = transform(b, key_byte, i)
        
        return bytes(result)
    
    def decrypt(self, data: bytes, key: bytes) -> bytes:
        # For XOR, encryption and decryption are the same
        return self.encrypt(data, key)
    
    def generate_lua_decryptor(self, key: bytes) -> str:
        key_table = ",".join(str(b) for b in key[:32])
        return f'''
local function xor_multibyte_decrypt(data)
    local key = {{{key_table}}}
    local key_len = #key
    local result = {{}}
    
    for i = 1, #data do
        local b = data:byte(i)
        local key_byte = key[((i - 1) % key_len) + 1]
        local idx = i - 1
        local mode = idx % 3
        local decrypted
        
        if mode == 0 then
            decrypted = bit32.bxor(bit32.bxor(b, key_byte), bit32.band(idx, 255))
        elseif mode == 1 then
            decrypted = bit32.bxor(bit32.bxor(b, key_byte), bit32.band(idx * 7, 255))
        else
            decrypted = bit32.bxor(bit32.bxor(b, key_byte), bit32.band(bit32.rshift(idx, 2), 255))
        end
        
        result[i] = string.char(decrypted % 256)
    end
    
    return table.concat(result)
end
'''

# ============================================
# RC4 Cipher
# ============================================

class RC4Cipher(BaseCipher):
    """RC4 stream cipher implementation"""
    
    def _rc4_init(self, key: bytes) -> List[int]:
        """Initialize RC4 S-box"""
        S = list(range(256))
        j = 0
        
        for i in range(256):
            j = (j + S[i] + key[i % len(key)]) % 256
            S[i], S[j] = S[j], S[i]
        
        return S
    
    def _rc4_stream(self, S: List[int], length: int) -> bytes:
        """Generate RC4 keystream"""
        S = S.copy()
        i = j = 0
        result = bytearray(length)
        
        for k in range(length):
            i = (i + 1) % 256
            j = (j + S[i]) % 256
            S[i], S[j] = S[j], S[i]
            result[k] = S[(S[i] + S[j]) % 256]
        
        return bytes(result)
    
    def encrypt(self, data: bytes, key: bytes) -> bytes:
        S = self._rc4_init(key)
        keystream = self._rc4_stream(S, len(data))
        return bytes(d ^ k for d, k in zip(data, keystream))
    
    def decrypt(self, data: bytes, key: bytes) -> bytes:
        return self.encrypt(data, key)  # RC4 is symmetric
    
    def generate_lua_decryptor(self, key: bytes) -> str:
        key_table = ",".join(str(b) for b in key[:32])
        return f'''
local function rc4_decrypt(data)
    local key = {{{key_table}}}
    
    -- Initialize S-box
    local S = {{}}
    for i = 0, 255 do S[i] = i end
    
    local j = 0
    for i = 0, 255 do
        j = (j + S[i] + key[(i % #key) + 1]) % 256
        S[i], S[j] = S[j], S[i]
    end
    
    -- Generate keystream and decrypt
    local result = {{}}
    local i, j = 0, 0
    
    for k = 1, #data do
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        local key_byte = S[(S[i] + S[j]) % 256]
        result[k] = string.char(bit32.bxor(data:byte(k), key_byte))
    end
    
    return table.concat(result)
end
'''

# ============================================
# Custom Feistel Cipher
# ============================================

class FeistelCipher(BaseCipher):
    """Custom Feistel network cipher"""
    
    def __init__(self, rounds: int = 16):
        self.rounds = rounds
    
    def _round_function(self, half: int, key: int, round_num: int) -> int:
        """Feistel round function"""
        # Mix with key
        mixed = half ^ key
        # Substitution (simple S-box simulation)
        mixed = ((mixed * 0x6789) ^ (mixed >> 4)) & 0xFFFFFFFF
        # Add round constant
        mixed = (mixed + round_num * 0x9E3779B9) & 0xFFFFFFFF
        return mixed
    
    def _derive_round_keys(self, key: bytes) -> List[int]:
        """Derive round keys from master key"""
        keys = []
        key_int = int.from_bytes(key[:16], 'little')
        
        for i in range(self.rounds):
            round_key = (key_int ^ (i * 0xDEADBEEF)) & 0xFFFFFFFF
            keys.append(round_key)
            key_int = ((key_int << 3) | (key_int >> 29)) & 0xFFFFFFFF
        
        return keys
    
    def encrypt(self, data: bytes, key: bytes) -> bytes:
        # Pad to multiple of 8 bytes
        padded = data + bytes(8 - (len(data) % 8) if len(data) % 8 else 0)
        result = bytearray()
        
        round_keys = self._derive_round_keys(key)
        
        for i in range(0, len(padded), 8):
            block = padded[i:i+8]
            left = int.from_bytes(block[:4], 'little')
            right = int.from_bytes(block[4:], 'little')
            
            # Feistel rounds
            for r in range(self.rounds):
                new_right = left ^ self._round_function(right, round_keys[r], r)
                left = right
                right = new_right & 0xFFFFFFFF
            
            result.extend(right.to_bytes(4, 'little'))
            result.extend(left.to_bytes(4, 'little'))
        
        return bytes(result)
    
    def decrypt(self, data: bytes, key: bytes) -> bytes:
        if len(data) % 8 != 0:
            raise ValueError("Data length must be multiple of 8")
        
        result = bytearray()
        round_keys = self._derive_round_keys(key)
        
        for i in range(0, len(data), 8):
            block = data[i:i+8]
            right = int.from_bytes(block[:4], 'little')
            left = int.from_bytes(block[4:], 'little')
            
            # Reverse Feistel rounds
            for r in range(self.rounds - 1, -1, -1):
                new_left = right ^ self._round_function(left, round_keys[r], r)
                right = left
                left = new_left & 0xFFFFFFFF
            
            result.extend(left.to_bytes(4, 'little'))
            result.extend(right.to_bytes(4, 'little'))
        
        return bytes(result).rstrip(b'\x00')
    
    def generate_lua_decryptor(self, key: bytes) -> str:
        # Generate round keys
        round_keys = self._derive_round_keys(key)
        keys_str = ",".join(str(k) for k in round_keys)
        
        return f'''
local function feistel_decrypt(data)
    local round_keys = {{{keys_str}}}
    local rounds = {self.rounds}
    
    local function round_func(half, key, round_num)
        local mixed = bit32.bxor(half, key)
        mixed = bit32.bxor(mixed * 0x6789, bit32.rshift(mixed, 4))
        mixed = bit32.band(mixed + round_num * 0x9E3779B9, 0xFFFFFFFF)
        return mixed
    end
    
    local result = {{}}
    
    for i = 1, #data, 8 do
        local block = data:sub(i, i + 7)
        if #block < 8 then break end
        
        -- Extract left and right (little endian)
        local right = block:byte(1) + block:byte(2) * 256 + 
                     block:byte(3) * 65536 + block:byte(4) * 16777216
        local left = block:byte(5) + block:byte(6) * 256 + 
                    block:byte(7) * 65536 + block:byte(8) * 16777216
        
        -- Reverse rounds
        for r = rounds, 1, -1 do
            local new_left = bit32.bxor(right, round_func(left, round_keys[r], r - 1))
            right = left
            left = bit32.band(new_left, 0xFFFFFFFF)
        end
        
        -- Output as bytes
        result[#result + 1] = string.char(
            bit32.band(left, 255),
            bit32.band(bit32.rshift(left, 8), 255),
            bit32.band(bit32.rshift(left, 16), 255),
            bit32.band(bit32.rshift(left, 24), 255),
            bit32.band(right, 255),
            bit32.band(bit32.rshift(right, 8), 255),
            bit32.band(bit32.rshift(right, 16), 255),
            bit32.band(bit32.rshift(right, 24), 255)
        )
    end
    
    return table.concat(result):gsub("%z+$", "")
end
'''

# ============================================
# Custom Substitution Cipher
# ============================================

class SubstitutionCipher(BaseCipher):
    """Key-dependent substitution cipher"""
    
    def _generate_sbox(self, key: bytes) -> Tuple[List[int], List[int]]:
        """Generate S-box and inverse from key"""
        # Initialize with identity
        sbox = list(range(256))
        
        # Fisher-Yates shuffle based on key
        random.seed(int.from_bytes(key[:8], 'little'))
        for i in range(255, 0, -1):
            j = random.randint(0, i)
            sbox[i], sbox[j] = sbox[j], sbox[i]
        
        # Generate inverse
        inv_sbox = [0] * 256
        for i, v in enumerate(sbox):
            inv_sbox[v] = i
        
        random.seed()  # Reset random state
        return sbox, inv_sbox
    
    def encrypt(self, data: bytes, key: bytes) -> bytes:
        sbox, _ = self._generate_sbox(key)
        return bytes(sbox[b] for b in data)
    
    def decrypt(self, data: bytes, key: bytes) -> bytes:
        _, inv_sbox = self._generate_sbox(key)
        return bytes(inv_sbox[b] for b in data)
    
    def generate_lua_decryptor(self, key: bytes) -> str:
        _, inv_sbox = self._generate_sbox(key)
        sbox_str = ",".join(str(b) for b in inv_sbox)
        
        return f'''
local function substitution_decrypt(data)
    local inv_sbox = {{{sbox_str}}}
    local result = {{}}
    
    for i = 1, #data do
        result[i] = string.char(inv_sbox[data:byte(i) + 1])
    end
    
    return table.concat(result)
end
'''

# ============================================
# Cipher Factory
# ============================================

class CipherFactory:
    """Factory for creating cipher instances"""
    
    _ciphers: Dict[EncryptionAlgorithm, type] = {
        EncryptionAlgorithm.XOR_SIMPLE: XORSimpleCipher,
        EncryptionAlgorithm.XOR_ROLLING: XORRollingCipher,
        EncryptionAlgorithm.XOR_MULTIBYTE: XORMultibyteCipher,
        EncryptionAlgorithm.RC4: RC4Cipher,
        EncryptionAlgorithm.CUSTOM_FEISTEL: FeistelCipher,
        EncryptionAlgorithm.CUSTOM_SUBSTITUTION: SubstitutionCipher,
    }
    
    @classmethod
    def create(cls, algorithm: EncryptionAlgorithm) -> BaseCipher:
        """Create cipher instance"""
        cipher_class = cls._ciphers.get(algorithm)
        if cipher_class is None:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        return cipher_class()
    
    @classmethod
    def get_all_algorithms(cls) -> List[EncryptionAlgorithm]:
        """Get list of all supported algorithms"""
        return list(cls._ciphers.keys())

# ============================================
# Layered Encryption
# ============================================

class LayeredEncryption:
    """Multi-layer encryption with different algorithms"""
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.layers: List[Tuple[BaseCipher, bytes]] = []
        
        # Initialize layers
        for i in range(config.num_layers):
            algo = config.layer_algorithms[i] if i < len(config.layer_algorithms) \
                   else EncryptionAlgorithm.XOR_ROLLING
            key = config.layer_keys[i] if i < len(config.layer_keys) \
                  else secrets.token_bytes(config.key_size)
            
            cipher = CipherFactory.create(algo)
            self.layers.append((cipher, key))
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt through all layers"""
        result = data
        
        for cipher, key in self.layers:
            result = cipher.encrypt(result, key)
        
        return result
    
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt through all layers (reverse order)"""
        result = data
        
        for cipher, key in reversed(self.layers):
            result = cipher.decrypt(result, key)
        
        return result
    
    def generate_lua_decryptor(self) -> str:
        """Generate combined Lua decryptor for all layers"""
        decryptors = []
        layer_calls = []
        
        for i, (cipher, key) in enumerate(reversed(self.layers)):
            func_name = f"_layer{i}_decrypt"
            decryptor_code = cipher.generate_lua_decryptor(key)
            # Rename the function
            decryptor_code = decryptor_code.replace(
                "local function", 
                f"local function {func_name}__TEMP"
            ).replace("__TEMP", "")
            decryptors.append(decryptor_code)
            layer_calls.append(func_name)
        
        # Generate combined decryptor
        combined = "\n".join(decryptors)
        
        # Chain the calls
        chain = "data"
        for call in layer_calls:
            # Extract function name
            func_name = call
            chain = f"{func_name}({chain})"
        
        combined += f'''
local function layered_decrypt(data)
    return {chain}
end
'''
        return combined

# ============================================
# String Encryptor
# ============================================

class StringEncryptor:
    """Specialized string encryption with various techniques"""
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.cipher = CipherFactory.create(config.string_algorithm)
        self.encrypted_strings: Dict[str, Tuple[bytes, bytes]] = {}  # original -> (encrypted, key)
    
    def encrypt_string(self, s: str) -> Tuple[bytes, bytes, int]:
        """
        Encrypt a string
        Returns: (encrypted_data, key, method_id)
        """
        data = s.encode('utf-8')
        
        # Generate per-string key if configured
        if self.config.per_string_keys:
            key = secrets.token_bytes(min(len(data) + 8, 32))
        else:
            key = self.config.master_key[:32]
        
        # Add random prefix/suffix for same strings to produce different output
        salt = secrets.token_bytes(4)
        salted_data = salt + struct.pack('<H', len(data)) + data
        
        # Encrypt
        encrypted = self.cipher.encrypt(salted_data, key)
        
        # Store for reference
        self.encrypted_strings[s] = (encrypted, key)
        
        return encrypted, key, self.config.string_algorithm.value
    
    def decrypt_string(self, encrypted: bytes, key: bytes) -> str:
        """Decrypt a string"""
        decrypted = self.cipher.decrypt(encrypted, key)
        
        # Remove salt and length prefix
        if len(decrypted) < 6:
            return ""
        
        length = struct.unpack('<H', decrypted[4:6])[0]
        return decrypted[6:6+length].decode('utf-8', errors='replace')
    
    def generate_lua_decryptor(self) -> str:
        """Generate Lua string decryptor"""
        base_decryptor = self.cipher.generate_lua_decryptor(self.config.master_key)
        
        return base_decryptor + '''
local function decrypt_string_wrapper(encrypted_b64, key_b64)
    -- Base64 decode
    local b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    
    local function b64_decode(input)
        local output = {}
        local val, bits = 0, 0
        for i = 1, #input do
            local c = input:sub(i, i)
            if c ~= "=" then
                local idx = b64:find(c, 1, true)
                if idx then
                    val = val * 64 + (idx - 1)
                    bits = bits + 6
                    if bits >= 8 then
                        bits = bits - 8
                        output[#output + 1] = string.char(math.floor(val / (2 ^ bits)) % 256)
                        val = val % (2 ^ bits)
                    end
                end
            end
        end
        return table.concat(output)
    end
    
    local encrypted = b64_decode(encrypted_b64)
    local key_bytes = {}
    local key_str = b64_decode(key_b64)
    for i = 1, #key_str do
        key_bytes[i] = key_str:byte(i)
    end
    
    -- Decrypt and extract string
    local decrypted = xor_rolling_decrypt(encrypted)  -- Or appropriate cipher
    if #decrypted < 6 then return "" end
    
    local length = decrypted:byte(5) + decrypted:byte(6) * 256
    return decrypted:sub(7, 6 + length)
end
'''

# ============================================
# Number Encryptor
# ============================================

class NumberEncryptor:
    """Specialized number encryption"""
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.xor_key = int.from_bytes(config.master_key[:8], 'little')
        self.mul_key = int.from_bytes(config.master_key[8:16], 'little') | 1  # Ensure odd
    
    def encrypt_integer(self, n: int) -> Tuple[int, int, int]:
        """
        Encrypt integer
        Returns: (encrypted_low, encrypted_high, method)
        """
        # XOR with key
        encrypted = n ^ self.xor_key
        
        # Split into parts
        low = encrypted & 0xFFFFFFFF
        high = (encrypted >> 32) & 0xFFFFFFFF
        
        return low, high, 1
    
    def decrypt_integer(self, low: int, high: int) -> int:
        """Decrypt integer"""
        encrypted = low | (high << 32)
        return encrypted ^ self.xor_key
    
    def encrypt_float(self, f: float) -> Tuple[int, int, int, int]:
        """
        Encrypt float
        Returns: (p1, p2, p3, method)
        """
        # Pack as IEEE 754 double
        packed = struct.pack('<d', f)
        value = struct.unpack('<Q', packed)[0]
        
        # XOR
        encrypted = value ^ self.xor_key
        
        # Split into three parts for more obfuscation
        p1 = encrypted & 0x1FFFFF
        p2 = (encrypted >> 21) & 0x1FFFFF
        p3 = (encrypted >> 42) & 0x3FFFFF
        
        return p1, p2, p3, 2
    
    def decrypt_float(self, p1: int, p2: int, p3: int) -> float:
        """Decrypt float"""
        encrypted = p1 | (p2 << 21) | (p3 << 42)
        value = encrypted ^ self.xor_key
        packed = struct.pack('<Q', value)
        return struct.unpack('<d', packed)[0]
    
    def generate_lua_decryptor(self) -> str:
        """Generate Lua number decryptor"""
        return f'''
local _NUM_XOR_KEY = {self.xor_key}

local function decrypt_integer(low, high)
    local encrypted = low + high * 4294967296
    return bit32.bxor(encrypted, _NUM_XOR_KEY)
end

local function decrypt_float(p1, p2, p3)
    local encrypted = p1 + p2 * 2097152 + p3 * 4398046511104
    local value = bit32.bxor(encrypted, _NUM_XOR_KEY)
    -- IEEE 754 conversion (simplified - may need full implementation)
    -- For now, store as encoded string and use loadstring
    return value  -- Placeholder
end
'''

# ============================================
# Bytecode Encryptor
# ============================================

class BytecodeEncryptor:
    """Encrypts entire bytecode blocks"""
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.cipher = CipherFactory.create(config.bytecode_algorithm)
        
        if config.use_layered_encryption:
            self.layered = LayeredEncryption(config)
        else:
            self.layered = None
    
    def encrypt(self, bytecode: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Encrypt bytecode
        Returns: (encrypted_data, metadata)
        """
        metadata = {
            'original_size': len(bytecode),
            'compressed': False,
            'layered': self.layered is not None,
            'algorithm': self.config.bytecode_algorithm.value,
        }
        
        data = bytecode
        
        # Compress if configured
        if self.config.compress_before_encrypt:
            compressed = zlib.compress(data, self.config.compression_level)
            if len(compressed) < len(data):
                data = compressed
                metadata['compressed'] = True
                metadata['compressed_size'] = len(compressed)
        
        # Add integrity check
        if self.config.add_integrity_check:
            if self.config.integrity_algorithm == HashAlgorithm.SHA256:
                checksum = hashlib.sha256(data).digest()
            elif self.config.integrity_algorithm == HashAlgorithm.MD5:
                checksum = hashlib.md5(data).digest()
            else:
                checksum = hashlib.sha256(data).digest()
            
            data = checksum + data
            metadata['has_checksum'] = True
            metadata['checksum_size'] = len(checksum)
        
        # Add junk bytes if configured
        if self.config.add_junk_bytes:
            junk_size = int(len(data) * self.config.junk_byte_ratio)
            junk = secrets.token_bytes(junk_size)
            # Interleave junk at specific positions
            data = self._add_junk(data, junk)
            metadata['junk_size'] = junk_size
        
        # Encrypt
        if self.layered:
            encrypted = self.layered.encrypt(data)
        else:
            encrypted = self.cipher.encrypt(data, self.config.master_key)
        
        metadata['encrypted_size'] = len(encrypted)
        
        return encrypted, metadata
    
    def decrypt(self, encrypted: bytes, metadata: Dict[str, Any]) -> bytes:
        """Decrypt bytecode"""
        # Decrypt
        if self.layered:
            data = self.layered.decrypt(encrypted)
        else:
            data = self.cipher.decrypt(encrypted, self.config.master_key)
        
        # Remove junk bytes
        if metadata.get('junk_size'):
            data = self._remove_junk(data, metadata['junk_size'])
        
        # Verify and remove checksum
        if metadata.get('has_checksum'):
            checksum_size = metadata['checksum_size']
            stored_checksum = data[:checksum_size]
            data = data[checksum_size:]
            
            # Verify
            if self.config.integrity_algorithm == HashAlgorithm.SHA256:
                computed = hashlib.sha256(data).digest()
            else:
                computed = hashlib.md5(data).digest()
            
            if stored_checksum != computed:
                raise ValueError("Integrity check failed!")
        
        # Decompress
        if metadata.get('compressed'):
            data = zlib.decompress(data)
        
        return data
    
    def _add_junk(self, data: bytes, junk: bytes) -> bytes:
        """Add junk bytes at calculated positions"""
        result = bytearray()
        junk_idx = 0
        
        for i, b in enumerate(data):
            result.append(b)
            # Add junk every N bytes
            if i % 10 == 9 and junk_idx < len(junk):
                result.append(junk[junk_idx])
                junk_idx += 1
        
        # Append remaining junk
        result.extend(junk[junk_idx:])
        
        return bytes(result)
    
    def _remove_junk(self, data: bytes, junk_size: int) -> bytes:
        """Remove junk bytes"""
        # Calculate expected original size
        # For every 10 bytes, 1 junk was added
        result = bytearray()
        junk_remaining = junk_size
        
        i = 0
        count = 0
        while i < len(data) and junk_remaining >= 0:
            result.append(data[i])
            count += 1
            i += 1
            
            # Skip junk byte every 10 original bytes
            if count % 10 == 0 and junk_remaining > 0 and i < len(data):
                i += 1  # Skip junk byte
                junk_remaining -= 1
        
        return bytes(result)
    
    def generate_lua_decryptor(self, metadata: Dict[str, Any]) -> str:
        """Generate Lua bytecode decryptor"""
        parts = []
        
        # Base cipher decryptor
        if self.layered:
            parts.append(self.layered.generate_lua_decryptor())
        else:
            parts.append(self.cipher.generate_lua_decryptor(self.config.master_key))
        
        # Junk removal
        if metadata.get('junk_size'):
            parts.append(f'''
local function remove_junk(data, junk_size)
    local result = {{}}
    local junk_remaining = {metadata['junk_size']}
    local i = 1
    local count = 0
    
    while i <= #data and junk_remaining >= 0 do
        result[#result + 1] = data:sub(i, i)
        count = count + 1
        i = i + 1
        
        if count % 10 == 0 and junk_remaining > 0 and i <= #data then
            i = i + 1
            junk_remaining = junk_remaining - 1
        end
    end
    
    return table.concat(result)
end
''')
        
        # Checksum verification
        if metadata.get('has_checksum'):
            checksum_size = metadata['checksum_size']
            parts.append(f'''
local function verify_checksum(data)
    local checksum = data:sub(1, {checksum_size})
    local payload = data:sub({checksum_size + 1})
    -- Simplified verification
    return payload, true
end
''')
        
        # Decompression (if using lzw or built-in)
        if metadata.get('compressed'):
            parts.append('''
-- Note: zlib decompression requires external library or custom implementation
local function decompress(data)
    -- Placeholder - actual implementation depends on target environment
    return data
end
''')
        
        return "\n".join(parts)

# ============================================
# Main Encryption Manager
# ============================================

class EncryptionManager:
    """Main manager for all encryption operations"""
    
    def __init__(self, config: EncryptionConfig = None):
        self.config = config or EncryptionConfig()
        self.config.generate_keys()
        
        # Initialize encryptors
        self.string_encryptor = StringEncryptor(self.config)
        self.number_encryptor = NumberEncryptor(self.config)
        self.bytecode_encryptor = BytecodeEncryptor(self.config)
    
    def encrypt_string(self, s: str) -> Dict[str, Any]:
        """Encrypt a string and return serializable result"""
        encrypted, key, method = self.string_encryptor.encrypt_string(s)
        
        return {
            'data': base64.b64encode(encrypted).decode('ascii'),
            'key': base64.b64encode(key).decode('ascii'),
            'method': method,
            'type': 'string'
        }
    
    def decrypt_string(self, encrypted_data: Dict[str, Any]) -> str:
        """Decrypt a string from serialized form"""
        encrypted = base64.b64decode(encrypted_data['data'])
        key = base64.b64decode(encrypted_data['key'])
        return self.string_encryptor.decrypt_string(encrypted, key)
    
    def encrypt_number(self, n: Union[int, float]) -> Dict[str, Any]:
        """Encrypt a number"""
        if isinstance(n, float):
            p1, p2, p3, method = self.number_encryptor.encrypt_float(n)
            return {
                'parts': [p1, p2, p3],
                'method': method,
                'type': 'float'
            }
        else:
            low, high, method = self.number_encryptor.encrypt_integer(n)
            return {
                'parts': [low, high],
                'method': method,
                'type': 'integer'
            }
    
    def decrypt_number(self, encrypted_data: Dict[str, Any]) -> Union[int, float]:
        """Decrypt a number"""
        parts = encrypted_data['parts']
        if encrypted_data['type'] == 'float':
            return self.number_encryptor.decrypt_float(parts[0], parts[1], parts[2])
        else:
            return self.number_encryptor.decrypt_integer(parts[0], parts[1])
    
    def encrypt_bytecode(self, bytecode: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Encrypt bytecode"""
        return self.bytecode_encryptor.encrypt(bytecode)
    
    def decrypt_bytecode(self, encrypted: bytes, metadata: Dict[str, Any]) -> bytes:
        """Decrypt bytecode"""
        return self.bytecode_encryptor.decrypt(encrypted, metadata)
    
    def generate_lua_runtime(self) -> str:
        """Generate complete Lua decryption runtime"""
        parts = [
            "-- Encryption Runtime",
            "-- Generated decryption functions",
            "",
            self.string_encryptor.generate_lua_decryptor(),
            "",
            self.number_encryptor.generate_lua_decryptor(),
            "",
        ]
        
        return "\n".join(parts)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of encryption configuration"""
        return {
            'string_algorithm': self.config.string_algorithm.name,
            'number_algorithm': self.config.number_algorithm.name,
            'bytecode_algorithm': self.config.bytecode_algorithm.name,
            'layered': self.config.use_layered_encryption,
            'num_layers': self.config.num_layers if self.config.use_layered_encryption else 0,
            'compression': self.config.compress_before_encrypt,
            'integrity': self.config.add_integrity_check,
            'key_size': self.config.key_size,
        }

# ============================================
# Utility Functions
# ============================================

def create_encryption_manager(
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.XOR_ROLLING,
    use_layers: bool = True,
    num_layers: int = 3,
    compress: bool = True
) -> EncryptionManager:
    """Create encryption manager with common settings"""
    config = EncryptionConfig(
        algorithm=algorithm,
        use_layered_encryption=use_layers,
        num_layers=num_layers,
        compress_before_encrypt=compress,
    )
    return EncryptionManager(config)


def quick_encrypt_string(s: str, key: bytes = None) -> Tuple[bytes, bytes]:
    """Quick string encryption"""
    config = EncryptionConfig()
    if key:
        config.master_key = key
    config.generate_keys()
    
    encryptor = StringEncryptor(config)
    encrypted, key, _ = encryptor.encrypt_string(s)
    return encrypted, key


def quick_decrypt_string(encrypted: bytes, key: bytes) -> str:
    """Quick string decryption"""
    config = EncryptionConfig()
    config.master_key = key
    
    encryptor = StringEncryptor(config)
    return encryptor.decrypt_string(encrypted, key)

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    print("=== Lua Encryption Layer ===\n")
    
    # Test configuration
    config = EncryptionConfig(
        algorithm=EncryptionAlgorithm.RC4,
        use_layered_encryption=True,
        num_layers=3,
        encrypt_strings=True,
        encrypt_numbers=True,
        compress_before_encrypt=True,
        add_integrity_check=True,
    )
    config.generate_keys()
    
    print("Configuration:")
    print(f"  Algorithm: {config.algorithm.name}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Key Size: {config.key_size}")
    print()
    
    # Create manager
    manager = EncryptionManager(config)
    
    # Test string encryption
    print("=== String Encryption Test ===")
    test_strings = [
        "Hello, World!",
        "print('test')",
        "local x = 42",
        "こんにちは",  # Unicode
    ]
    
    for s in test_strings:
        encrypted = manager.encrypt_string(s)
        decrypted = manager.decrypt_string(encrypted)
        match = "✓" if s == decrypted else "✗"
        print(f"  {match} '{s[:20]}...' -> {len(encrypted['data'])} chars")
    print()
    
    # Test number encryption
    print("=== Number Encryption Test ===")
    test_numbers = [42, -17, 3.14159, 0, 1e10]
    
    for n in test_numbers:
        encrypted = manager.encrypt_number(n)
        decrypted = manager.decrypt_number(encrypted)
        if isinstance(n, float):
            match = "✓" if abs(n - decrypted) < 0.0001 else "✗"
        else:
            match = "✓" if n == decrypted else "✗"
        print(f"  {match} {n} -> {encrypted['parts']}")
    print()
    
    # Test bytecode encryption
    print("=== Bytecode Encryption Test ===")
    test_bytecode = b'\x1bLua\x51\x00\x01\x04\x04\x04\x08\x00' + bytes(range(256))
    
    encrypted_bc, metadata = manager.encrypt_bytecode(test_bytecode)
    decrypted_bc = manager.decrypt_bytecode(encrypted_bc, metadata)
    
    print(f"  Original size: {len(test_bytecode)}")
    print(f"  Encrypted size: {len(encrypted_bc)}")
    print(f"  Compressed: {metadata.get('compressed', False)}")
    print(f"  Has checksum: {metadata.get('has_checksum', False)}")
    print(f"  Match: {'✓' if test_bytecode == decrypted_bc else '✗'}")
    print()
    
    # Generate Lua runtime
    print("=== Lua Runtime Generation ===")
    runtime = manager.generate_lua_runtime()
    print(f"  Runtime size: {len(runtime)} characters")
    print(f"  Functions defined: {runtime.count('local function')}")
    print()
    
    # Configuration summary
    print("=== Configuration Summary ===")
    summary = manager.get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Encryption layer tests completed!")
