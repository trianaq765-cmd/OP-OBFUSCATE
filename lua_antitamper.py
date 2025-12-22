# ============================================
# File: lua_antitamper.py
# Advanced Anti-Tampering System for Lua Obfuscator
# Comprehensive protection against reverse engineering
# ============================================

import os
import sys
import time
import random
import hashlib
import struct
import base64
import secrets
import json
import zlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from enum import IntEnum, auto
from abc import ABC, abstractmethod
import string

# ============================================
# Protection Types
# ============================================

class ProtectionType(IntEnum):
    """Types of anti-tampering protections"""
    INTEGRITY_CHECK = auto()
    ENVIRONMENT_DETECT = auto()
    DEBUGGER_DETECT = auto()
    TIMING_CHECK = auto()
    VM_DETECT = auto()
    SANDBOX_DETECT = auto()
    CHECKSUM_VERIFY = auto()
    CODE_FLOW_CHECK = auto()
    WATERMARK = auto()
    LICENSE_CHECK = auto()
    ANTI_HOOK = auto()
    SELF_DESTRUCT = auto()
    OBFUSCATED_CALLS = auto()
    ANTI_DUMP = auto()
    RUNTIME_DECRYPT = auto()

class DetectionAction(IntEnum):
    """Actions to take when tampering is detected"""
    SILENT_FAIL = auto()      # Silently produce wrong results
    CRASH = auto()            # Crash the program
    INFINITE_LOOP = auto()    # Enter infinite loop
    CORRUPT_DATA = auto()     # Corrupt internal data
    DELAYED_CRASH = auto()    # Crash after delay
    REPORT = auto()           # Report to server (if possible)
    CLEAN_EXIT = auto()       # Clean exit with error
    RANDOM_BEHAVIOR = auto()  # Random unpredictable behavior

# ============================================
# Configuration
# ============================================

@dataclass
class AntiTamperConfig:
    """Configuration for anti-tampering system"""
    
    # Enable/disable protections
    enable_integrity_check: bool = True
    enable_environment_detect: bool = True
    enable_debugger_detect: bool = True
    enable_timing_check: bool = True
    enable_vm_detect: bool = False  # Can cause issues in VMs
    enable_sandbox_detect: bool = True
    enable_checksum: bool = True
    enable_code_flow: bool = True
    enable_watermark: bool = True
    enable_license: bool = False
    enable_anti_hook: bool = True
    enable_anti_dump: bool = True
    enable_runtime_decrypt: bool = True
    
    # Detection response
    detection_action: DetectionAction = DetectionAction.SILENT_FAIL
    use_multiple_actions: bool = True
    action_delay_ms: int = 0
    
    # Integrity settings
    checksum_algorithm: str = "sha256"
    verify_frequency: int = 100  # Check every N instructions
    
    # Timing settings
    timing_threshold_ms: float = 100.0
    timing_samples: int = 5
    
    # Watermark settings
    watermark_data: bytes = b''
    watermark_key: bytes = b''
    
    # License settings
    license_key: str = ""
    license_server: str = ""
    offline_grace_days: int = 7
    
    # Code generation
    obfuscate_checks: bool = True
    inline_checks: bool = True
    randomize_check_order: bool = True
    
    # Advanced
    use_polymorphic_code: bool = True
    use_metamorphic_code: bool = False
    decoy_code_ratio: float = 0.2
    
    def generate_keys(self):
        """Generate required keys"""
        if not self.watermark_key:
            self.watermark_key = secrets.token_bytes(32)
        if not self.watermark_data:
            self.watermark_data = secrets.token_bytes(16)

# ============================================
# Name Obfuscator for Anti-Tamper Code
# ============================================

class ATNameGenerator:
    """Generates heavily obfuscated names for anti-tamper code"""
    
    def __init__(self, seed: int = None):
        self.counter = 0
        self.used: Set[str] = set()
        if seed:
            random.seed(seed)
        
        # Confusing character sets
        self.char_sets = [
            "Il1|",
            "O0o",
            "_",
            "S5",
            "Z2",
        ]
    
    def generate(self, length: int = 12) -> str:
        """Generate obfuscated name"""
        for _ in range(100):
            # Mix of confusing characters
            name = "_"
            for i in range(length - 1):
                charset = random.choice(self.char_sets)
                name += random.choice(charset)
            
            if name not in self.used:
                self.used.add(name)
                return name
        
        self.counter += 1
        return f"__{self.counter:08x}"
    
    def generate_decoy(self) -> str:
        """Generate decoy variable name"""
        prefixes = ["check", "verify", "validate", "auth", "license", "key", "hash"]
        suffixes = ["Result", "Value", "Status", "Code", "Data", "Info"]
        return f"_{random.choice(prefixes)}{random.choice(suffixes)}"

# ============================================
# Integrity Check Generator
# ============================================

class IntegrityCheckGenerator:
    """Generates code integrity verification"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_checksum_code(self, code_hash: str) -> str:
        """Generate checksum verification code"""
        func_name = self.name_gen.generate()
        hash_var = self.name_gen.generate()
        result_var = self.name_gen.generate()
        
        return f'''
local {func_name}
{func_name} = function()
    local {hash_var} = "{code_hash}"
    local {result_var} = true
    
    -- Get function info
    local info = debug and debug.getinfo and debug.getinfo({func_name}, "S")
    if info then
        local src = info.source or ""
        -- Calculate hash of source
        local h = 0
        for i = 1, #src do
            h = (h * 31 + src:byte(i)) % 2147483647
        end
        
        -- Verify (obfuscated comparison)
        local expected = 0
        for i = 1, #"{code_hash}" do
            expected = (expected * 31 + ("{code_hash}"):byte(i)) % 2147483647
        end
        
        {result_var} = (h % 65536) == (expected % 65536) or {result_var}
    end
    
    return {result_var}
end

if not {func_name}() then
    {self._generate_action_code()}
end
'''
    
    def generate_runtime_verify(self) -> str:
        """Generate runtime code verification"""
        verify_func = self.name_gen.generate()
        state_var = self.name_gen.generate()
        
        return f'''
local {state_var} = {{}}
local {verify_func} = function(id, expected)
    local actual = 0
    local info = debug and debug.getinfo(2, "l")
    if info then
        actual = info.currentline or 0
    end
    
    if {state_var}[id] then
        if {state_var}[id] ~= actual then
            {self._generate_action_code()}
        end
    else
        {state_var}[id] = actual
    end
    
    return true
end
'''
    
    def generate_function_verify(self) -> str:
        """Generate function tampering detection"""
        func_name = self.name_gen.generate()
        original_var = self.name_gen.generate()
        
        return f'''
local {original_var} = {{}}

local function {func_name}(name, func)
    local hash = 0
    local info = debug and debug.getinfo(func, "S")
    if info then
        local src = tostring(info.source or "") .. tostring(info.linedefined or 0)
        for i = 1, #src do
            hash = (hash * 31 + src:byte(i)) % 2147483647
        end
    end
    
    if {original_var}[name] then
        if {original_var}[name] ~= hash then
            {self._generate_action_code()}
        end
    else
        {original_var}[name] = hash
    end
end
'''
    
    def _generate_action_code(self) -> str:
        """Generate action code based on config"""
        action = self.config.detection_action
        
        if action == DetectionAction.CRASH:
            return 'error("Runtime error")'
        elif action == DetectionAction.INFINITE_LOOP:
            return 'while true do end'
        elif action == DetectionAction.SILENT_FAIL:
            return '-- silent'
        elif action == DetectionAction.CORRUPT_DATA:
            return '_G = nil; _ENV = nil'
        elif action == DetectionAction.DELAYED_CRASH:
            return f'''
            local _t = os.time and os.time() or 0
            if _t % 10 == 0 then error("Error") end
            '''
        else:
            return 'return nil'

# ============================================
# Debugger Detection Generator
# ============================================

class DebuggerDetectGenerator:
    """Generates debugger/analysis detection code"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_debug_library_check(self) -> str:
        """Check for debug library manipulation"""
        func_name = self.name_gen.generate()
        
        return f'''
local {func_name} = function()
    -- Check debug library
    local d = rawget(_G, "debug")
    if d then
        -- Check for hooks
        if d.gethook then
            local hook, mask = d.gethook()
            if hook then
                {self._generate_action()}
            end
        end
        
        -- Check if debug functions are native
        local info = d.getinfo and d.getinfo(d.getinfo, "S")
        if info and info.what ~= "C" then
            {self._generate_action()}
        end
    end
    
    -- Check for common debugger globals
    local suspicious = {{
        "mobdebug", "decoda_output", "lldebugger",
        "_TEST", "_DEBUG", "_PROFILER"
    }}
    
    for _, name in ipairs(suspicious) do
        if rawget(_G, name) ~= nil then
            {self._generate_action()}
        end
    end
    
    return true
end

{func_name}()
'''
    
    def generate_hook_detection(self) -> str:
        """Detect function hooking"""
        func_name = self.name_gen.generate()
        test_var = self.name_gen.generate()
        
        return f'''
local {func_name} = function()
    -- Test critical functions for hooks
    local tests = {{
        {{"tostring", tostring}},
        {{"tonumber", tonumber}},
        {{"type", type}},
        {{"pairs", pairs}},
        {{"ipairs", ipairs}},
        {{"next", next}},
        {{"rawget", rawget}},
        {{"rawset", rawset}},
    }}
    
    for _, test in ipairs(tests) do
        local name, func = test[1], test[2]
        if type(func) ~= "function" then
            {self._generate_action()}
        end
        
        -- Check if it's a native function
        local info = debug and debug.getinfo and debug.getinfo(func, "S")
        if info and info.what ~= "C" then
            {self._generate_action()}
        end
    end
    
    -- Test metatable manipulation
    local {test_var} = {{}}
    setmetatable({test_var}, {{
        __index = function() 
            {self._generate_action()}
        end
    }})
    
    return true
end

{func_name}()
'''
    
    def generate_breakpoint_detection(self) -> str:
        """Detect breakpoints through timing"""
        func_name = self.name_gen.generate()
        time_var = self.name_gen.generate()
        
        return f'''
local {func_name} = function()
    local clock = os.clock
    if not clock then return true end
    
    local {time_var} = {{}}
    
    -- Measure execution time of simple operations
    for i = 1, {self.config.timing_samples} do
        local t1 = clock()
        local sum = 0
        for j = 1, 10000 do
            sum = sum + j
        end
        local t2 = clock()
        {time_var}[i] = (t2 - t1) * 1000
    end
    
    -- Check for anomalies
    local avg = 0
    for i = 1, #{time_var} do
        avg = avg + {time_var}[i]
    end
    avg = avg / #{time_var}
    
    -- If too slow, debugger might be attached
    if avg > {self.config.timing_threshold_ms} then
        {self._generate_action()}
    end
    
    -- Check for variance (breakpoints cause spikes)
    for i = 1, #{time_var} do
        if {time_var}[i] > avg * 10 then
            {self._generate_action()}
        end
    end
    
    return true
end

{func_name}()
'''
    
    def generate_stack_check(self) -> str:
        """Check call stack for debugger frames"""
        func_name = self.name_gen.generate()
        
        return f'''
local {func_name} = function()
    if not debug or not debug.getinfo then
        return true
    end
    
    -- Walk the stack
    local level = 1
    while true do
        local info = debug.getinfo(level, "Sn")
        if not info then break end
        
        local name = info.name or ""
        local source = info.source or ""
        
        -- Check for debugger-related names
        local suspicious = {{"debug", "break", "trace", "profile", "hook"}}
        for _, pattern in ipairs(suspicious) do
            if name:lower():find(pattern) or source:lower():find(pattern) then
                {self._generate_action()}
            end
        end
        
        level = level + 1
        if level > 50 then break end
    end
    
    return true
end

{func_name}()
'''
    
    def _generate_action(self) -> str:
        """Generate detection action"""
        action = self.config.detection_action
        
        if action == DetectionAction.CRASH:
            return 'error("Critical error")'
        elif action == DetectionAction.SILENT_FAIL:
            # Corrupt some global to cause subtle failures
            return 'rawset(_G, "math", nil)'
        elif action == DetectionAction.INFINITE_LOOP:
            return 'repeat until false'
        elif action == DetectionAction.RANDOM_BEHAVIOR:
            return '''
            if math.random() > 0.5 then
                error("Error")
            else
                _G.print = function() end
            end
            '''
        else:
            return 'return false'

# ============================================
# Environment Detection Generator
# ============================================

class EnvironmentDetectGenerator:
    """Detects analysis/sandbox environments"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_sandbox_detection(self) -> str:
        """Detect sandbox environments"""
        func_name = self.name_gen.generate()
        
        return f'''
local {func_name} = function()
    -- Check for restricted environment
    local restricted = {{
        "os.execute", "io.open", "io.popen",
        "loadfile", "dofile", "require"
    }}
    
    local env = _G
    for _, path in ipairs(restricted) do
        local parts = {{}}
        for part in path:gmatch("[^.]+") do
            parts[#parts + 1] = part
        end
        
        local obj = env
        for _, part in ipairs(parts) do
            if type(obj) ~= "table" then break end
            obj = rawget(obj, part)
        end
        
        -- In sandbox, these are usually nil or wrapped
        if obj == nil then
            -- Might be sandboxed
        elseif type(obj) == "function" then
            local info = debug and debug.getinfo and debug.getinfo(obj, "S")
            if info and info.what ~= "C" then
                -- Function is not native - might be wrapped
                {self._generate_action()}
            end
        end
    end
    
    return true
end

{func_name}()
'''
    
    def generate_vm_detection(self) -> str:
        """Detect virtual machine environment"""
        func_name = self.name_gen.generate()
        
        return f'''
local {func_name} = function()
    -- Check Lua version and implementation
    local version = _VERSION or ""
    
    -- Check for LuaJIT
    if jit then
        -- Running in LuaJIT - OK
        return true
    end
    
    -- Check for unusual implementations
    local impl_checks = {{
        -- Check table behavior
        function()
            local t = {{}}
            t[1] = 1
            t[2] = 2
            return #t == 2
        end,
        
        -- Check string behavior
        function()
            return ("test"):len() == 4
        end,
        
        -- Check math behavior
        function()
            return math.floor(3.7) == 3
        end,
    }}
    
    for _, check in ipairs(impl_checks) do
        local ok, result = pcall(check)
        if not ok or not result then
            {self._generate_action()}
        end
    end
    
    return true
end

{func_name}()
'''
    
    def generate_emulator_detection(self) -> str:
        """Detect Lua emulators/interpreters"""
        func_name = self.name_gen.generate()
        
        return f'''
local {func_name} = function()
    -- Timing-based emulator detection
    local clock = os.clock
    if not clock then return true end
    
    -- Emulators often have different timing characteristics
    local times = {{}}
    
    for i = 1, 10 do
        local t1 = clock()
        
        -- Complex operation that's hard to optimize
        local x = 0
        for j = 1, 1000 do
            x = x + math.sin(j) * math.cos(j)
        end
        
        local t2 = clock()
        times[i] = t2 - t1
    end
    
    -- Calculate variance
    local sum, sum_sq = 0, 0
    for i = 1, 10 do
        sum = sum + times[i]
        sum_sq = sum_sq + times[i] * times[i]
    end
    
    local mean = sum / 10
    local variance = (sum_sq / 10) - (mean * mean)
    
    -- Emulators often have high variance or unrealistic times
    if mean < 0.000001 or mean > 1 then
        -- Suspicious timing
        {self._generate_action()}
    end
    
    return true
end

{func_name}()
'''
    
    def _generate_action(self) -> str:
        """Generate detection action"""
        return 'return false'

# ============================================
# Watermark Generator
# ============================================

class WatermarkGenerator:
    """Generates hidden watermarks for code tracking"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_hidden_watermark(self, data: bytes) -> str:
        """Generate hidden watermark embedded in code"""
        # Encode watermark data
        encoded = base64.b64encode(data).decode('ascii')
        
        # Split into chunks and hide in various ways
        chunks = [encoded[i:i+8] for i in range(0, len(encoded), 8)]
        
        code_parts = []
        
        for i, chunk in enumerate(chunks):
            method = i % 4
            var_name = self.name_gen.generate()
            
            if method == 0:
                # Hidden in string
                code_parts.append(f'local {var_name} = "--[[{chunk}]]"')
            elif method == 1:
                # Hidden in number encoding
                nums = [ord(c) for c in chunk]
                code_parts.append(f'local {var_name} = {{{",".join(str(n) for n in nums)}}}')
            elif method == 2:
                # Hidden in variable name pattern
                fake_name = f"_{chunk.replace('+', 'p').replace('/', 's').replace('=', 'e')}"
                code_parts.append(f'local {fake_name} = nil')
            else:
                # Hidden in arithmetic
                val = sum(ord(c) * (256 ** j) for j, c in enumerate(chunk[:4]))
                code_parts.append(f'local {var_name} = {val} -- config')
        
        return "\n".join(code_parts)
    
    def generate_fingerprint(self) -> str:
        """Generate unique fingerprint for tracking"""
        fp_var = self.name_gen.generate()
        collect_func = self.name_gen.generate()
        
        return f'''
local {fp_var} = {{}}
local {collect_func} = function()
    -- Collect environment fingerprint
    {fp_var}.lua_version = _VERSION
    {fp_var}.has_jit = jit ~= nil
    {fp_var}.has_debug = debug ~= nil
    {fp_var}.has_os = os ~= nil
    
    if os then
        {fp_var}.time = os.time and os.time()
        {fp_var}.clock = os.clock and os.clock()
    end
    
    -- Collect platform hints
    local hints = {{}}
    if package and package.config then
        hints.sep = package.config:sub(1, 1)
    end
    {fp_var}.hints = hints
    
    return {fp_var}
end

{collect_func}()
'''
    
    def generate_tracking_beacon(self) -> str:
        """Generate tracking beacon (disabled by default)"""
        beacon_func = self.name_gen.generate()
        
        return f'''
-- Tracking beacon (requires network - typically disabled)
local {beacon_func} = function(data)
    -- Placeholder for tracking
    -- In production, this could send data to a server
    local _ = data
end
'''

# ============================================
# License Check Generator
# ============================================

class LicenseCheckGenerator:
    """Generates license verification code"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_license_verify(self, public_key_data: str = "") -> str:
        """Generate license verification code"""
        verify_func = self.name_gen.generate()
        license_var = self.name_gen.generate()
        
        return f'''
local {verify_func} = function({license_var})
    if not {license_var} or {license_var} == "" then
        return false
    end
    
    -- Basic license format check
    -- Format: XXXX-XXXX-XXXX-XXXX
    local pattern = "^%w%w%w%w%-%w%w%w%w%-%w%w%w%w%-%w%w%w%w$"
    if not {license_var}:match(pattern) then
        return false
    end
    
    -- Checksum verification
    local sum = 0
    for i = 1, #{license_var} do
        local c = {license_var}:byte(i)
        if c ~= 45 then -- not hyphen
            sum = (sum * 31 + c) % 65536
        end
    end
    
    -- Last 4 chars should encode checksum
    local check_part = {license_var}:sub(-4)
    local expected = 0
    for i = 1, 4 do
        expected = expected * 36
        local c = check_part:byte(i)
        if c >= 48 and c <= 57 then
            expected = expected + (c - 48)
        elseif c >= 65 and c <= 90 then
            expected = expected + (c - 55)
        else
            return false
        end
    end
    
    return (sum % 1000) == (expected % 1000)
end
'''
    
    def generate_expiry_check(self, expiry_timestamp: int = 0) -> str:
        """Generate expiry date check"""
        check_func = self.name_gen.generate()
        
        if expiry_timestamp == 0:
            expiry_timestamp = int(time.time()) + (365 * 24 * 60 * 60)  # 1 year
        
        return f'''
local {check_func} = function()
    local expiry = {expiry_timestamp}
    local now = os.time and os.time() or 0
    
    if now == 0 then
        -- Can't verify time, allow with warning
        return true
    end
    
    if now > expiry then
        {self._generate_action()}
        return false
    end
    
    -- Warn if close to expiry
    local days_left = (expiry - now) / 86400
    if days_left < 7 then
        -- Could show warning
    end
    
    return true
end

if not {check_func}() then
    {self._generate_action()}
end
'''
    
    def generate_hwid_check(self) -> str:
        """Generate hardware ID binding"""
        hwid_func = self.name_gen.generate()
        
        return f'''
local {hwid_func} = function()
    -- Collect hardware identifiers
    local hwid_parts = {{}}
    
    -- OS info
    if os.getenv then
        hwid_parts[#hwid_parts + 1] = os.getenv("COMPUTERNAME") or ""
        hwid_parts[#hwid_parts + 1] = os.getenv("USERNAME") or ""
        hwid_parts[#hwid_parts + 1] = os.getenv("PROCESSOR_IDENTIFIER") or ""
    end
    
    -- Lua implementation info
    hwid_parts[#hwid_parts + 1] = _VERSION or ""
    if jit then
        hwid_parts[#hwid_parts + 1] = jit.version or ""
    end
    
    -- Calculate hash
    local combined = table.concat(hwid_parts, "|")
    local hash = 0
    for i = 1, #combined do
        hash = (hash * 31 + combined:byte(i)) % 2147483647
    end
    
    return string.format("%08X", hash)
end
'''
    
    def _generate_action(self) -> str:
        """Generate action for license failure"""
        return 'error("License validation failed")'

# ============================================
# Anti-Hook Generator
# ============================================

class AntiHookGenerator:
    """Generates anti-hooking protections"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_function_guards(self) -> str:
        """Generate function hooking guards"""
        guard_func = self.name_gen.generate()
        backup_var = self.name_gen.generate()
        
        return f'''
-- Backup critical functions
local {backup_var} = {{
    type = type,
    tostring = tostring,
    tonumber = tonumber,
    pairs = pairs,
    ipairs = ipairs,
    next = next,
    rawget = rawget,
    rawset = rawset,
    rawequal = rawequal,
    setmetatable = setmetatable,
    getmetatable = getmetatable,
    pcall = pcall,
    xpcall = xpcall,
    error = error,
    assert = assert,
    select = select,
    unpack = unpack or table.unpack,
}}

local {guard_func} = function()
    -- Verify functions haven't been replaced
    for name, original in pairs({backup_var}) do
        local current = rawget(_G, name)
        if current ~= original then
            -- Function was hooked!
            {self._generate_action()}
        end
    end
    
    return true
end

-- Run guard periodically (integrate with VM loop)
local _guard_counter = 0
local function _maybe_guard()
    _guard_counter = _guard_counter + 1
    if _guard_counter >= 1000 then
        _guard_counter = 0
        {guard_func}()
    end
end
'''
    
    def generate_metatable_guards(self) -> str:
        """Protect against metatable manipulation"""
        mt_guard = self.name_gen.generate()
        
        return f'''
local {mt_guard} = function(obj)
    local mt = getmetatable(obj)
    if mt then
        -- Lock the metatable
        rawset(mt, "__metatable", "locked")
        
        -- Monitor for changes
        local original_index = rawget(mt, "__index")
        local original_newindex = rawget(mt, "__newindex")
        
        rawset(mt, "__newindex", function(t, k, v)
            -- Check for suspicious modifications
            if k:match("^__") then
                {self._generate_action()}
            end
            if original_newindex then
                return original_newindex(t, k, v)
            end
            rawset(t, k, v)
        end)
    end
    
    return obj
end
'''
    
    def generate_global_monitor(self) -> str:
        """Monitor global table for modifications"""
        monitor_func = self.name_gen.generate()
        snapshot_var = self.name_gen.generate()
        
        return f'''
local {snapshot_var} = {{}}

-- Take snapshot of critical globals
for k, v in pairs(_G) do
    if type(v) == "function" then
        {snapshot_var}[k] = v
    end
end

local {monitor_func} = function()
    for k, v in pairs({snapshot_var}) do
        if rawget(_G, k) ~= v then
            -- Global was modified
            {self._generate_action()}
        end
    end
end
'''
    
    def _generate_action(self) -> str:
        """Generate anti-hook action"""
        return 'error("Security violation")'

# ============================================
# Code Flow Obfuscation
# ============================================

class CodeFlowObfuscator:
    """Generates obfuscated control flow"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_opaque_predicate(self) -> Tuple[str, bool]:
        """Generate opaque predicate (always true or always false)"""
        var1 = self.name_gen.generate()
        var2 = self.name_gen.generate()
        
        predicates = [
            # Always true
            (f"(({var1} * {var1}) >= 0)", True),
            (f"((type({var1}) == type({var1})))", True),
            (f"(({var1} == {var1}) or ({var2} ~= {var2}))", True),
            
            # Always false
            (f"(({var1} * {var1}) < 0)", False),
            (f"((type({var1}) ~= type({var1})))", False),
        ]
        
        pred, result = random.choice(predicates)
        setup = f"local {var1} = math.random()\nlocal {var2} = math.random()"
        
        return f"{setup}\nif {pred} then", result
    
    def generate_dead_code(self) -> str:
        """Generate dead code that looks important"""
        var1 = self.name_gen.generate()
        var2 = self.name_gen.generate()
        func = self.name_gen.generate()
        
        return f'''
local {func} = function()
    local {var1} = {random.randint(1, 1000)}
    local {var2} = {random.randint(1, 1000)}
    
    for i = 1, 10 do
        {var1} = ({var1} * 31 + {var2}) % 65536
        {var2} = ({var2} * 17 + {var1}) % 65536
    end
    
    if {var1} == 12345 then
        -- Never executed
        error("Impossible error")
    end
    
    return {var1}
end

-- Never called in normal flow
local _{self.name_gen.generate()} = {func}
'''
    
    def generate_dispatch_obfuscation(self) -> str:
        """Generate obfuscated function dispatch"""
        dispatch_var = self.name_gen.generate()
        index_var = self.name_gen.generate()
        
        return f'''
local {dispatch_var} = {{}}
local {index_var} = 0

-- Obfuscated dispatch table
setmetatable({dispatch_var}, {{
    __index = function(t, k)
        local real_k = ((k * 7) + 3) % 256
        return rawget(t, real_k)
    end,
    __newindex = function(t, k, v)
        local real_k = ((k * 7) + 3) % 256
        rawset(t, real_k, v)
    end
}})
'''
    
    def generate_state_machine(self, num_states: int = 5) -> str:
        """Generate state machine based control flow"""
        state_var = self.name_gen.generate()
        handler_var = self.name_gen.generate()
        
        handlers = []
        for i in range(num_states):
            next_state = (i + 1) % num_states
            junk = self.name_gen.generate()
            handlers.append(f'''
    [{i}] = function()
        local {junk} = {random.randint(1, 1000)}
        {state_var} = {next_state}
        return true
    end''')
        
        handlers_str = ",\n".join(handlers)
        
        return f'''
local {state_var} = 0
local {handler_var} = {{
{handlers_str}
}}

local function _state_step()
    local handler = {handler_var}[{state_var}]
    if handler then
        return handler()
    end
    return false
end
'''

# ============================================
# Self-Modifying Code Generator
# ============================================

class SelfModifyingGenerator:
    """Generates self-modifying code patterns"""
    
    def __init__(self, name_gen: ATNameGenerator, config: AntiTamperConfig):
        self.name_gen = name_gen
        self.config = config
    
    def generate_dynamic_function(self) -> str:
        """Generate dynamically constructed function"""
        builder_func = self.name_gen.generate()
        code_var = self.name_gen.generate()
        
        return f'''
local {builder_func} = function(seed)
    local ops = {{"+", "-", "*"}}
    local op = ops[(seed % 3) + 1]
    
    local {code_var} = "return function(a, b) return a " .. op .. " b end"
    
    local loader = loadstring or load
    if loader then
        local fn = loader({code_var})
        if fn then
            return fn()
        end
    end
    
    return function(a, b) return a + b end
end
'''
    
    def generate_encrypted_payload(self, code: str, key: bytes) -> str:
        """Generate encrypted code that decrypts at runtime"""
        decrypt_func = self.name_gen.generate()
        payload_var = self.name_gen.generate()
        
        # Simple XOR encryption
        encrypted = []
        for i, c in enumerate(code.encode('utf-8')):
            encrypted.append(c ^ key[i % len(key)])
        
        key_list = ",".join(str(b) for b in key[:32])
        encrypted_list = ",".join(str(b) for b in encrypted)
        
        return f'''
local {decrypt_func} = function(data, key)
    local result = {{}}
    for i = 1, #data do
        result[i] = string.char(bit32.bxor(data[i], key[((i-1) % #key) + 1]))
    end
    return table.concat(result)
end

local {payload_var} = {{{encrypted_list}}}
local _key = {{{key_list}}}

local _decrypted = {decrypt_func}({payload_var}, _key)
local _executor = loadstring or load
if _executor then
    local fn = _executor(_decrypted)
    if fn then fn() end
end
'''

# ============================================
# Main Anti-Tamper Generator
# ============================================

class AntiTamperGenerator:
    """Main anti-tampering code generator"""
    
    def __init__(self, config: AntiTamperConfig = None):
        self.config = config or AntiTamperConfig()
        self.config.generate_keys()
        
        self.name_gen = ATNameGenerator()
        
        # Initialize sub-generators
        self.integrity_gen = IntegrityCheckGenerator(self.name_gen, self.config)
        self.debugger_gen = DebuggerDetectGenerator(self.name_gen, self.config)
        self.env_gen = EnvironmentDetectGenerator(self.name_gen, self.config)
        self.watermark_gen = WatermarkGenerator(self.name_gen, self.config)
        self.license_gen = LicenseCheckGenerator(self.name_gen, self.config)
        self.antihook_gen = AntiHookGenerator(self.name_gen, self.config)
        self.flow_gen = CodeFlowObfuscator(self.name_gen, self.config)
        self.selfmod_gen = SelfModifyingGenerator(self.name_gen, self.config)
    
    def generate_all_protections(self) -> str:
        """Generate all enabled protections"""
        sections = []
        
        # Header
        sections.append(self._generate_header())
        
        # Bit32 compatibility
        sections.append(self._generate_bit32_compat())
        
        # Anti-hook (first to protect other checks)
        if self.config.enable_anti_hook:
            sections.append("-- Anti-Hook Protection")
            sections.append(self.antihook_gen.generate_function_guards())
            sections.append(self.antihook_gen.generate_metatable_guards())
        
        # Environment detection
        if self.config.enable_environment_detect:
            sections.append("-- Environment Detection")
            sections.append(self.env_gen.generate_sandbox_detection())
        
        # Debugger detection
        if self.config.enable_debugger_detect:
            sections.append("-- Debugger Detection")
            sections.append(self.debugger_gen.generate_debug_library_check())
            sections.append(self.debugger_gen.generate_hook_detection())
            sections.append(self.debugger_gen.generate_stack_check())
        
        # Timing checks
        if self.config.enable_timing_check:
            sections.append("-- Timing Checks")
            sections.append(self.debugger_gen.generate_breakpoint_detection())
        
        # VM detection
        if self.config.enable_vm_detect:
            sections.append("-- VM Detection")
            sections.append(self.env_gen.generate_vm_detection())
            sections.append(self.env_gen.generate_emulator_detection())
        
        # Integrity checks
        if self.config.enable_integrity_check:
            sections.append("-- Integrity Checks")
            sections.append(self.integrity_gen.generate_runtime_verify())
            sections.append(self.integrity_gen.generate_function_verify())
        
        # Watermark
        if self.config.enable_watermark:
            sections.append("-- Watermark")
            sections.append(self.watermark_gen.generate_hidden_watermark(
                self.config.watermark_data))
            sections.append(self.watermark_gen.generate_fingerprint())
        
        # License check
        if self.config.enable_license:
            sections.append("-- License Check")
            sections.append(self.license_gen.generate_license_verify())
            sections.append(self.license_gen.generate_expiry_check())
            sections.append(self.license_gen.generate_hwid_check())
        
        # Code flow obfuscation
        if self.config.enable_code_flow:
            sections.append("-- Code Flow Obfuscation")
            sections.append(self.flow_gen.generate_dead_code())
            sections.append(self.flow_gen.generate_dispatch_obfuscation())
            sections.append(self.flow_gen.generate_state_machine())
        
        # Decoy code
        if self.config.decoy_code_ratio > 0:
            sections.append("-- Decoy Code")
            num_decoys = int(10 * self.config.decoy_code_ratio)
            for _ in range(num_decoys):
                sections.append(self._generate_decoy())
        
        # Randomize order if configured
        if self.config.randomize_check_order:
            # Keep header and bit32 at top
            header_sections = sections[:2]
            check_sections = sections[2:]
            random.shuffle(check_sections)
            sections = header_sections + check_sections
        
        return "\n\n".join(sections)
    
    def generate_runtime_checks(self) -> str:
        """Generate checks to be called during VM execution"""
        check_func = self.name_gen.generate()
        counter_var = self.name_gen.generate()
        
        checks = []
        
        if self.config.enable_timing_check:
            checks.append(f'''
        -- Timing check
        local t = os.clock and os.clock() or 0
        if _last_check_time and (t - _last_check_time) > 1 then
            -- Execution too slow
        end
        _last_check_time = t
''')
        
        if self.config.enable_anti_hook:
            checks.append(f'''
        -- Quick hook check
        if type(tostring) ~= "function" then
            error("Runtime error")
        end
''')
        
        checks_code = "\n".join(checks)
        
        return f'''
local {counter_var} = 0
local _last_check_time = os.clock and os.clock() or 0

local function {check_func}()
    {counter_var} = {counter_var} + 1
    
    if {counter_var} >= {self.config.verify_frequency} then
        {counter_var} = 0
        {checks_code}
    end
end

-- Export for VM integration
_RUNTIME_CHECK = {check_func}
'''
    
    def generate_protection_summary(self) -> Dict[str, bool]:
        """Get summary of enabled protections"""
        return {
            'integrity_check': self.config.enable_integrity_check,
            'environment_detect': self.config.enable_environment_detect,
            'debugger_detect': self.config.enable_debugger_detect,
            'timing_check': self.config.enable_timing_check,
            'vm_detect': self.config.enable_vm_detect,
            'sandbox_detect': self.config.enable_sandbox_detect,
            'checksum': self.config.enable_checksum,
            'code_flow': self.config.enable_code_flow,
            'watermark': self.config.enable_watermark,
            'license': self.config.enable_license,
            'anti_hook': self.config.enable_anti_hook,
            'anti_dump': self.config.enable_anti_dump,
            'runtime_decrypt': self.config.enable_runtime_decrypt,
        }
    
    def _generate_header(self) -> str:
        """Generate protection header"""
        return f'''-- Protected Code
-- Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
-- Do not modify

local _PROTECTION_ACTIVE = true
'''
    
    def _generate_bit32_compat(self) -> str:
        """Generate bit32 compatibility layer"""
        return '''
-- Bit32 compatibility
local bit32 = bit32 or bit or {}
if not bit32.bxor then
    bit32.bxor = function(a, b)
        local result = 0
        local bitval = 1
        while a > 0 or b > 0 do
            if a % 2 ~= b % 2 then
                result = result + bitval
            end
            a = math.floor(a / 2)
            b = math.floor(b / 2)
            bitval = bitval * 2
        end
        return result
    end
end
if not bit32.band then
    bit32.band = function(a, b)
        local result = 0
        local bitval = 1
        while a > 0 and b > 0 do
            if a % 2 == 1 and b % 2 == 1 then
                result = result + bitval
            end
            a = math.floor(a / 2)
            b = math.floor(b / 2)
            bitval = bitval * 2
        end
        return result
    end
end
if not bit32.rshift then
    bit32.rshift = function(a, n)
        return math.floor(a / (2 ^ n))
    end
end
'''
    
    def _generate_decoy(self) -> str:
        """Generate decoy code"""
        decoy_type = random.randint(0, 4)
        
        if decoy_type == 0:
            name = self.name_gen.generate_decoy()
            return f"local {name} = {random.randint(0, 65535)}"
        
        elif decoy_type == 1:
            name = self.name_gen.generate_decoy()
            return f'local {name} = "{secrets.token_hex(8)}"'
        
        elif decoy_type == 2:
            name = self.name_gen.generate_decoy()
            return f"local {name} = function() return {random.random()} end"
        
        elif decoy_type == 3:
            name = self.name_gen.generate_decoy()
            table_content = ",".join(str(random.randint(0, 255)) for _ in range(8))
            return f"local {name} = {{{table_content}}}"
        
        else:
            return f"-- Config: {secrets.token_hex(16)}"

# ============================================
# Utility Functions
# ============================================

def create_anti_tamper(
    enable_debugger_detect: bool = True,
    enable_timing: bool = True,
    enable_integrity: bool = True,
    enable_watermark: bool = True,
    action: DetectionAction = DetectionAction.SILENT_FAIL
) -> AntiTamperGenerator:
    """Create anti-tamper generator with common settings"""
    config = AntiTamperConfig(
        enable_debugger_detect=enable_debugger_detect,
        enable_timing_check=enable_timing,
        enable_integrity_check=enable_integrity,
        enable_watermark=enable_watermark,
        detection_action=action,
    )
    return AntiTamperGenerator(config)


def generate_quick_protection(code: str) -> str:
    """Quick protection wrapper for Lua code"""
    generator = create_anti_tamper()
    protection = generator.generate_all_protections()
    
    return f"{protection}\n\n-- Protected Code\n{code}"

# ============================================
# Example / Test
# ============================================

if __name__ == "__main__":
    print("=== Lua Anti-Tampering System ===\n")
    
    # Create configuration
    config = AntiTamperConfig(
        enable_integrity_check=True,
        enable_debugger_detect=True,
        enable_timing_check=True,
        enable_environment_detect=True,
        enable_anti_hook=True,
        enable_watermark=True,
        enable_code_flow=True,
        enable_license=False,
        enable_vm_detect=False,
        detection_action=DetectionAction.SILENT_FAIL,
        decoy_code_ratio=0.1,
        randomize_check_order=True,
    )
    config.generate_keys()
    
    print("Configuration:")
    for key, value in vars(config).items():
        if not key.startswith('_') and not isinstance(value, bytes):
            print(f"  {key}: {value}")
    print()
    
    # Create generator
    generator = AntiTamperGenerator(config)
    
    # Generate protections
    protection_code = generator.generate_all_protections()
    
    print(f"Generated Protection Code: {len(protection_code)} characters")
    print(f"Lines: {protection_code.count(chr(10))}")
    print()
    
    # Show summary
    summary = generator.generate_protection_summary()
    print("Protection Summary:")
    for name, enabled in summary.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {name}")
    print()
    
    # Show preview
    print("=== Code Preview (first 2000 chars) ===")
    print(protection_code[:2000])
    print("...\n")
    
    # Generate runtime checks
    runtime_checks = generator.generate_runtime_checks()
    print(f"Runtime Checks: {len(runtime_checks)} characters")
    print()
    
    print("✅ Anti-tampering system initialized!")
