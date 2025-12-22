# ============================================
# File: discord_bot.py (FIXED)
# Discord Bot untuk Lua Obfuscator
# ============================================

import discord
from discord.ext import commands
import asyncio
import aiohttp
import os
import io
import json
import time
from datetime import datetime
import tempfile
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Safe Imports with Error Handling
# ============================================

try:
    from config_manager import ConfigManager
    HAS_CONFIG_MANAGER = True
except ImportError:
    logger.warning("config_manager not found, using defaults")
    HAS_CONFIG_MANAGER = False
    
    # Dummy ConfigManager
    class ConfigManager:
        def list_presets(self):
            return ['minimal', 'low', 'medium', 'high', 'extreme']
        
        def get_preset(self, name):
            return None
        
        def get_preset_info(self, name):
            return {'description': f'{name} preset'}

try:
    from pipeline import ObfuscationPipeline, PipelineResult
    HAS_PIPELINE = True
except ImportError:
    logger.warning("pipeline not found")
    HAS_PIPELINE = False

try:
    # Import FIXED parser
    from lua_parser import parse_bytecode, detect_format, LuaChunk
    HAS_PARSER = True
except ImportError:
    logger.warning("lua_parser not found")
    HAS_PARSER = False
    
    def parse_bytecode(data):
        """Dummy parser"""
        return None
    
    def detect_format(data):
        return "UNKNOWN", "UNKNOWN"

# ============================================
# Bot Configuration
# ============================================

TOKEN = os.getenv('DISCORD_TOKEN')
PREFIX = os.getenv('BOT_PREFIX', '!')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 8 * 1024 * 1024))  # 8MB

ALLOWED_ROLES = [r.strip() for r in os.getenv('ALLOWED_ROLES', '').split(',') if r.strip()]
ALLOWED_USERS = [u.strip() for u in os.getenv('ALLOWED_USERS', '').split(',') if u.strip()]

# ============================================
# Bot Setup
# ============================================

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents)

config_manager = ConfigManager()
active_jobs = {}

# ============================================
# Helper Functions
# ============================================

def check_permission(ctx) -> bool:
    """Check if user has permission"""
    if str(ctx.author.id) in ALLOWED_USERS:
        return True
    
    if hasattr(ctx.author, 'roles') and ALLOWED_ROLES:
        user_roles = [role.name for role in ctx.author.roles]
        if any(role in ALLOWED_ROLES for role in user_roles):
            return True
    
    if not ALLOWED_ROLES and not ALLOWED_USERS:
        return True
    
    return False

async def download_file(url: str) -> bytes:
    """Download file from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            raise Exception(f"Failed to download: HTTP {response.status}")

def create_embed(title: str, description: str = "", color: int = 0x00ff00) -> discord.Embed:
    """Create Discord embed"""
    embed = discord.Embed(
        title=title,
        description=description,
        color=color,
        timestamp=datetime.now()
    )
    embed.set_footer(text="Lua Obfuscator Bot")
    return embed

def validate_bytecode(data: bytes) -> tuple:
    """
    Validate bytecode and detect format.
    Returns: (is_valid, format_name, version_name, error_message)
    """
    if not data or len(data) < 4:
        return False, None, None, "File too small"
    
    try:
        # Use the FIXED parser's detect_format
        if HAS_PARSER:
            fmt, version = detect_format(data)
            
            if fmt == "UNKNOWN":
                return False, None, None, "Unknown bytecode format"
            
            # Try to parse
            chunk = parse_bytecode(data)
            
            if chunk and hasattr(chunk, 'header'):
                if chunk.header.is_valid:
                    return True, fmt, version, None
                else:
                    errors = chunk.header.validation_errors
                    return False, fmt, version, "; ".join(errors[:3])
            
            return True, fmt, version, None
        else:
            # Basic validation without parser
            if data[:4] == b'\x1bLua':
                return True, "STANDARD_LUA", f"5.{data[4] & 0x0F}", None
            elif data[:3] == b'\x1bLJ':
                return True, "LUAJIT", "2.x", None
            elif 0 <= data[0] <= 6:
                return True, "LUAU", f"v{data[0]}", None
            else:
                return False, None, None, "Unknown format"
                
    except Exception as e:
        return False, None, None, str(e)[:100]

# ============================================
# Bot Events
# ============================================

@bot.event
async def on_ready():
    """Bot startup"""
    logger.info(f'✅ Bot logged in as {bot.user}')
    logger.info(f'📡 Connected to {len(bot.guilds)} servers')
    
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name=f"{PREFIX}help | Lua Obfuscator"
        )
    )

@bot.event
async def on_command_error(ctx, error):
    """Handle errors"""
    if isinstance(error, commands.CommandNotFound):
        return
    
    if isinstance(error, commands.MissingRequiredArgument):
        embed = create_embed(
            "❌ Missing Argument",
            f"Required: `{error.param.name}`",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    logger.error(f"Command error: {error}")
    embed = create_embed(
        "❌ Error",
        str(error)[:200],
        color=0xff0000
    )
    await ctx.send(embed=embed)

# ============================================
# Bot Commands
# ============================================

@bot.command(name='help')
async def help_command(ctx):
    """Show help"""
    embed = create_embed("📚 Lua Obfuscator Bot Help")
    
    commands_list = f"""
**{PREFIX}obfuscate** - Obfuscate Lua bytecode
**{PREFIX}analyze** - Analyze bytecode (detect format)
**{PREFIX}presets** - List available presets
**{PREFIX}info** `<preset>` - Get preset details
**{PREFIX}stats** - Bot statistics
**{PREFIX}ping** - Check latency
    """
    
    embed.add_field(name="Commands", value=commands_list, inline=False)
    
    usage = f"""
1. Upload a `.luac` file with:
   `{PREFIX}obfuscate [preset]`

2. Or reply to a message with file

**Supported formats:**
• Lua 5.1, 5.2, 5.3, 5.4
• LuaJIT 2.0, 2.1
• Roblox Luau
    """
    
    embed.add_field(name="Usage", value=usage, inline=False)
    
    await ctx.send(embed=embed)

@bot.command(name='ping')
async def ping(ctx):
    """Check latency"""
    latency = round(bot.latency * 1000)
    embed = create_embed("🏓 Pong!", f"Latency: **{latency}ms**")
    await ctx.send(embed=embed)

@bot.command(name='presets')
async def list_presets(ctx):
    """List available presets"""
    presets = config_manager.list_presets()
    
    embed = create_embed("🎨 Available Presets")
    
    # FIX: Indentasi benar
    preset_info = {
        'minimal': '⚪ Basic protection only',
        'low': '🟢 Light obfuscation',
        'medium': '🟡 Balanced security (default)',
        'high': '🟠 Strong protection',
        'extreme': '🔴 Maximum security',
        'stealth': '🟣 Anti-detection focus',
        'performance': '🔵 Speed optimized',
        'luraph': '🎨 Luraph-style output'
    }
    
    for preset in presets:
        description = preset_info.get(preset, 'Custom preset')
        embed.add_field(
            name=preset.capitalize(),
            value=description,
            inline=True
        )
    
    await ctx.send(embed=embed)

@bot.command(name='info')
async def preset_info(ctx, preset: str = 'medium'):
    """Get preset information"""
    try:
        info = config_manager.get_preset_info(preset.lower())
        config = config_manager.get_preset(preset.lower())
        
        embed = create_embed(
            f"ℹ️ Preset: {preset.capitalize()}",
            info.get('description', 'No description')
        )
        
        features = []
        if config:
            if hasattr(config, 'transform'):
                if getattr(config.transform, 'shuffle_opcodes', False):
                    features.append("✓ Opcode shuffling")
                if getattr(config.transform, 'encrypt_strings', False):
                    features.append("✓ String encryption")
                if getattr(config.transform, 'encrypt_numbers', False):
                    features.append("✓ Number encryption")
                if getattr(config.transform, 'add_junk_code', False):
                    features.append("✓ Junk code injection")
            
            if hasattr(config, 'antitamper'):
                if getattr(config.antitamper, 'enable_debugger_detect', False):
                    features.append("✓ Anti-debugging")
                if getattr(config.antitamper, 'enable_integrity_check', False):
                    features.append("✓ Integrity checks")
        
        embed.add_field(
            name="Features",
            value="\n".join(features) if features else "Basic obfuscation",
            inline=False
        )
        
        await ctx.send(embed=embed)
        
    except ValueError:
        embed = create_embed(
            "❌ Unknown Preset",
            f"Preset `{preset}` not found. Use `{PREFIX}presets`",
            color=0xff0000
        )
        await ctx.send(embed=embed)

@bot.command(name='analyze')
async def analyze(ctx):
    """Analyze bytecode format (NEW COMMAND)"""
    
    # Get attachment
    attachment = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
    elif ctx.message.reference:
        try:
            ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            if ref_msg.attachments:
                attachment = ref_msg.attachments[0]
        except:
            pass
    
    if not attachment:
        embed = create_embed(
            "❌ No File",
            "Please upload or reply to a bytecode file.",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Download and analyze
    try:
        data = await download_file(attachment.url)
        is_valid, fmt, version, error = validate_bytecode(data)
        
        if is_valid:
            embed = create_embed(
                "✅ Bytecode Analysis",
                f"**File:** {attachment.filename}\n"
                f"**Size:** {len(data):,} bytes\n"
                f"**Format:** {fmt}\n"
                f"**Version:** {version}"
            )
            
            # Try to get more details
            if HAS_PARSER:
                try:
                    chunk = parse_bytecode(data)
                    if chunk and hasattr(chunk, 'stats'):
                        stats = chunk.stats
                        embed.add_field(
                            name="Statistics",
                            value=f"Functions: {stats.get('total_functions', 'N/A')}\n"
                                  f"Instructions: {stats.get('total_instructions', 'N/A')}\n"
                                  f"Constants: {stats.get('total_constants', 'N/A')}\n"
                                  f"Strings: {stats.get('total_strings', 'N/A')}",
                            inline=False
                        )
                except:
                    pass
        else:
            embed = create_embed(
                "❌ Invalid Bytecode",
                f"**File:** {attachment.filename}\n"
                f"**Error:** {error or 'Unknown format'}",
                color=0xff0000
            )
            
            # Show hex dump
            hex_preview = data[:32].hex(' ')
            embed.add_field(
                name="Hex Preview",
                value=f"```\n{hex_preview}\n```",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        embed = create_embed(
            "❌ Error",
            str(e)[:200],
            color=0xff0000
        )
        await ctx.send(embed=embed)

@bot.command(name='obfuscate', aliases=['obf'])
async def obfuscate(ctx, preset: str = 'medium'):
    """Obfuscate Lua bytecode"""
    
    if not check_permission(ctx):
        embed = create_embed(
            "❌ Permission Denied",
            "You don't have permission to use this command.",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Get attachment
    attachment = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
    elif ctx.message.reference:
        try:
            ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            if ref_msg.attachments:
                attachment = ref_msg.attachments[0]
        except:
            pass
    
    if not attachment:
        embed = create_embed(
            "❌ No File",
            f"Upload a `.luac` file or reply to one.\n"
            f"Usage: `{PREFIX}obfuscate [preset]`",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check extension
    valid_extensions = ('.luac', '.lua', '.out', '.bin')
    if not any(attachment.filename.lower().endswith(ext) for ext in valid_extensions):
        embed = create_embed(
            "❌ Invalid File",
            f"Supported: {', '.join(valid_extensions)}",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check size
    if attachment.size > MAX_FILE_SIZE:
        embed = create_embed(
            "❌ File Too Large",
            f"Maximum: {MAX_FILE_SIZE / (1024*1024):.1f}MB",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Start processing
    start_embed = create_embed(
        "⚙️ Processing",
        f"**File:** {attachment.filename}\n"
        f"**Size:** {attachment.size:,} bytes\n"
        f"**Preset:** {preset}",
        color=0x3498db
    )
    status_msg = await ctx.send(embed=start_embed)
    
    try:
        # Download
        file_data = await download_file(attachment.url)
        
        # Validate with FIXED parser
        is_valid, fmt, version, error = validate_bytecode(file_data)
        
        if not is_valid:
            raise Exception(f"Invalid bytecode ({fmt or 'unknown'}): {error}")
        
        # Update status
        await status_msg.edit(embed=create_embed(
            "⚙️ Processing",
            f"**Format:** {fmt} {version}\n"
            f"**Preset:** {preset}\n"
            "Obfuscating...",
            color=0x3498db
        ))
        
        # Check if pipeline available
        if not HAS_PIPELINE:
            raise Exception("Obfuscation pipeline not available")
        
        # Process
        with tempfile.NamedTemporaryFile(suffix='.luac', delete=False) as f:
            f.write(file_data)
            input_path = f.name
        
        output_path = input_path.replace('.luac', '_obfuscated.lua')
        
        try:
            config = config_manager.get_preset(preset.lower())
        except ValueError:
            raise Exception(f"Unknown preset: {preset}")
        
        if config is None:
            raise Exception(f"Preset '{preset}' not configured")
        
        pipeline = ObfuscationPipeline(config)
        
        job_id = str(ctx.message.id)
        active_jobs[job_id] = {
            'user': ctx.author.id,
            'start': time.time(),
            'status': 'processing'
        }
        
        result = await asyncio.to_thread(pipeline.process, input_path, output_path)
        
        active_jobs[job_id]['status'] = 'completed' if result.success else 'failed'
        
        if result.success:
            with open(output_path, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            success_embed = create_embed(
                "✅ Obfuscation Complete",
                f"**Format:** {fmt} {version}\n"
                f"**Input:** {result.input_size:,} bytes\n"
                f"**Output:** {result.output_size:,} bytes\n"
                f"**Ratio:** {result.size_ratio:.2f}x\n"
                f"**Time:** {result.total_time:.2f}s"
            )
            
            await status_msg.edit(embed=success_embed)
            
            output_filename = attachment.filename.rsplit('.', 1)[0] + '_obfuscated.lua'
            await ctx.send(
                f"<@{ctx.author.id}> Here's your obfuscated file:",
                file=discord.File(io.StringIO(output_content), filename=output_filename)
            )
        else:
            error_msg = result.errors[0] if result.errors else "Unknown error"
            raise Exception(error_msg)
        
        # Cleanup
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        
        if job_id in active_jobs:
            del active_jobs[job_id]
        
    except Exception as e:
        logger.error(f"Obfuscation error: {e}")
        error_embed = create_embed(
            "❌ Obfuscation Failed",
            str(e)[:500],
            color=0xff0000
        )
        await status_msg.edit(embed=error_embed)

@bot.command(name='stats')
async def stats(ctx):
    """Bot statistics"""
    embed = create_embed("📊 Bot Statistics")
    
    total_members = sum(g.member_count or 0 for g in bot.guilds)
    
    embed.add_field(
        name="Bot Info",
        value=f"**Servers:** {len(bot.guilds)}\n"
              f"**Users:** {total_members:,}\n"
              f"**Latency:** {round(bot.latency * 1000)}ms",
        inline=True
    )
    
    embed.add_field(
        name="Jobs",
        value=f"**Active:** {len(active_jobs)}\n"
              f"**Presets:** {len(config_manager.list_presets())}",
        inline=True
    )
    
    # Components status
    components = []
    components.append(f"{'✅' if HAS_PARSER else '❌'} Parser")
    components.append(f"{'✅' if HAS_PIPELINE else '❌'} Pipeline")
    components.append(f"{'✅' if HAS_CONFIG_MANAGER else '❌'} Config")
    
    embed.add_field(
        name="Components",
        value="\n".join(components),
        inline=True
    )
    
    # System info
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024
        embed.add_field(
            name="System",
            value=f"**Memory:** {memory:.1f}MB\n"
                  f"**CPU:** {psutil.cpu_percent()}%",
            inline=True
        )
    except ImportError:
        pass
    
    await ctx.send(embed=embed)

# ============================================
# Main Entry Point
# ============================================

def run_discord_bot():
    """Run Discord bot"""
    if not TOKEN:
        logger.error("❌ DISCORD_TOKEN not set!")
        print("Please set DISCORD_TOKEN environment variable")
        return
    
    logger.info("🤖 Starting Discord bot...")
    logger.info(f"📍 Prefix: {PREFIX}")
    logger.info(f"📦 Parser: {'Available' if HAS_PARSER else 'NOT FOUND'}")
    logger.info(f"📦 Pipeline: {'Available' if HAS_PIPELINE else 'NOT FOUND'}")
    
    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        logger.error("❌ Invalid Discord token")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    run_discord_bot()