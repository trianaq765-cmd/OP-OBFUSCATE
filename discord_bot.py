# ============================================
# File: discord_bot.py (FIXED untuk Source Code)
# ============================================

import discord
from discord.ext import commands
import asyncio
import aiohttp
import os
import io
import time
from datetime import datetime
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import obfuscator components
try:
    from config_manager import ConfigManager
    HAS_CONFIG = True
except ImportError:
    logger.warning("config_manager not found")
    HAS_CONFIG = False
    ConfigManager = None

try:
    from pipeline import ObfuscationPipeline, PipelineResult
    HAS_PIPELINE = True
except ImportError:
    logger.warning("pipeline not found")
    HAS_PIPELINE = False

# ❌ TIDAK PERLU IMPORT INI UNTUK SOURCE CODE OBFUSCATION
# from lua_parser import parse_bytecode  

# ============================================
# Bot Configuration
# ============================================

TOKEN = os.getenv('DISCORD_TOKEN')
PREFIX = os.getenv('BOT_PREFIX', '!')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 8 * 1024 * 1024))

ALLOWED_ROLES = os.getenv('ALLOWED_ROLES', '').split(',') if os.getenv('ALLOWED_ROLES') else []
ALLOWED_USERS = os.getenv('ALLOWED_USERS', '').split(',') if os.getenv('ALLOWED_USERS') else []

# ============================================
# Bot Setup
# ============================================

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents)

config_manager = ConfigManager() if HAS_CONFIG else None
active_jobs = {}

# ============================================
# Helper Functions
# ============================================

def check_permission(ctx):
    if str(ctx.author.id) in ALLOWED_USERS:
        return True
    if ALLOWED_ROLES:
        user_roles = [role.name for role in ctx.author.roles]
        if any(role in ALLOWED_ROLES for role in user_roles):
            return True
    if not ALLOWED_ROLES and not ALLOWED_USERS:
        return True
    return False

async def download_file(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            raise Exception(f"Failed to download: {response.status}")

def create_embed(title: str, description: str = "", color=0x00ff00):
    embed = discord.Embed(
        title=title,
        description=description,
        color=color,
        timestamp=datetime.now()
    )
    embed.set_footer(text="Lua Obfuscator Bot")
    return embed

def is_lua_source(data: bytes) -> bool:
    """Check if data is Lua source code (not bytecode)"""
    # Bytecode signatures
    if data[:4] == b'\x1bLua':  # Standard Lua
        return False
    if data[:3] == b'\x1bLJ':   # LuaJIT
        return False
    if len(data) > 0 and data[0] <= 6:  # Possible Luau bytecode
        # Additional check - Luau bytecode is not valid UTF-8 text
        try:
            text = data[:100].decode('utf-8')
            # If it decodes and looks like code, it's source
            if any(kw in text for kw in ['function', 'local', 'if', 'end', 'return', '--', '=']):
                return True
        except:
            return False
    
    # Try to decode as UTF-8
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def validate_lua_source(source: str) -> tuple:
    """Basic validation of Lua source code"""
    if not source or not source.strip():
        return False, "Empty file"
    
    # Check for common Lua syntax
    lua_keywords = ['function', 'local', 'if', 'then', 'end', 'for', 'while', 
                    'do', 'return', 'nil', 'true', 'false', 'and', 'or', 'not']
    
    has_lua_content = any(kw in source for kw in lua_keywords)
    
    if not has_lua_content:
        # Mungkin masih valid (misal hanya assignment)
        if '=' not in source and '(' not in source:
            return False, "Doesn't look like Lua code"
    
    return True, None

# ============================================
# Bot Events
# ============================================

@bot.event
async def on_ready():
    logger.info(f'✅ Bot logged in as {bot.user}')
    logger.info(f'📡 Connected to {len(bot.guilds)} servers')
    logger.info(f"📦 Pipeline: {'✓' if HAS_PIPELINE else '✗'}")
    logger.info(f"📦 Config: {'✓' if HAS_CONFIG else '✗'}")
    
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name=f"{PREFIX}help | Lua Obfuscator"
        )
    )

@bot.event
async def on_command_error(ctx, error):
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
        f"{str(error)[:200]}",
        color=0xff0000
    )
    await ctx.send(embed=embed)

# ============================================
# Commands
# ============================================

@bot.command(name='help')
async def help_command(ctx):
    embed = create_embed("📚 Lua Obfuscator Help")
    
    embed.add_field(
        name="Commands",
        value=f"""
**{PREFIX}obfuscate** `[preset]` - Obfuscate Lua code
**{PREFIX}presets** - List available presets
**{PREFIX}info** `<preset>` - Preset details
**{PREFIX}stats** - Bot statistics
**{PREFIX}ping** - Check latency
        """,
        inline=False
    )
    
    embed.add_field(
        name="Usage",
        value=f"""
1. Upload `.lua` file with command:
   `{PREFIX}obfuscate [preset]`

2. Or reply to message with file

Default preset: `medium`
        """,
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command(name='ping')
async def ping(ctx):
    embed = create_embed("🏓 Pong!", f"Latency: **{round(bot.latency * 1000)}ms**")
    await ctx.send(embed=embed)

@bot.command(name='presets')
async def list_presets(ctx):
    if not HAS_CONFIG:
        await ctx.send("Config not available")
        return
    
    presets = config_manager.list_presets()
    embed = create_embed("🎨 Available Presets")
    
    preset_info = {
        'minimal': '⚪ Basic protection',
        'low': '🟢 Light obfuscation',
        'medium': '🟡 Balanced (default)',
        'high': '🟠 Strong protection',
        'extreme': '🔴 Maximum security',
        'stealth': '🟣 Anti-detection focus',
        'performance': '🔵 Speed optimized',
        'luraph': '🎨 Luraph-style'
    }
    
    for preset in presets:
        desc = preset_info.get(preset, 'Custom preset')
        embed.add_field(name=preset.capitalize(), value=desc, inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='info')
async def preset_info(ctx, preset: str = 'medium'):
    if not HAS_CONFIG:
        await ctx.send("Config not available")
        return
    
    try:
        info = config_manager.get_preset_info(preset.lower())
        embed = create_embed(
            f"ℹ️ Preset: {preset.capitalize()}",
            info.get('description', 'No description')
        )
        await ctx.send(embed=embed)
    except ValueError:
        embed = create_embed(
            "❌ Unknown Preset",
            f"Use `{PREFIX}presets` to see options",
            color=0xff0000
        )
        await ctx.send(embed=embed)

@bot.command(name='obfuscate', aliases=['obf'])
async def obfuscate(ctx, preset: str = 'medium'):
    """Obfuscate Lua source code"""
    
    # Check dependencies
    if not HAS_PIPELINE:
        embed = create_embed(
            "❌ Not Available",
            "Obfuscation pipeline not loaded",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check permission
    if not check_permission(ctx):
        embed = create_embed("❌ Permission Denied", color=0xff0000)
        await ctx.send(embed=embed)
        return
    
    # Get attachment
    attachment = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
    elif ctx.message.reference:
        try:
            ref = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            if ref.attachments:
                attachment = ref.attachments[0]
        except:
            pass
    
    if not attachment:
        embed = create_embed(
            "❌ No File",
            f"Upload a `.lua` file with `{PREFIX}obfuscate [preset]`",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check extension
    if not attachment.filename.lower().endswith('.lua'):
        embed = create_embed(
            "❌ Invalid File",
            "Please upload a `.lua` file (Lua source code)",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check size
    if attachment.size > MAX_FILE_SIZE:
        embed = create_embed(
            "❌ File Too Large",
            f"Max size: {MAX_FILE_SIZE // (1024*1024)}MB",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Start processing
    status_msg = await ctx.send(embed=create_embed(
        "⚙️ Processing",
        f"**File:** {attachment.filename}\n"
        f"**Size:** {attachment.size:,} bytes\n"
        f"**Preset:** {preset}",
        color=0x3498db
    ))
    
    try:
        start_time = time.time()
        
        # Download file
        file_data = await download_file(attachment.url)
        
        # ============================================
        # ✅ VALIDASI UNTUK SOURCE CODE, BUKAN BYTECODE
        # ============================================
        
        # Check if it's actually source code
        if not is_lua_source(file_data):
            raise Exception(
                "This looks like bytecode (.luac), not source code (.lua). "
                "Please upload Lua source code."
            )
        
        # Decode as text
        try:
            source_code = file_data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                source_code = file_data.decode('latin-1')
            except:
                raise Exception("Could not decode file as text")
        
        # Validate Lua source
        is_valid, error = validate_lua_source(source_code)
        if not is_valid:
            raise Exception(f"Invalid Lua source: {error}")
        
        # Update status
        await status_msg.edit(embed=create_embed(
            "⚙️ Processing",
            f"**File:** {attachment.filename}\n"
            f"**Lines:** {len(source_code.splitlines()):,}\n"
            f"**Status:** Obfuscating...",
            color=0x3498db
        ))
        
        # Get config
        try:
            config = config_manager.get_preset(preset.lower())
        except ValueError:
            raise Exception(f"Unknown preset: {preset}")
        
        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', 
                                          delete=False, encoding='utf-8') as f:
            f.write(source_code)
            input_path = f.name
        
        output_path = input_path.replace('.lua', '_obfuscated.lua')
        
        # Track job
        job_id = str(ctx.message.id)
        active_jobs[job_id] = {
            'user': ctx.author.id,
            'start': time.time(),
            'status': 'processing'
        }
        
        # ============================================
        # ✅ PROSES OBFUSCATION (SOURCE CODE)
        # ============================================
        
        pipeline = ObfuscationPipeline(config)
        result = await asyncio.to_thread(
            pipeline.process,
            input_path,
            output_path
        )
        
        elapsed = time.time() - start_time
        active_jobs[job_id]['status'] = 'completed' if result.success else 'failed'
        
        if result.success:
            # Read output
            with open(output_path, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            # Success embed
            embed = create_embed(
                "✅ Obfuscation Complete",
                f"**Input:** {attachment.filename}\n"
                f"**Preset:** {preset}\n"
                f"**Input Size:** {len(source_code):,} chars\n"
                f"**Output Size:** {len(output_content):,} chars\n"
                f"**Ratio:** {len(output_content)/len(source_code):.2f}x\n"
                f"**Time:** {elapsed:.2f}s",
                color=0x00ff00
            )
            await status_msg.edit(embed=embed)
            
            # Send file
            output_filename = attachment.filename.replace('.lua', '_obfuscated.lua')
            await ctx.send(
                f"<@{ctx.author.id}> Here's your obfuscated code:",
                file=discord.File(io.StringIO(output_content), filename=output_filename)
            )
        else:
            error_msg = result.errors[0] if result.errors else "Unknown error"
            embed = create_embed(
                "❌ Obfuscation Failed",
                f"Error: {error_msg[:500]}",
                color=0xff0000
            )
            await status_msg.edit(embed=embed)
        
        # Cleanup
        try:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass
        
        if job_id in active_jobs:
            del active_jobs[job_id]
            
    except Exception as e:
        logger.error(f"Obfuscation error: {e}")
        embed = create_embed(
            "❌ Error",
            f"{str(e)[:500]}",
            color=0xff0000
        )
        await status_msg.edit(embed=embed)

@bot.command(name='stats')
async def stats(ctx):
    embed = create_embed("📊 Bot Statistics")
    
    embed.add_field(
        name="Bot",
        value=f"Servers: {len(bot.guilds)}\nLatency: {round(bot.latency*1000)}ms",
        inline=True
    )
    
    embed.add_field(
        name="Status",
        value=f"Pipeline: {'✅' if HAS_PIPELINE else '❌'}\n"
              f"Config: {'✅' if HAS_CONFIG else '❌'}",
        inline=True
    )
    
    embed.add_field(
        name="Jobs",
        value=f"Active: {len(active_jobs)}",
        inline=True
    )
    
    await ctx.send(embed=embed)

# ============================================
# Main
# ============================================

def run_discord_bot():
    if not TOKEN:
        print("❌ DISCORD_TOKEN not set!")
        return
    
    logger.info(f"🤖 Starting bot with prefix: {PREFIX}")
    
    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        logger.error("❌ Invalid token")
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    run_discord_bot()