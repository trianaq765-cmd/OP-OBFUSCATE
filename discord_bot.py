# ============================================
# File: discord_bot.py
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

# Import obfuscator components
from config_manager import ConfigManager
from pipeline import ObfuscationPipeline, PipelineResult
from lua_parser import parse_bytecode

# ============================================
# Bot Configuration
# ============================================

TOKEN = os.getenv('DISCORD_TOKEN')
PREFIX = os.getenv('BOT_PREFIX', '!')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 8 * 1024 * 1024))  # 8MB

# Allowed roles/users
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

# Config manager
config_manager = ConfigManager()

# Active jobs tracking
active_jobs = {}

# ============================================
# Helper Functions
# ============================================

def check_permission(ctx):
    """Check if user has permission to use bot"""
    # Check if user ID is allowed
    if str(ctx.author.id) in ALLOWED_USERS:
        return True
    
    # Check if user has allowed role
    if ALLOWED_ROLES:
        user_roles = [role.name for role in ctx.author.roles]
        if any(role in ALLOWED_ROLES for role in user_roles):
            return True
    
    # If no restrictions set, allow everyone
    if not ALLOWED_ROLES and not ALLOWED_USERS:
        return True
    
    return False

async def download_file(url: str) -> bytes:
    """Download file from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            raise Exception(f"Failed to download file: {response.status}")

def create_embed(title: str, description: str = "", color=0x00ff00):
    """Create Discord embed"""
    embed = discord.Embed(
        title=title,
        description=description,
        color=color,
        timestamp=datetime.now()
    )
    embed.set_footer(text="Lua Obfuscator Bot")
    return embed

# ============================================
# Bot Events
# ============================================

@bot.event
async def on_ready():
    """Bot startup"""
    print(f'✅ Bot logged in as {bot.user}')
    print(f'📡 Connected to {len(bot.guilds)} servers')
    
    # Set status
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
            f"Command requires: `{error.param.name}`",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Log other errors
    print(f"Error: {error}")
    embed = create_embed(
        "❌ Error",
        f"An error occurred: {str(error)[:200]}",
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
    **{PREFIX}presets** - List available presets
    **{PREFIX}info** `<preset>` - Get preset details
    **{PREFIX}stats** - Bot statistics
    **{PREFIX}ping** - Check bot latency
    """
    
    embed.add_field(name="Commands", value=commands_list, inline=False)
    
    usage = f"""
    1. Upload a `.luac` file with command:
    `{PREFIX}obfuscate [preset]`
    
    2. Or reply to a message with file:
    `{PREFIX}obfuscate [preset]`
    
    Default preset: `medium`
    """
    
    embed.add_field(name="Usage", value=usage, inline=False)
    
    await ctx.send(embed=embed)

@bot.command(name='ping')
async def ping(ctx):
    """Check bot latency"""
    latency = round(bot.latency * 1000)
    embed = create_embed(
        "🏓 Pong!",
        f"Latency: **{latency}ms**"
    )
    await ctx.send(embed=embed)

@bot.command(name='presets')
async def list_presets(ctx):
    """List available presets"""
    presets = config_manager.list_presets()
    
    embed = create_embed("🎨 Available Presets")
    
    preset_info = {
        'minimal': '⚪ Basic protection only',
        'low': '🟢 Light obfuscation',
        'medium': '🟡 Balanced security (default)',
        'high': '🟠 Strong protection',
        'extreme': '🔴 Maximum security',
        'stealth': '🟣 Anti-detection focus',
        'performance': '🔵 Speed optimized'
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
            info['description']
        )
        
        # Add configuration details
        features = []
        if config.transform.shuffle_opcodes:
            features.append("✓ Opcode shuffling")
        if config.transform.encrypt_strings:
            features.append("✓ String encryption")
        if config.transform.encrypt_numbers:
            features.append("✓ Number encryption")
        if config.transform.add_junk_code:
            features.append("✓ Junk code injection")
        if config.antitamper.enable_debugger_detect:
            features.append("✓ Anti-debugging")
        if config.antitamper.enable_integrity_check:
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
            f"Preset `{preset}` not found. Use `{PREFIX}presets` to see available options.",
            color=0xff0000
        )
        await ctx.send(embed=embed)

@bot.command(name='obfuscate', aliases=['obf'])
async def obfuscate(ctx, preset: str = 'medium'):
    """Obfuscate Lua bytecode"""
    
    # Check permission
    if not check_permission(ctx):
        embed = create_embed(
            "❌ Permission Denied",
            "You don't have permission to use this command.",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check for attachment
    attachment = None
    
    # Check current message
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
    
    # Check referenced message (reply)
    elif ctx.message.reference:
        referenced = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if referenced.attachments:
            attachment = referenced.attachments[0]
    
    if not attachment:
        embed = create_embed(
            "❌ No File",
            f"Please upload a `.luac` file or reply to a message with the file.\n"
            f"Usage: `{PREFIX}obfuscate [preset]`",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check file extension
    if not attachment.filename.endswith('.luac'):
        embed = create_embed(
            "❌ Invalid File",
            "Please upload a `.luac` (Lua bytecode) file.",
            color=0xff0000
        )
        await ctx.send(embed=embed)
        return
    
    # Check file size
    if attachment.size > MAX_FILE_SIZE:
        embed = create_embed(
            "❌ File Too Large",
            f"Maximum file size: {MAX_FILE_SIZE / (1024*1024):.1f}MB",
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
        # Download file
        file_data = await download_file(attachment.url)
        
        # Validate bytecode
        try:
            parse_bytecode(file_data)
        except Exception as e:
            raise Exception(f"Invalid bytecode: {str(e)[:100]}")
        
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.luac', delete=False) as input_file:
            input_file.write(file_data)
            input_path = input_file.name
        
        output_path = input_path.replace('.luac', '_obfuscated.lua')
        
        # Get configuration
        try:
            config = config_manager.get_preset(preset.lower())
        except ValueError:
            raise Exception(f"Unknown preset: {preset}")
        
        # Create pipeline
        pipeline = ObfuscationPipeline(config)
        
        # Store job
        job_id = str(ctx.message.id)
        active_jobs[job_id] = {
            'user': ctx.author.id,
            'start': time.time(),
            'status': 'processing'
        }
        
        # Process file
        result = await asyncio.to_thread(
            pipeline.process,
            input_path,
            output_path
        )
        
        # Update job status
        active_jobs[job_id]['status'] = 'completed' if result.success else 'failed'
        
        if result.success:
            # Read output file
            with open(output_path, 'r', encoding='utf-8') as f:
                output_content = f.read()
            
            # Create success embed
            success_embed = create_embed(
                "✅ Obfuscation Complete",
                f"**Input:** {attachment.filename} ({result.input_size:,} bytes)\n"
                f"**Output:** {result.output_size:,} bytes\n"
                f"**Ratio:** {result.size_ratio:.2f}x\n"
                f"**Time:** {result.total_time:.2f}s",
                color=0x00ff00
            )
            
            # Add statistics
            if result.stats:
                stats_text = f"Functions: {result.stats.get('input', {}).get('functions', 0)}\n"
                stats_text += f"Instructions: {result.stats.get('input', {}).get('instructions', 0)}"
                success_embed.add_field(name="Statistics", value=stats_text, inline=False)
            
            # Update status message
            await status_msg.edit(embed=success_embed)
            
            # Send obfuscated file
            output_filename = attachment.filename.replace('.luac', '_obfuscated.lua')
            output_file = discord.File(
                io.StringIO(output_content),
                filename=output_filename
            )
            
            await ctx.send(
                f"<@{ctx.author.id}> Here's your obfuscated file:",
                file=output_file
            )
            
        else:
            # Failed
            error_msg = result.errors[0] if result.errors else "Unknown error"
            error_embed = create_embed(
                "❌ Obfuscation Failed",
                f"Error: {error_msg[:200]}",
                color=0xff0000
            )
            await status_msg.edit(embed=error_embed)
        
        # Cleanup
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        
        # Remove job
        del active_jobs[job_id]
        
    except Exception as e:
        error_embed = create_embed(
            "❌ Error",
            f"Failed to process file: {str(e)[:200]}",
            color=0xff0000
        )
        await status_msg.edit(embed=error_embed)

@bot.command(name='stats')
async def stats(ctx):
    """Show bot statistics"""
    embed = create_embed("📊 Bot Statistics")
    
    # Bot stats
    embed.add_field(
        name="Bot Info",
        value=f"**Servers:** {len(bot.guilds)}\n"
              f"**Users:** {sum(g.member_count for g in bot.guilds)}\n"
              f"**Latency:** {round(bot.latency * 1000)}ms",
        inline=True
    )
    
    # Active jobs
    embed.add_field(
        name="Jobs",
        value=f"**Active:** {len(active_jobs)}\n"
              f"**Presets:** {len(config_manager.list_presets())}",
        inline=True
    )
    
    # System
    import psutil
    if hasattr(psutil, 'Process'):
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024
        embed.add_field(
            name="System",
            value=f"**Memory:** {memory:.1f}MB\n"
                  f"**CPU:** {psutil.cpu_percent()}%",
            inline=True
        )
    
    await ctx.send(embed=embed)

# ============================================
# Slash Commands (Optional)
# ============================================

@bot.tree.command(name="obfuscate", description="Obfuscate Lua bytecode")
async def slash_obfuscate(interaction: discord.Interaction, file: discord.Attachment, preset: str = "medium"):
    """Slash command for obfuscation"""
    # Defer response
    await interaction.response.defer()
    
    # Similar logic to regular command
    # ... (simplified for brevity)
    
    await interaction.followup.send("Processing your file...")

# ============================================
# Main Entry Point
# ============================================

def run_discord_bot():
    """Run Discord bot"""
    if not TOKEN:
        print("❌ DISCORD_TOKEN not set in environment variables")
        return
    
    print("🤖 Starting Discord bot...")
    print(f"📍 Prefix: {PREFIX}")
    print(f"🔒 Restricted: {'Yes' if ALLOWED_ROLES or ALLOWED_USERS else 'No'}")
    
    try:
        bot.run(TOKEN)
    except discord.LoginFailure:
        print("❌ Invalid Discord token")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_discord_bot()
