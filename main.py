# ============================================
# File: main.py (Simplified)
# Entry point untuk Discord Bot + Web API
# ============================================

import os
import sys
from threading import Thread

def run_web():
    """Run Flask web server"""
    from web_api import create_app
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

def run_bot():
    """Run Discord bot"""
    from discord_bot import run_discord_bot
    run_discord_bot()

if __name__ == '__main__':
    mode = os.getenv('MODE', 'both').lower()
    
    if mode == 'web':
        print("🌐 Starting Web API only...")
        run_web()
    elif mode == 'bot':
        print("🤖 Starting Discord Bot only...")
        run_bot()
    else:
        print("🚀 Starting Web API + Discord Bot...")
        # Run web in thread
        web_thread = Thread(target=run_web, daemon=True)
        web_thread.start()
        
        # Run bot in main thread
        run_bot()
