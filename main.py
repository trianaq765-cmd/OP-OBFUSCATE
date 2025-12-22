# ============================================
# File: main.py
# Complete Integration - Main Entry Point
# Part 5: Final Integration
# ============================================

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# ============================================
# Module Checker & Auto-Installer
# ============================================

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_modules = {
        'flask': 'Flask',
        'flask_cors': 'flask-cors',
        'Crypto': 'pycryptodome',
    }
    
    missing = []
    for module, package in required_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing dependencies detected. Installing...")
        import subprocess
        for package in missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("✓ Dependencies installed")

# Check dependencies first
check_and_install_dependencies()

# Now import our modules
from cli import LuaObfuscatorCLI
from web_api import create_app, APIConfig
from config_manager import ConfigManager
from pipeline import ObfuscationPipeline, ConsoleProgress

# ============================================
# Main Application Class
# ============================================

class LuaObfuscatorApp:
    """Main application controller"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.config_manager = ConfigManager()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('LuaObfuscator')
        logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # File handler (optional)
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'obfuscator.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def run(self, args=None):
        """Main entry point"""
        parser = self._create_parser()
        args = parser.parse_args(args)
        
        try:
            if args.mode == 'cli':
                return self.run_cli(args.cli_args)
            elif args.mode == 'web':
                return self.run_web(args)
            elif args.mode == 'quick':
                return self.run_quick(args)
            elif args.mode == 'batch':
                return self.run_batch(args)
            elif args.mode == 'doctor':
                return self.run_doctor()
            else:
                parser.print_help()
                return 0
        
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description=f'Lua Obfuscator v{self.version} - Complete Protection Suite',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Modes:
  cli      - Command line interface
  web      - Web API server
  quick    - Quick obfuscation
  batch    - Batch processing
  doctor   - System check

Examples:
  # CLI mode
  python main.py cli obfuscate input.luac -o output.lua
  
  # Web server
  python main.py web --port 8080
  
  # Quick obfuscation
  python main.py quick input.luac -o output.lua --level high
  
  # Batch processing
  python main.py batch files.txt --preset extreme
  
  # System check
  python main.py doctor
            """
        )
        
        parser.add_argument('--version', action='version', 
                          version=f'Lua Obfuscator v{self.version}')
        
        subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
        
        # CLI mode
        cli_parser = subparsers.add_parser('cli', help='CLI mode')
        cli_parser.add_argument('cli_args', nargs='*', help='CLI arguments')
        
        # Web mode
        web_parser = subparsers.add_parser('web', help='Web server mode')
        web_parser.add_argument('--host', default='0.0.0.0', help='Server host')
        web_parser.add_argument('--port', type=int, default=5000, help='Server port')
        web_parser.add_argument('--debug', action='store_true', help='Debug mode')
        
        # Quick mode
        quick_parser = subparsers.add_parser('quick', help='Quick obfuscation')
        quick_parser.add_argument('input', help='Input file')
        quick_parser.add_argument('-o', '--output', required=True, help='Output file')
        quick_parser.add_argument('-l', '--level', 
                                 choices=['low', 'medium', 'high', 'extreme'],
                                 default='medium', help='Protection level')
        
        # Batch mode
        batch_parser = subparsers.add_parser('batch', help='Batch processing')
        batch_parser.add_argument('file_list', help='File with list of inputs')
        batch_parser.add_argument('--preset', default='medium', help='Config preset')
        batch_parser.add_argument('--output-dir', default='output', help='Output directory')
        
        # Doctor mode
        subparsers.add_parser('doctor', help='System check')
        
        return parser
    
    # ========================================
    # Mode Implementations
    # ========================================
    
    def run_cli(self, cli_args):
        """Run CLI mode"""
        self.logger.info("Starting CLI mode")
        cli = LuaObfuscatorCLI()
        return cli.run(cli_args)
    
    def run_web(self, args):
        """Run web server mode"""
        self.logger.info(f"Starting web server on {args.host}:{args.port}")
        
        # Set environment variables
        os.environ['HOST'] = args.host
        os.environ['PORT'] = str(args.port)
        os.environ['DEBUG'] = str(args.debug)
        
        # Create and run app
        app = create_app()
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0
    
    def run_quick(self, args):
        """Run quick obfuscation"""
        print(f"\n🚀 Quick Obfuscation")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Level:  {args.level}")
        print()
        
        # Get config
        config = self.config_manager.get_preset(args.level)
        
        # Create pipeline
        progress = ConsoleProgress(verbose=True)
        pipeline = ObfuscationPipeline(config, progress_callback=progress)
        
        # Process
        result = pipeline.process(args.input, args.output)
        
        if result.success:
            print(f"\n✅ Success!")
            print(f"Output size: {result.output_size:,} bytes")
            print(f"Time: {result.total_time:.2f}s")
            return 0
        else:
            print(f"\n❌ Failed: {result.errors[0] if result.errors else 'Unknown error'}")
            return 1
    
    def run_batch(self, args):
        """Run batch processing"""
        print(f"\n📦 Batch Processing")
        print(f"File list: {args.file_list}")
        print(f"Preset: {args.preset}")
        print()
        
        # Read file list
        if not os.path.exists(args.file_list):
            print(f"Error: File list not found: {args.file_list}")
            return 1
        
        with open(args.file_list, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        if not files:
            print("Error: No files in list")
            return 1
        
        print(f"Found {len(files)} files to process")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get config
        config = self.config_manager.get_preset(args.preset)
        
        # Process each file
        successful = 0
        failed = 0
        
        for i, input_file in enumerate(files, 1):
            if not os.path.exists(input_file):
                print(f"[{i}/{len(files)}] ❌ {input_file} - File not found")
                failed += 1
                continue
            
            # Generate output name
            base_name = Path(input_file).stem
            output_file = os.path.join(args.output_dir, f"{base_name}_obfuscated.lua")
            
            print(f"[{i}/{len(files)}] Processing {input_file}...")
            
            # Process
            pipeline = ObfuscationPipeline(config)
            result = pipeline.process(input_file, output_file)
            
            if result.success:
                print(f"  ✅ → {output_file}")
                successful += 1
            else:
                print(f"  ❌ Failed: {result.errors[0] if result.errors else 'Unknown'}")
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Batch Complete: {successful} succeeded, {failed} failed")
        print(f"{'='*60}")
        
        return 0 if failed == 0 else 1
    
    def run_doctor(self):
        """Run system check"""
        print("\n🔍 System Doctor - Checking Installation")
        print("=" * 60)
        
        checks = []
        
        # Check Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        checks.append(('Python version', py_version, sys.version_info >= (3, 7)))
        
        # Check required files
        required_files = [
            'lua_parser.py',
            'lua_transformer.py',
            'lua_vm_generator.py',
            'lua_encryption.py',
            'lua_antitamper.py',
            'config_manager.py',
            'pipeline.py',
            'cli.py',
            'web_api.py'
        ]
        
        for file in required_files:
            checks.append((f"File: {file}", 'Found' if os.path.exists(file) else 'Missing', 
                          os.path.exists(file)))
        
        # Check dependencies
        dependencies = [
            ('Flask', 'flask'),
            ('Flask-CORS', 'flask_cors'),
            ('PyCryptodome', 'Crypto'),
        ]
        
        for name, module in dependencies:
            try:
                __import__(module)
                checks.append((f"Module: {name}", 'Installed', True))
            except ImportError:
                checks.append((f"Module: {name}", 'Not installed', False))
        
        # Check directories
        for dir_name in ['uploads', 'outputs', 'logs', '.cache']:
            exists = os.path.exists(dir_name)
            if not exists:
                os.makedirs(dir_name, exist_ok=True)
            checks.append((f"Directory: {dir_name}", 'Created' if not exists else 'Exists', True))
        
        # Display results
        all_pass = True
        for check, status, passed in checks:
            icon = '✅' if passed else '❌'
            print(f"{icon} {check:30} {status}")
            if not passed:
                all_pass = False
        
        print("=" * 60)
        if all_pass:
            print("✅ All checks passed! System is ready.")
        else:
            print("❌ Some checks failed. Please fix the issues above.")
        
        return 0 if all_pass else 1

# ============================================
# Deployment Helper
# ============================================

class DeploymentHelper:
    """Helps with deployment configuration"""
    
    @staticmethod
    def generate_dockerfile():
        """Generate Dockerfile"""
        dockerfile = """
# Dockerfile for Lua Obfuscator
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./

# Create directories
RUN mkdir -p uploads outputs logs .cache

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=5000

# Run application
CMD ["python", "main.py", "web"]
"""
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile)
        print("✓ Generated Dockerfile")
    
    @staticmethod
    def generate_docker_compose():
        """Generate docker-compose.yml"""
        compose = """
version: '3.8'

services:
  obfuscator:
    build: .
    ports:
      - "5000:5000"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DEBUG=${DEBUG:-False}
      - RATE_LIMIT_PER_MIN=${RATE_LIMIT_PER_MIN:-10}
      - MAX_UPLOAD_SIZE=${MAX_UPLOAD_SIZE:-16777216}
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    restart: unless-stopped
"""
        with open('docker-compose.yml', 'w') as f:
            f.write(compose)
        print("✓ Generated docker-compose.yml")
    
    @staticmethod
    def generate_render_yaml():
        """Generate render.yaml for Render deployment"""
        render_yaml = """
services:
  - type: web
    name: lua-obfuscator
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py web
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: SECRET_KEY
        generateValue: true
      - key: RATE_LIMIT_PER_MIN
        value: 10
      - key: MAX_UPLOAD_SIZE
        value: 16777216
"""
        with open('render.yaml', 'w') as f:
            f.write(render_yaml)
        print("✓ Generated render.yaml")
    
    @staticmethod
    def generate_procfile():
        """Generate Procfile for Heroku"""
        with open('Procfile', 'w') as f:
            f.write('web: python main.py web\n')
        print("✓ Generated Procfile")
    
    @staticmethod
    def generate_railway_json():
        """Generate railway.json for Railway deployment"""
        railway = {
            "build": {
                "builder": "NIXPACKS"
            },
            "deploy": {
                "startCommand": "python main.py web",
                "healthcheckPath": "/health",
                "restartPolicyType": "ON_FAILURE",
                "restartPolicyMaxRetries": 10
            }
        }
        with open('railway.json', 'w') as f:
            json.dump(railway, f, indent=2)
        print("✓ Generated railway.json")
    
    @staticmethod
    def generate_all():
        """Generate all deployment files"""
        print("\n🚀 Generating deployment configurations...")
        DeploymentHelper.generate_dockerfile()
        DeploymentHelper.generate_docker_compose()
        DeploymentHelper.generate_render_yaml()
        DeploymentHelper.generate_procfile()
        DeploymentHelper.generate_railway_json()
        print("\n✅ All deployment files generated!")

# ============================================
# Setup Script
# ============================================

class SetupWizard:
    """Interactive setup wizard"""
    
    @staticmethod
    def run():
        """Run setup wizard"""
        print("\n🧙 Lua Obfuscator Setup Wizard")
        print("=" * 60)
        
        # Check system
        print("\n1. Checking system...")
        app = LuaObfuscatorApp()
        result = app.run_doctor()
        
        if result != 0:
            print("\n⚠️  Please fix the issues above before continuing.")
            return
        
        # Generate configs
        print("\n2. Generate deployment configs? (y/n): ", end='')
        if input().lower() == 'y':
            DeploymentHelper.generate_all()
        
        # Create example config
        print("\n3. Create example configuration? (y/n): ", end='')
        if input().lower() == 'y':
            config_manager = ConfigManager()
            example_config = config_manager.get_preset('medium')
            config_manager.save_config(example_config, 'config.example.json')
            print("✓ Created config.example.json")
        
        # Create .env file
        print("\n4. Create .env file? (y/n): ", end='')
        if input().lower() == 'y':
            env_content = """
# Lua Obfuscator Environment Variables
SECRET_KEY=change-this-to-random-string
DEBUG=False
HOST=0.0.0.0
PORT=5000
RATE_LIMIT_PER_MIN=10
MAX_UPLOAD_SIZE=16777216
"""
            with open('.env', 'w') as f:
                f.write(env_content)
            print("✓ Created .env file")
        
        print("\n" + "=" * 60)
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Edit .env file with your settings")
        print("2. Test with: python main.py quick <input> -o <output>")
        print("3. Start web server: python main.py web")
        print("4. Deploy using Docker: docker-compose up")

# ============================================
# Main Entry Point
# ============================================

def main():
    """Main entry point"""
    # Check if running setup
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        SetupWizard.run()
        return 0
    
    # Check if generating deployment configs
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy-config':
        DeploymentHelper.generate_all()
        return 0
    
    # Run main application
    app = LuaObfuscatorApp()
    return app.run()

if __name__ == '__main__':
    sys.exit(main())
