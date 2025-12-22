# ============================================
# File: web_api.py
# Flask Web API untuk Lua Obfuscator
# Part 4: Web API
# ============================================

import os
import sys
import io
import time
import uuid
import secrets
import hashlib
import threading
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import asdict
from datetime import datetime, timedelta
import json

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Import komponen obfuscator
from config_manager import ConfigManager, ObfuscatorConfig
from pipeline import ObfuscationPipeline, PipelineResult, ProgressCallback
from lua_parser import parse_bytecode

# ============================================
# Configuration
# ============================================

class APIConfig:
    """API Configuration"""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Upload
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'outputs')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_UPLOAD_SIZE', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'luac', 'lua'}
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT', 'True').lower() == 'true'
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MIN', 10))
    
    # Job management
    MAX_JOBS = int(os.getenv('MAX_JOBS', 100))
    JOB_TIMEOUT = int(os.getenv('JOB_TIMEOUT', 300))  # 5 minutes
    CLEANUP_INTERVAL = int(os.getenv('CLEANUP_INTERVAL', 3600))  # 1 hour
    
    # API Keys
    REQUIRE_API_KEY = os.getenv('REQUIRE_API_KEY', 'False').lower() == 'true'
    API_KEYS = os.getenv('API_KEYS', '').split(',') if os.getenv('API_KEYS') else []

# ============================================
# Job Management
# ============================================

class Job:
    """Represents an obfuscation job"""
    
    def __init__(self, job_id: str, input_file: str, config_name: str):
        self.id = job_id
        self.input_file = input_file
        self.config_name = config_name
        self.status = 'pending'  # pending, processing, completed, failed
        self.progress = 0.0
        self.current_stage = ''
        self.result: Optional[PipelineResult] = None
        self.output_file: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'status': self.status,
            'progress': self.progress,
            'current_stage': self.current_stage,
            'config': self.config_name,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error,
            'result': self._result_to_dict() if self.result else None,
        }
    
    def _result_to_dict(self) -> Dict:
        """Convert result to dict"""
        if not self.result:
            return {}
        
        return {
            'success': self.result.success,
            'input_size': self.result.input_size,
            'output_size': self.result.output_size,
            'size_ratio': self.result.size_ratio,
            'total_time': self.result.total_time,
            'stage_times': self.result.stage_times,
            'stats': self.result.stats,
            'errors': self.result.errors,
            'warnings': self.result.warnings,
        }

class JobManager:
    """Manages obfuscation jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def create_job(self, input_file: str, config_name: str) -> Job:
        """Create new job"""
        job_id = str(uuid.uuid4())
        job = Job(job_id, input_file, config_name)
        
        with self.lock:
            if len(self.jobs) >= APIConfig.MAX_JOBS:
                # Remove oldest completed jobs
                self._cleanup_old_jobs()
            
            self.jobs[job_id] = job
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, **kwargs):
        """Update job properties"""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                for key, value in kwargs.items():
                    setattr(job, key, value)
    
    def _cleanup_old_jobs(self):
        """Clean up old completed jobs"""
        cutoff = datetime.now() - timedelta(seconds=APIConfig.JOB_TIMEOUT)
        
        to_remove = []
        for job_id, job in self.jobs.items():
            if job.status in ['completed', 'failed'] and job.completed_at:
                if job.completed_at < cutoff:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
    
    def _cleanup_loop(self):
        """Periodic cleanup loop"""
        while True:
            time.sleep(APIConfig.CLEANUP_INTERVAL)
            with self.lock:
                self._cleanup_old_jobs()

# ============================================
# Progress Callback for API
# ============================================

class APIProgressCallback(ProgressCallback):
    """Progress callback for API jobs"""
    
    def __init__(self, job_manager: JobManager, job_id: str):
        self.job_manager = job_manager
        self.job_id = job_id
        self.stage_count = 9
    
    def on_stage_start(self, stage, stage_name: str):
        progress = (stage.value - 1) / self.stage_count
        self.job_manager.update_job(
            self.job_id,
            current_stage=stage_name,
            progress=progress
        )
    
    def on_stage_complete(self, stage, stage_name: str, elapsed: float):
        progress = stage.value / self.stage_count
        self.job_manager.update_job(
            self.job_id,
            progress=progress
        )
    
    def on_error(self, stage, error: str):
        self.job_manager.update_job(
            self.job_id,
            status='failed',
            error=error
        )

# ============================================
# Rate Limiting
# ============================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int, window: int = 60):
        self.max_requests = max_requests
        self.window = window  # seconds
        self.requests: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        if not APIConfig.RATE_LIMIT_ENABLED:
            return True
        
        now = time.time()
        
        with self.lock:
            # Clean old requests
            if key in self.requests:
                self.requests[key] = [
                    t for t in self.requests[key]
                    if now - t < self.window
                ]
            else:
                self.requests[key] = []
            
            # Check limit
            if len(self.requests[key]) >= self.max_requests:
                return False
            
            # Add request
            self.requests[key].append(now)
            return True
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests"""
        if not APIConfig.RATE_LIMIT_ENABLED:
            return 999
        
        with self.lock:
            if key not in self.requests:
                return self.max_requests
            
            return self.max_requests - len(self.requests[key])

# ============================================
# Flask Application
# ============================================

def create_app() -> Flask:
    """Create and configure Flask app"""
    app = Flask(__name__)
    app.config.from_object(APIConfig)
    
    # Enable CORS
    CORS(app)
    
    # Create directories
    os.makedirs(APIConfig.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(APIConfig.OUTPUT_FOLDER, exist_ok=True)
    
    # Initialize components
    config_manager = ConfigManager()
    job_manager = JobManager()
    rate_limiter = RateLimiter(APIConfig.RATE_LIMIT_PER_MINUTE)
    
    # ========================================
    # Middleware
    # ========================================
    
    @app.before_request
    def check_api_key():
        """Check API key if required"""
        if not APIConfig.REQUIRE_API_KEY:
            return
        
        # Skip for health check
        if request.path == '/health':
            return
        
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in APIConfig.API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401
    
    @app.before_request
    def check_rate_limit():
        """Check rate limit"""
        # Get client identifier
        client_id = request.headers.get('X-Forwarded-For', request.remote_addr)
        
        if not rate_limiter.is_allowed(client_id):
            return jsonify({
                'error': 'Rate limit exceeded',
                'retry_after': 60
            }), 429
    
    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(e):
        """Handle file too large"""
        max_mb = APIConfig.MAX_CONTENT_LENGTH / (1024 * 1024)
        return jsonify({
            'error': f'File too large. Maximum size: {max_mb}MB'
        }), 413
    
    # ========================================
    # Helper Functions
    # ========================================
    
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in APIConfig.ALLOWED_EXTENSIONS
    
    def get_client_id() -> str:
        """Get client identifier for rate limiting"""
        return request.headers.get('X-Forwarded-For', request.remote_addr)
    
    # ========================================
    # Routes: Info & Health
    # ========================================
    
    @app.route('/')
    def index():
        """API index page"""
        return render_template_string(INDEX_HTML)
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
        })
    
    @app.route('/api/info')
    def api_info():
        """API information"""
        return jsonify({
            'name': 'Lua Obfuscator API',
            'version': '1.0.0',
            'endpoints': {
                'obfuscate': '/api/obfuscate',
                'job_status': '/api/jobs/<job_id>',
                'download': '/api/download/<job_id>',
                'presets': '/api/presets',
            },
            'limits': {
                'max_file_size': APIConfig.MAX_CONTENT_LENGTH,
                'rate_limit': APIConfig.RATE_LIMIT_PER_MINUTE,
                'max_jobs': APIConfig.MAX_JOBS,
            }
        })
    
    # ========================================
    # Routes: Configuration
    # ========================================
    
    @app.route('/api/presets')
    def list_presets():
        """List available configuration presets"""
        presets = []
        for name in config_manager.list_presets():
            info = config_manager.get_preset_info(name)
            presets.append({
                'name': name,
                'description': info['description'],
            })
        
        return jsonify({'presets': presets})
    
    @app.route('/api/presets/<name>')
    def get_preset(name: str):
        """Get preset configuration"""
        try:
            config = config_manager.get_preset(name)
            config_dict = config_manager._config_to_dict(config)
            return jsonify(config_dict)
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
    
    # ========================================
    # Routes: Obfuscation
    # ========================================
    
    @app.route('/api/obfuscate', methods=['POST'])
    def obfuscate():
        """Obfuscate Lua bytecode"""
        # Check file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {APIConfig.ALLOWED_EXTENSIONS}'
            }), 400
        
        # Get configuration
        preset = request.form.get('preset', 'medium')
        
        try:
            config = config_manager.get_preset(preset)
        except ValueError:
            return jsonify({'error': f'Unknown preset: {preset}'}), 400
        
        # Custom config overrides
        if 'config' in request.form:
            try:
                custom_config = json.loads(request.form['config'])
                # Apply overrides (simplified)
                # In production, would need proper merging
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid config JSON'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        input_path = os.path.join(APIConfig.UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(input_path)
        
        # Validate file
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            parse_bytecode(data)
        except Exception as e:
            os.remove(input_path)
            return jsonify({'error': f'Invalid bytecode: {str(e)}'}), 400
        
        # Create job
        job = job_manager.create_job(input_path, preset)
        
        # Start processing in background
        output_path = os.path.join(APIConfig.OUTPUT_FOLDER, f"{job.id}_output.lua")
        
        def process_job():
            try:
                job_manager.update_job(job.id, status='processing', started_at=datetime.now())
                
                # Create pipeline
                progress_callback = APIProgressCallback(job_manager, job.id)
                pipeline = ObfuscationPipeline(config, progress_callback=progress_callback)
                
                # Process
                result = pipeline.process(input_path, output_path)
                
                # Update job
                job_manager.update_job(
                    job.id,
                    status='completed' if result.success else 'failed',
                    result=result,
                    output_file=output_path if result.success else None,
                    error=result.errors[0] if result.errors else None,
                    completed_at=datetime.now()
                )
                
            except Exception as e:
                job_manager.update_job(
                    job.id,
                    status='failed',
                    error=str(e),
                    completed_at=datetime.now()
                )
        
        # Start thread
        thread = threading.Thread(target=process_job)
        thread.daemon = True
        thread.start()
        
        # Return job ID
        return jsonify({
            'job_id': job.id,
            'status': 'pending',
            'message': 'Job created successfully'
        }), 202
    
    @app.route('/api/jobs/<job_id>')
    def get_job_status(job_id: str):
        """Get job status"""
        job = job_manager.get_job(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job.to_dict())
    
    @app.route('/api/download/<job_id>')
    def download_result(job_id: str):
        """Download obfuscated file"""
        job = job_manager.get_job(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        if job.status != 'completed':
            return jsonify({'error': 'Job not completed yet'}), 400
        
        if not job.output_file or not os.path.exists(job.output_file):
            return jsonify({'error': 'Output file not found'}), 404
        
        return send_file(
            job.output_file,
            as_attachment=True,
            download_name='obfuscated.lua',
            mimetype='text/plain'
        )
    
    # ========================================
    # Routes: Statistics
    # ========================================
    
    @app.route('/api/stats')
    def get_stats():
        """Get API statistics"""
        with job_manager.lock:
            total_jobs = len(job_manager.jobs)
            pending = sum(1 for j in job_manager.jobs.values() if j.status == 'pending')
            processing = sum(1 for j in job_manager.jobs.values() if j.status == 'processing')
            completed = sum(1 for j in job_manager.jobs.values() if j.status == 'completed')
            failed = sum(1 for j in job_manager.jobs.values() if j.status == 'failed')
        
        client_id = get_client_id()
        remaining = rate_limiter.get_remaining(client_id)
        
        return jsonify({
            'jobs': {
                'total': total_jobs,
                'pending': pending,
                'processing': processing,
                'completed': completed,
                'failed': failed,
            },
            'rate_limit': {
                'remaining': remaining,
                'limit': APIConfig.RATE_LIMIT_PER_MINUTE,
            }
        })
    
    return app

# ============================================
# HTML Template
# ============================================

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Lua Obfuscator API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .endpoint {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }
        .method {
            display: inline-block;
            padding: 4px 8px;
            background: #007bff;
            color: white;
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
        }
        .method.post { background: #28a745; }
        code {
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .info-box {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 Lua Obfuscator API</h1>
        
        <div class="info-box">
            <strong>Status:</strong> Online<br>
            <strong>Version:</strong> 1.0.0<br>
            <strong>Documentation:</strong> See endpoints below
        </div>
        
        <h2>📡 Endpoints</h2>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/health</strong>
            <p>Health check endpoint</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/api/info</strong>
            <p>Get API information and limits</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/api/presets</strong>
            <p>List available configuration presets</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/api/obfuscate</strong>
            <p>Upload and obfuscate Lua bytecode</p>
            <p><strong>Parameters:</strong></p>
            <ul>
                <li><code>file</code> - Bytecode file (.luac)</li>
                <li><code>preset</code> - Configuration preset (default: medium)</li>
            </ul>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/api/jobs/{job_id}</strong>
            <p>Get job status</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <strong>/api/download/{job_id}</strong>
            <p>Download obfuscated result</p>
        </div>
        
        <h2>💡 Example Usage</h2>
        
        <h3>Using cURL:</h3>
        <pre>
# Upload file for obfuscation
curl -X POST http://localhost:5000/api/obfuscate \\
  -F "file=@script.luac" \\
  -F "preset=high"

# Check job status
curl http://localhost:5000/api/jobs/{job_id}

# Download result
curl -O http://localhost:5000/api/download/{job_id}
        </pre>
        
        <h3>Using Python:</h3>
        <pre>
import requests

# Upload file
with open('script.luac', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/obfuscate',
        files={'file': f},
        data={'preset': 'high'}
    )

job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:5000/api/jobs/{job_id}')
print(status.json())

# Download result (when completed)
result = requests.get(f'http://localhost:5000/api/download/{job_id}')
with open('obfuscated.lua', 'wb') as f:
    f.write(result.content)
        </pre>
        
        <h2>⚙️ Configuration Presets</h2>
        <ul>
            <li><strong>minimal</strong> - Basic protection</li>
            <li><strong>low</strong> - Light obfuscation</li>
            <li><strong>medium</strong> - Balanced (default)</li>
            <li><strong>high</strong> - Strong protection</li>
            <li><strong>extreme</strong> - Maximum security</li>
            <li><strong>stealth</strong> - Anti-detection focus</li>
            <li><strong>performance</strong> - Speed optimized</li>
        </ul>
        
        <h2>📊 Rate Limits</h2>
        <p>Default: 10 requests per minute per IP address</p>
        
        <h2>🔑 API Keys</h2>
        <p>If enabled, include header: <code>X-API-Key: your-key-here</code></p>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
            <p>Lua Obfuscator API v1.0.0 | Built with Flask</p>
        </footer>
    </div>
</body>
</html>
"""

# ============================================
# Main Entry Point
# ============================================

if __name__ == '__main__':
    app = create_app()
    
    print("=" * 60)
    print("🚀 Lua Obfuscator API Server")
    print("=" * 60)
    print(f"Host: {APIConfig.HOST}")
    print(f"Port: {APIConfig.PORT}")
    print(f"Debug: {APIConfig.DEBUG}")
    print(f"Max file size: {APIConfig.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB")
    print(f"Rate limit: {APIConfig.RATE_LIMIT_PER_MINUTE} req/min")
    print("=" * 60)
    print("\nServer starting...")
    print(f"Open browser: http://localhost:{APIConfig.PORT}")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        debug=APIConfig.DEBUG
  )
