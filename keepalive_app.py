"""
Hugging Face Space Keep-Alive Web App
Sends periodic requests every 24 hours to keep your HF space active
"""

import os
import requests
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HUGGINGFACE_SPACE_URL = os.getenv("HF_SPACE_URL", "https://your-space.hf.space")
PING_INTERVAL_HOURS = 24

# Store ping history
ping_history = []
last_ping_time = None
last_ping_status = None

app = FastAPI(
    title="HuggingFace Keep-Alive Service",
    description="Keeps your HuggingFace space active by sending periodic requests",
    version="1.0.0"
)

def ping_huggingface_space():
    """Send a request to the HuggingFace space to keep it alive"""
    global last_ping_time, last_ping_status
    
    try:
        logger.info(f"Pinging HuggingFace space: {HUGGINGFACE_SPACE_URL}")
        response = requests.get(HUGGINGFACE_SPACE_URL, timeout=30)
        
        last_ping_time = datetime.now()
        last_ping_status = {
            "success": True,
            "status_code": response.status_code,
            "timestamp": last_ping_time.isoformat(),
            "message": f"Successfully pinged - Status {response.status_code}"
        }
        
        ping_history.insert(0, last_ping_status)
        if len(ping_history) > 50:  # Keep last 50 pings
            ping_history.pop()
        
        logger.info(f"✓ Ping successful - Status: {response.status_code}")
        return last_ping_status
        
    except Exception as e:
        logger.error(f"✗ Ping failed: {str(e)}")
        last_ping_time = datetime.now()
        last_ping_status = {
            "success": False,
            "status_code": None,
            "timestamp": last_ping_time.isoformat(),
            "message": f"Ping failed: {str(e)}"
        }
        
        ping_history.insert(0, last_ping_status)
        if len(ping_history) > 50:
            ping_history.pop()
        
        return last_ping_status

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Schedule the ping job every 24 hours
scheduler.add_job(
    func=ping_huggingface_space,
    trigger=IntervalTrigger(hours=PING_INTERVAL_HOURS),
    id='ping_job',
    name='Ping HuggingFace Space',
    replace_existing=True
)

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

logger.info(f"✓ Scheduler started - Will ping every {PING_INTERVAL_HOURS} hours")
logger.info(f"✓ Target URL: {HUGGINGFACE_SPACE_URL}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard page"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HuggingFace Keep-Alive Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 900px;
                margin: 0 auto;
            }}
            
            .header {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            
            h1 {{
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2em;
            }}
            
            .subtitle {{
                color: #666;
                font-size: 1.1em;
            }}
            
            .config-box {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            
            .config-item {{
                margin-bottom: 15px;
            }}
            
            .config-label {{
                font-weight: 600;
                color: #555;
                margin-bottom: 5px;
            }}
            
            .config-value {{
                background: #f5f5f5;
                padding: 10px 15px;
                border-radius: 8px;
                font-family: monospace;
                word-break: break-all;
            }}
            
            .status-box {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            
            .status-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            
            h2 {{
                color: #333;
                font-size: 1.5em;
            }}
            
            .btn {{
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                transition: all 0.3s;
            }}
            
            .btn-primary {{
                background: #667eea;
                color: white;
            }}
            
            .btn-primary:hover {{
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }}
            
            .btn-secondary {{
                background: #48bb78;
                color: white;
            }}
            
            .btn-secondary:hover {{
                background: #38a169;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(72, 187, 120, 0.4);
            }}
            
            .status-info {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            
            .status-row {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #e0e0e0;
            }}
            
            .status-row:last-child {{
                border-bottom: none;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
            }}
            
            .badge-success {{
                background: #c6f6d5;
                color: #22543d;
            }}
            
            .badge-error {{
                background: #fed7d7;
                color: #742a2a;
            }}
            
            .badge-warning {{
                background: #feebc8;
                color: #7c2d12;
            }}
            
            .history-box {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            
            .history-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .history-time {{
                font-size: 0.9em;
                color: #666;
            }}
            
            .loading {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .button-group {{
                display: flex;
                gap: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 HuggingFace Keep-Alive Dashboard</h1>
                <p class="subtitle">Automatically ping your HuggingFace space every 24 hours to keep it active</p>
            </div>
            
            <div class="config-box">
                <h2 style="margin-bottom: 20px;">⚙️ Configuration</h2>
                <div class="config-item">
                    <div class="config-label">Target URL:</div>
                    <div class="config-value" id="targetUrl">{HUGGINGFACE_SPACE_URL}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Ping Interval:</div>
                    <div class="config-value">Every {PING_INTERVAL_HOURS} hours</div>
                </div>
            </div>
            
            <div class="status-box">
                <div class="status-header">
                    <h2>📊 Current Status</h2>
                    <div class="button-group">
                        <button class="btn btn-secondary" onclick="pingNow()">Ping Now</button>
                        <button class="btn btn-primary" onclick="refreshStatus()">Refresh</button>
                    </div>
                </div>
                
                <div class="status-info" id="statusInfo">
                    <div class="status-row">
                        <strong>Last Ping:</strong>
                        <span id="lastPingTime">Loading...</span>
                    </div>
                    <div class="status-row">
                        <strong>Status:</strong>
                        <span id="lastPingStatus">Loading...</span>
                    </div>
                    <div class="status-row">
                        <strong>Message:</strong>
                        <span id="lastPingMessage">Loading...</span>
                    </div>
                </div>
            </div>
            
            <div class="history-box">
                <h2 style="margin-bottom: 20px;">📜 Ping History</h2>
                <div id="historyList">
                    <p style="color: #666;">Loading history...</p>
                </div>
            </div>
        </div>
        
        <script>
            async function refreshStatus() {{
                try {{
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    const lastPingTime = document.getElementById('lastPingTime');
                    const lastPingStatus = document.getElementById('lastPingStatus');
                    const lastPingMessage = document.getElementById('lastPingMessage');
                    
                    if (data.last_ping) {{
                        const date = new Date(data.last_ping.timestamp);
                        lastPingTime.textContent = date.toLocaleString();
                        
                        if (data.last_ping.success) {{
                            lastPingStatus.innerHTML = '<span class="status-badge badge-success">✓ Success</span>';
                        }} else {{
                            lastPingStatus.innerHTML = '<span class="status-badge badge-error">✗ Failed</span>';
                        }}
                        
                        lastPingMessage.textContent = data.last_ping.message;
                    }} else {{
                        lastPingTime.textContent = 'No pings yet';
                        lastPingStatus.innerHTML = '<span class="status-badge badge-warning">⏳ Waiting</span>';
                        lastPingMessage.textContent = 'Waiting for first scheduled ping';
                    }}
                    
                    // Update history
                    const historyList = document.getElementById('historyList');
                    if (data.history && data.history.length > 0) {{
                        historyList.innerHTML = data.history.map(ping => {{
                            const date = new Date(ping.timestamp);
                            const badge = ping.success ? 
                                '<span class="status-badge badge-success">✓</span>' : 
                                '<span class="status-badge badge-error">✗</span>';
                            return `
                                <div class="history-item">
                                    <div>
                                        ${{badge}}
                                        <span class="history-time">${{date.toLocaleString()}}</span>
                                    </div>
                                    <div>${{ping.message}}</div>
                                </div>
                            `;
                        }}).join('');
                    }} else {{
                        historyList.innerHTML = '<p style="color: #666;">No ping history yet</p>';
                    }}
                }} catch (error) {{
                    console.error('Error refreshing status:', error);
                }}
            }}
            
            async function pingNow() {{
                const button = event.target;
                button.disabled = true;
                button.innerHTML = '<span class="loading"></span> Pinging...';
                
                try {{
                    const response = await fetch('/api/ping', {{ method: 'POST' }});
                    const data = await response.json();
                    
                    if (data.success) {{
                        alert('✓ Ping successful!');
                    }} else {{
                        alert('✗ Ping failed: ' + data.message);
                    }}
                    
                    await refreshStatus();
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }} finally {{
                    button.disabled = false;
                    button.textContent = 'Ping Now';
                }}
            }}
            
            // Auto-refresh every 30 seconds
            setInterval(refreshStatus, 30000);
            
            // Initial load
            refreshStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def get_status():
    """Get current status and ping history"""
    return {
        "last_ping": last_ping_status,
        "history": ping_history[:10],  # Return last 10 pings
        "config": {
            "url": HUGGINGFACE_SPACE_URL,
            "interval_hours": PING_INTERVAL_HOURS
        }
    }

@app.post("/api/ping")
async def manual_ping():
    """Manually trigger a ping"""
    result = ping_huggingface_space()
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "scheduler_running": scheduler.running,
        "jobs": len(scheduler.get_jobs())
    }

if __name__ == "__main__":
    import uvicorn
    
    # Perform initial ping on startup
    logger.info("Performing initial ping...")
    ping_huggingface_space()
    
    uvicorn.run(app, host="0.0.0.0", port=7860)
