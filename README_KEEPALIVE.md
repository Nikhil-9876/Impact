# HuggingFace Keep-Alive Service 🚀

A standalone web application that automatically pings your HuggingFace space every 24 hours to keep it active and prevent it from going to sleep.

## Features

- ✅ **Automatic Pings**: Sends requests every 24 hours automatically
- 📊 **Dashboard**: Beautiful web interface to monitor ping status
- 📜 **History**: Track the last 50 pings with timestamps and status
- 🔘 **Manual Trigger**: Ping on-demand with a single button click
- 🔄 **Auto-refresh**: Dashboard updates every 30 seconds
- 🎨 **Modern UI**: Clean, responsive design with gradient background

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your HuggingFace Space URL

Set the environment variable with your HuggingFace space URL:

**Windows (PowerShell):**
```powershell
$env:HF_SPACE_URL = "https://your-username-your-space.hf.space"
```

**Linux/Mac:**
```bash
export HF_SPACE_URL="https://your-username-your-space.hf.space"
```

### 3. Run the Application

```bash
python keepalive_app.py
```

The web interface will be available at: `http://localhost:7860`

## Configuration Options

You can customize the behavior by setting these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_SPACE_URL` | Your HuggingFace space URL | `https://your-space.hf.space` |
| Port | Application port (in code) | `7860` |
| Interval | Ping interval in hours (in code) | `24` |

## Usage

### Web Dashboard

1. Open your browser and go to `http://localhost:7860`
2. View the current status and ping history
3. Click "Ping Now" to manually trigger a ping
4. Click "Refresh" to update the dashboard

### API Endpoints

- `GET /` - Web dashboard
- `GET /api/status` - Get current status and history (JSON)
- `POST /api/ping` - Manually trigger a ping
- `GET /health` - Health check endpoint

### Example API Response

```json
{
  "last_ping": {
    "success": true,
    "status_code": 200,
    "timestamp": "2026-02-05T10:30:00",
    "message": "Successfully pinged - Status 200"
  },
  "history": [...],
  "config": {
    "url": "https://your-space.hf.space",
    "interval_hours": 24
  }
}
```

## Deployment

### Deploy on Another HuggingFace Space

1. Create a new HuggingFace Space
2. Upload `keepalive_app.py` and `requirements.txt`
3. Set the `HF_SPACE_URL` secret in your Space settings
4. The app will start automatically and keep your main space alive!

### Deploy on Render/Railway/Other Services

1. Deploy this application to any cloud service
2. Set the `HF_SPACE_URL` environment variable
3. The service will keep running and ping your HuggingFace space every 24 hours

## How It Works

1. **Background Scheduler**: Uses APScheduler to run tasks in the background
2. **Periodic Pings**: Sends HTTP GET requests to your HuggingFace space every 24 hours
3. **History Tracking**: Stores the last 50 pings with status and timestamps
4. **Web Interface**: Provides real-time monitoring through a modern dashboard

## Troubleshooting

**Issue**: Pings are failing
- Verify your HuggingFace space URL is correct
- Check if your space is publicly accessible
- Review the error message in the ping history

**Issue**: Scheduler not working
- Check the logs for any errors
- Ensure the application is running continuously
- Verify the `/health` endpoint shows `scheduler_running: true`

## License

MIT License - Feel free to use and modify as needed!
