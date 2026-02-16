# AI Audio Detector - FastAPI

FastAPI-based REST API for detecting AI-generated vs human speech across multiple Indian languages.

## Features

- 🚀 Fast and efficient inference with WavLM + Ensemble Heads
- 🎯 Sliding window analysis for complete audio coverage (up to 60 seconds)
- 🔒 API key authentication for secure access
- 🎤 Supports multiple languages: Tamil, English, Hindi, Malayalam, Telugu
- 📊 Audio quality validation before processing
- 🌐 CORS enabled for cross-origin requests
- 📚 Auto-generated API documentation (Swagger UI)
- 🐳 Docker support for easy deployment

## Model Architecture

- **Backbone**: Microsoft WavLM-base (frozen)
- **Classification Heads**: AASIST + OC-Softmax ensemble
- **Input**: 5-second audio windows @ 16kHz
- **Processing**: Bandpass filtering → Peak normalization → Sliding window analysis

## Installation

### Option 1: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key as environment variable
export API_KEY="your-secret-api-key"

# Run the API
python api.py
```

The API will be available at `http://localhost:8000`

### Option 2: Using uvicorn directly

```bash
pip install -r requirements.txt
export API_KEY="your-secret-api-key"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Docker

```bash
# Build the Docker image
docker build -t ai-audio-detector-api .

# Run the container
docker run -p 7860:7860 -e API_KEY="your-secret-api-key" ai-audio-detector-api
```

## API Endpoints

### 1. Root Endpoint
```bash
GET /
```

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "AI Audio Detector API - Voice Classification System",
  "version": "1.0.0",
  "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
  "max_audio_duration": "60s",
  "authentication": "Required: x-api-key header"
}
```

### 2. Health Check
```bash
GET /health
```

Returns API status and model information.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "threshold": 0.5,
  "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
  "api_version": "1.0.0"
}
```

### 3. Direct File Upload (Recommended - Easiest!)
```bash
POST /api/detect-from-file
Headers: x-api-key: your-secret-api-key
```

**This is the easiest way to use the API!** Just upload your audio file directly - no base64 encoding needed.

**Request Headers:**
- `x-api-key`: Your API key (required)

**Form Data:**
- `file`: Audio file (MP3, WAV, FLAC, etc.) - required
- `language`: Language (Tamil, English, Hindi, Malayalam, Telugu) - optional, defaults to "English"
- `threshold`: Custom detection threshold (0.0-1.0) - optional, defaults to 0.5

**Response:**
Same as voice detection endpoint below (DetectionResponse)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/detect-from-file" \
  -H "x-api-key: your-secret-api-key" \
  -F "file=@audio.mp3" \
  -F "language=Tamil" \
  -F "threshold=0.5"
```

### 4. Voice Detection (Base64 Method)
```bash
POST /api/voice-detection
Headers: x-api-key: your-secret-api-key
```

Detect AI voice from base64 encoded MP3 audio. Use this if you need to send audio as JSON.

**Request Headers:**
- `x-api-key`: Your API key (required)

**Request Body:**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA...",
  "threshold": 0.5
}
```

**Response (Success - AI Generated):**
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.8523,
  "explanation": "High confidence AI detection. The audio shows clear temporal and spectral patterns typical of AI-generated speech, with synthetic characteristics in voice quality and delivery.",
  "audioDuration": 8.5,
  "segmentsAnalyzed": 4
}
```

**Response (Success - Human):**
```json
{
  "status": "success",
  "language": "English",
  "classification": "HUMAN",
  "confidenceScore": 0.7821,
  "explanation": "Moderate-high confidence human speech. The audio shows clear natural speech patterns with authentic human voice characteristics and spontaneous variations.",
  "audioDuration": 5.2,
  "segmentsAnalyzed": 2
}
```

**Response (Success - Uncertain/Grey Zone):**
```json
{
  "status": "success",
  "language": "Hindi",
  "classification": "UNCERTAIN",
  "confidenceScore": 0.08,
  "explanation": "Uncertain classification (slightly AI-leaning). The audio exhibits mixed characteristics that fall within the grey zone between clear AI-generated and clear human speech. The model cannot confidently classify this audio. This may occur with: (1) high-quality AI voices that closely mimic human speech, (2) heavily processed human recordings, or (3) audio with ambiguous or degraded quality. Consider additional verification or context for accurate determination.",
  "audioDuration": 6.3,
  "segmentsAnalyzed": 3
}
```

**Response (Rejected - Poor Quality):**
```json
{
  "status": "rejected",
  "language": "Tamil",
  "classification": "REJECTED_NON_SPEECH",
  "confidenceScore": 0.0,
  "explanation": "Audio validation failed: Content appears to be non-speech (music, noise, or unclear audio). Details: Audio does not appear to be clear speech (ZCR: 0.821, Centroid: 5234Hz, Rolloff: 9821Hz).",
  "validationError": "Audio does not appear to be clear speech..."
}
```

### 5. Encode Audio to Base64 (Helper Endpoint)
```bash
POST /api/encode-to-base64
Headers: x-api-key: your-secret-api-key
```

Upload an audio file and get its base64 encoded string for testing.

**Request:**
- Form data with `file` field containing the audio file

**Response:**
```json
{
  "status": "success",
  "filename": "audio.mp3",
  "fileSize": 125437,
  "base64Length": 167249,
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

## Classification Types

### Success Classifications
- **AI_GENERATED**: Audio is detected as AI-generated speech (AI probability ≥ 0.65)
  - Confidence score 0.5-1.0: based on strength of AI signal
  - Higher confidence = stronger AI characteristics
- **HUMAN**: Audio is detected as human speech (AI probability < 0.5)
  - Confidence score 0.5-1.0: based on strength of human signal
  - Higher confidence = stronger human characteristics
- **UNCERTAIN**: Audio falls in the grey zone with mixed characteristics (AI probability 0.5-0.65)
  - Confidence score 0.5-1.0: based on ambiguity level
  - Higher confidence = more ambiguous (closer to 0.575, the center)
  - Lower confidence = closer to classification boundaries
  - This indicates the model cannot confidently classify the audio
  - May occur with high-quality AI voices, heavily processed human recordings, or ambiguous audio
  - Consider additional verification or context for accurate determination

### Rejection Classifications
- **REJECTED_SILENT**: Audio is too quiet or contains too much silence
- **REJECTED_NON_SPEECH**: Audio appears to be music, noise, or non-speech
- **REJECTED_POOR_QUALITY**: Audio quality is too poor (clipped/distorted)
- **REJECTED_INVALID**: Audio is invalid or corrupted

### Confidence Score Range
All confidence scores range from **0.5 to 1.0** across all classification types:
- **0.5**: Minimum confidence (at classification boundary)
- **1.0**: Maximum confidence (strongest signal or most ambiguous for UNCERTAIN)

### Grey Zone Explanation
The API implements a grey zone (AI probability 0.5-0.65) where the model returns **UNCERTAIN** instead of forcing a classification. This provides more honest and reliable results by acknowledging when the audio exhibits ambiguous characteristics.

## Usage Examples

### Python with requests (Direct File Upload - Easiest!)

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-secret-api-key"
headers = {"x-api-key": API_KEY}

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Direct file upload (RECOMMENDED - no base64 needed!)
with open("audio.mp3", "rb") as audio_file:
    files = {"file": audio_file}
    data = {
        "language": "Tamil",
        "threshold": "0.5"  # optional
    }
    response = requests.post(
        f"{API_URL}/api/detect-from-file",
        files=files,
        data=data,
        headers=headers
    )

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidenceScore']:.4f}")
print(f"Explanation: {result['explanation']}")
```

### Python with requests (Base64 Method)

```python
import requests
import base64

API_URL = "http://localhost:8000"
API_KEY = "your-secret-api-key"
headers = {"x-api-key": API_KEY}

# Encode audio to base64
with open("audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Voice detection with base64
payload = {
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": audio_base64,
    "threshold": 0.5  # optional
}
response = requests.post(
    f"{API_URL}/api/voice-detection",
    json=payload,
    headers=headers
)
result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidenceScore']:.4f}")
print(f"Explanation: {result['explanation']}")
```

### cURL (Direct File Upload)

```bash
# Health check
curl http://localhost:8000/health

# Direct file upload (EASIEST METHOD!)
curl -X POST "http://localhost:8000/api/detect-from-file" \
  -H "x-api-key: your-secret-api-key" \
  -F "file=@audio.mp3" \
  -F "language=Tamil"

# With custom threshold
curl -X POST "http://localhost:8000/api/detect-from-file" \
  -H "x-api-key: your-secret-api-key" \
  -F "file=@audio.mp3" \
  -F "language=Tamil" \
  -F "threshold=0.6"
```

### cURL (Base64 Method)

```bash
# Voice detection with base64
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "'$(base64 -w 0 audio.mp3)'"
  }'
```

### JavaScript/Fetch (Direct File Upload)

```javascript
const API_KEY = "your-secret-api-key";
const headers = { "x-api-key": API_KEY };

// Direct file upload (EASIEST METHOD!)
async function detectVoiceFromFile(audioFile, language = "Tamil") {
  const formData = new FormData();
  formData.append("file", audioFile);
  formData.append("language", language);
  
  const response = await fetch("http://localhost:8000/api/detect-from-file", {
    method: "POST",
    headers: headers,
    body: formData
  });
  
  const result = await response.json();
  console.log("Classification:", result.classification);
  console.log("Confidence:", result.confidenceScore);
  console.log("Explanation:", result.explanation);
  return result;
}

// Use with file input
document.getElementById("audioInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  detectVoiceFromFile(file, "Tamil");
});
```

### JavaScript/Fetch (Base64 Method)

```javascript
const API_KEY = "your-secret-api-key";
const headers = { 
  "x-api-key": API_KEY,
  "Content-Type": "application/json"
};

// Convert file to base64
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Voice detection with base64
async function detectVoiceBase64(audioFile) {
  const audioBase64 = await fileToBase64(audioFile);
  
  const response = await fetch("http://localhost:8000/api/voice-detection", {
    method: "POST",
    headers: headers,
    body: JSON.stringify({
      language: "Tamil",
      audioFormat: "mp3",
      audioBase64: audioBase64
    })
  });
  
  const result = await response.json();
  console.log("Classification:", result.classification);
  console.log("Confidence:", result.confidenceScore);
  return result;
}
```

## Audio Requirements

- **Format**: MP3 (base64 encoded)
- **Duration**: Up to 60 seconds
- **Quality**: Clear speech without excessive noise, music, or distortion
- **Languages**: Tamil, English, Hindi, Malayalam, Telugu

## Model Files

Place your trained model files in the same directory as `api.py`:
- `best_model.pt` - Trained model checkpoint (required)
- `optimal_threshold.txt` - Optimal detection threshold (optional, defaults to 0.5)

If `best_model.pt` is not found, the API will use randomly initialized heads (not recommended for production).

## Configuration

Key configuration parameters in `api.py`:

```python
# Audio processing
SAMPLE_RATE = 16000
TARGET_DURATION = 5.0  # Fixed 5-second windows
MAX_AUDIO_DURATION = 60.0  # Maximum input duration
SLIDING_WINDOW_HOP = 2.5  # 50% overlap

# Preprocessing (MUST match training config)
USE_DENOISE = False  # Model trained without denoising
USE_BANDPASS = True
NORM_TYPE = "peak"

# Model ensemble
AASIST_WEIGHT = 0.6
OCSOFT_WEIGHT = 0.4

# Detection threshold
OPTIMAL_THRESHOLD = 0.5  # Adjust based on your use case
```

## Authentication

The API requires authentication via the `x-api-key` header. Set your API key as an environment variable:

```bash
# Linux/Mac
export API_KEY="your-secret-api-key"

# Windows (PowerShell)
$env:API_KEY="your-secret-api-key"

# Docker
docker run -e API_KEY="your-secret-api-key" ...
```

**Security Note**: In production, use strong, randomly generated API keys and consider implementing rate limiting.

## Testing

Visit the interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Deployment

### Hugging Face Spaces

1. Create a new Space with Docker SDK
2. Upload files: `api.py`, `requirements.txt`, `Dockerfile`
3. Add model files: `best_model.pt`, `optimal_threshold.txt`
4. Set `API_KEY` in Space secrets
5. The API will start automatically on port 7860

### Production with Gunicorn

```bash
pip install gunicorn
export API_KEY="your-secret-api-key"
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### "Import could not be resolved" errors
Install dependencies: `pip install -r requirements.txt`

### "API key is required" error
Set the `API_KEY` environment variable or pass it in the `x-api-key` header

### "Invalid API key" error
Ensure your `x-api-key` header matches the `API_KEY` environment variable

### MP3 decoding fails
Install ffmpeg: `apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)

### Model not loaded
Ensure `best_model.pt` is in the same directory as `api.py`

## Performance

- **Short audio (≤5s)**: ~0.2-0.5 seconds per request (GPU)
- **Long audio (60s)**: ~2-3 seconds per request (GPU with 12 windows)
- **Throughput**: ~10-20 requests/sec on T4 GPU

## License

MIT License
