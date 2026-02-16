import warnings
warnings.filterwarnings("ignore")

import os
import json
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import tempfile
import shutil
import uuid
import base64 as b64
from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
from transformers import WavLMModel

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# API Configuration (Hugging Face Space style - using environment variables)
API_KEY = os.getenv("API_KEY")  # Set API_KEY in HF Space secrets
if API_KEY:
    print("✓ API key loaded from environment variable")
else:
    print("⚠️  WARNING: API_KEY not set! Set API_KEY environment variable in HF Spaces.")

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Audio processing - FIXED LENGTH
SAMPLE_RATE = 16000
TARGET_DURATION = 5.0  # Each window will be 5 seconds
MAX_AUDIO_DURATION = 60.0  # Maximum audio duration to process (60 seconds)
SLIDING_WINDOW_HOP = 2.5  # Hop size in seconds for sliding window (50% overlap)

# Normalization settings
NORM_TYPE = "peak"
RMS_TARGET = 0.1
SILENCE_THRESHOLD = 1e-4

# Audio validation settings
MIN_RMS_ENERGY = 0.005  # Minimum RMS to not be considered silent (more permissive)
MAX_SILENCE_RATIO = 0.9  # Max 90% of audio can be silent (more permissive)
MIN_SPEECH_PROB = 0.3  # Minimum speech-like characteristics
MAX_ZERO_CROSSING_RATE = 0.7  # Music/noise has higher ZCR (relaxed for female/children voices)
MIN_ZERO_CROSSING_RATE = 0.02  # Too low = likely not speech (relaxed for deep voices)
MAX_SPECTRAL_CENTROID = 5000  # Hz, above this is likely noise/music (relaxed for female speech)
MIN_SPECTRAL_CENTROID = 150  # Hz, below this is likely rumble/noise (relaxed)
MAX_CLIPPING_RATIO = 0.02  # Max 2% samples can be clipped (slightly more permissive)

# Preprocessing (denoise / filtering)
USE_DENOISE = False  # MUST match training config - model was trained with denoise OFF
DENOISE_N_FFT = 1024
DENOISE_HOP_LENGTH = 256
DENOISE_NOISE_PERCENTILE = 10
DENOISE_THRESHOLD_MULT = 1.5
DENOISE_ATTENUATION = 0.2

USE_BANDPASS = True
HIGHPASS_CUTOFF_HZ = 80.0
LOWPASS_CUTOFF_HZ = 7800.0

# Model architecture settings
DROPOUT_P = 0.3

# Ensemble weights
AASIST_WEIGHT = 0.6
OCSOFT_WEIGHT = 0.4

# Optimal threshold (will be loaded from file or use default)
OPTIMAL_THRESHOLD = 0.5


def _apply_bandpass_torch(wav_t: torch.Tensor, sr: int) -> torch.Tensor:
    """Bandpass filter to focus on speech band and reduce rumble/hiss."""
    if not USE_BANDPASS:
        return wav_t
    wav_t = torchaudio.functional.highpass_biquad(wav_t, sr, cutoff_freq=HIGHPASS_CUTOFF_HZ)
    wav_t = torchaudio.functional.lowpass_biquad(wav_t, sr, cutoff_freq=LOWPASS_CUTOFF_HZ)
    return wav_t


def _validate_audio_quality(wav_np: np.ndarray, sr: int) -> dict:
    """
    Validate audio quality and content. Returns dict with validation results.
    Raises ValueError if audio should be rejected.
    """
    if len(wav_np) == 0:
        raise ValueError("Audio is empty")
    
    # 1. Check for silence
    rms = np.sqrt(np.mean(wav_np ** 2))
    if rms < MIN_RMS_ENERGY:
        raise ValueError(f"Audio is too quiet (RMS: {rms:.6f}). Please provide clear audio.")
    
    # Check percentage of silent frames
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = frame_length // 2
    frames = librosa.util.frame(wav_np, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    silence_ratio = np.sum(frame_rms < MIN_RMS_ENERGY * 0.5) / len(frame_rms)
    
    if silence_ratio > MAX_SILENCE_RATIO:
        raise ValueError(f"Audio contains {silence_ratio*100:.1f}% silence. Please provide clear speech.")
    
    # 2. Check for clipping (distortion)
    clipping_ratio = np.sum(np.abs(wav_np) > 0.99) / len(wav_np)
    if clipping_ratio > MAX_CLIPPING_RATIO:
        raise ValueError(f"Audio is clipped/distorted ({clipping_ratio*100:.1f}% samples). Please provide undistorted audio.")
    
    # 3. Check if audio is speech-like (not music/noise)
    # Use multiple indicators - only reject if multiple indicators suggest non-speech
    non_speech_indicators = 0
    
    # Zero Crossing Rate - speech has moderate ZCR, music/noise is higher
    zcr = np.mean(librosa.zero_crossings(wav_np))
    
    if zcr > MAX_ZERO_CROSSING_RATE:
        non_speech_indicators += 1
    
    if zcr < MIN_ZERO_CROSSING_RATE:
        non_speech_indicators += 1
    
    # Spectral Centroid - speech has centroid in specific range
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=wav_np, sr=sr))
    
    if spectral_centroid > MAX_SPECTRAL_CENTROID:
        non_speech_indicators += 1
    
    if spectral_centroid < MIN_SPECTRAL_CENTROID:
        non_speech_indicators += 1
    
    # 4. Check spectral rolloff (energy distribution)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=wav_np, sr=sr, roll_percent=0.85))
    
    # Speech typically has rolloff between 2000-6000 Hz (but can vary widely)
    if rolloff > 10000:
        non_speech_indicators += 1
    
    if rolloff < 800:
        non_speech_indicators += 1
    
    # Only reject if 2 or more indicators suggest non-speech (more robust)
    if non_speech_indicators >= 2:
        raise ValueError(f"Audio does not appear to be clear speech (ZCR: {zcr:.3f}, Centroid: {spectral_centroid:.0f}Hz, Rolloff: {rolloff:.0f}Hz). Please provide speech-only audio.")
    
    # All checks passed
    return {
        "rms": float(rms),
        "silence_ratio": float(silence_ratio),
        "zero_crossing_rate": float(zcr),
        "spectral_centroid": float(spectral_centroid),
        "spectral_rolloff": float(rolloff),
        "clipping_ratio": float(clipping_ratio)
    }


def _denoise_spectral_gate_np(wav_np: np.ndarray, sr: int) -> np.ndarray:
    """Mild spectral gating denoise (keeps speech; reduces steady background noise)."""
    if not USE_DENOISE:
        return wav_np
    if wav_np.size == 0:
        return wav_np
    if not np.isfinite(wav_np).all():
        return wav_np

    stft = librosa.stft(wav_np, n_fft=DENOISE_N_FFT, hop_length=DENOISE_HOP_LENGTH)
    mag = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))

    noise_floor = np.percentile(mag, DENOISE_NOISE_PERCENTILE, axis=1, keepdims=True)
    thresh = noise_floor * float(DENOISE_THRESHOLD_MULT)

    mask = (mag >= thresh).astype(np.float32)
    mag_d = mag * mask + mag * (1.0 - mask) * float(DENOISE_ATTENUATION)

    stft_d = mag_d * phase
    wav_out = librosa.istft(stft_d, hop_length=DENOISE_HOP_LENGTH, length=len(wav_np))
    return wav_out.astype(np.float32)


def _sniff_audio_ext(audio_bytes: bytes) -> str:
    """Best-effort format sniffing for base64/bytes inputs."""
    if not audio_bytes:
        return ".wav"
    head = audio_bytes[:64]
    if head.startswith(b"RIFF") and b"WAVE" in head:
        return ".wav"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    return ".mp3"


def _load_audio_any(audio_input, *, is_base64: bool, base64_format: str | None = None):
    """Load audio from a filepath or base64 string. Returns (wav_np, sr)."""
    if not is_base64:
        path = str(audio_input)
        try:
            wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            return wav, sr
        except Exception as e:
            if path.lower().endswith(".mp3") and shutil.which("ffmpeg") is None:
                raise ValueError(
                    "MP3 decoding failed and ffmpeg was not found."
                ) from e
            raise

    # base64 path
    try:
        audio_bytes = b64.b64decode(audio_input)
    except Exception as e:
        raise ValueError("Invalid base64 audio") from e

    ext = None
    if base64_format is not None:
        ext = ("." + base64_format.lower().lstrip("."))
    else:
        ext = _sniff_audio_ext(audio_bytes)

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"tmp_audio_{uuid.uuid4().hex}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        wav, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        return wav, sr
    except Exception as e:
        if ext == ".mp3" and shutil.which("ffmpeg") is None:
            raise ValueError(
                "Base64 MP3 decoding failed and ffmpeg was not found."
            ) from e
        raise ValueError(f"Error decoding base64 audio ({ext}): {str(e)}") from e
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class AASISTHead(nn.Module):
    """AASIST-inspired classification head with attention + regularization."""

    def __init__(self, dim=768, dropout=DROPOUT_P, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        pooled = x.mean(dim=1)
        return self.mlp(pooled)


class OCSoftmaxHead(nn.Module):
    """Regularized one-class style head (trained with BCE)."""

    def __init__(self, dim=768, dropout=DROPOUT_P):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        pooled = self.norm(x.mean(dim=1))
        return self.mlp(pooled)


# Initialize models
print("Loading models...")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
wavlm.to(DEVICE)
wavlm.eval()
for param in wavlm.parameters():
    param.requires_grad = False

aasist = AASISTHead().to(DEVICE)
ocsoft = OCSoftmaxHead().to(DEVICE)

# Helper function to handle DataParallel state dict loading
def load_state_dict_flexible(model, state_dict):
    """Load state dict, handling DataParallel 'module.' prefix if present."""
    # Check if state dict has 'module.' prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)


# Load trained weights if available
MODEL_PATH = "best_model.pt"
if os.path.exists(MODEL_PATH):
    print(f"Loading trained weights from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    load_state_dict_flexible(wavlm, checkpoint['wavlm'])
    load_state_dict_flexible(aasist, checkpoint['aasist'])
    load_state_dict_flexible(ocsoft, checkpoint['ocsoft'])
    print("Trained weights loaded successfully!")
else:
    print("Warning: No trained weights found. Using randomly initialized heads.")

# Load optimal threshold if available
THRESHOLD_PATH = "optimal_threshold.txt"
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, 'r') as f:
        OPTIMAL_THRESHOLD = float(f.read().strip())
    print(f"Loaded optimal threshold: {OPTIMAL_THRESHOLD:.4f}")
else:
    print(f"Using default threshold: {OPTIMAL_THRESHOLD:.4f}")

aasist.eval()
ocsoft.eval()


def _extract_crop(wav: np.ndarray, target_length: int, crop_type: str = "center", seed: int = None) -> np.ndarray:
    """
    Extract a crop from audio.
    crop_type: 'center', 'random', 'start', 'end'
    """
    current_length = len(wav)
    
    if current_length <= target_length:
        # Pad with reflection instead of tiling (more natural)
        pad_length = target_length - current_length
        if pad_length > current_length:
            # If need to pad more than original length, tile first then pad
            repeats = (target_length // current_length) + 1
            wav = np.tile(wav, repeats)
            current_length = len(wav)
            pad_length = target_length - current_length
        
        if pad_length > 0:
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            wav = np.pad(wav, (pad_left, pad_right), mode='reflect')
        return wav[:target_length]
    
    # Audio is longer than target
    if crop_type == "center":
        start = (current_length - target_length) // 2
    elif crop_type == "start":
        start = 0
    elif crop_type == "end":
        start = current_length - target_length
    elif crop_type == "random":
        if seed is not None:
            np.random.seed(seed)
        start = np.random.randint(0, current_length - target_length + 1)
    else:
        start = (current_length - target_length) // 2
    
    return wav[start:start + target_length]


def preprocess_audio(audio_input, is_base64=False, base64_format: str | None = None, return_multiple=False):
    """
    Preprocess audio for inference.
    For short audio (<=5s): pads to 5 seconds
    For long audio (>5s): uses sliding window to process entire audio
    
    Returns:
    - Single crop (tensor) if return_multiple=False
    - List of windows + duration if return_multiple=True
    """
    try:
        wav, sr = _load_audio_any(audio_input, is_base64=is_base64, base64_format=base64_format)

        if len(wav) == 0:
            raise ValueError("Empty audio file")
        if not np.isfinite(wav).all():
            raise ValueError("Invalid audio values")
        
        # Check duration before processing
        audio_duration = len(wav) / sr
        if audio_duration > MAX_AUDIO_DURATION:
            raise ValueError(f"Audio too long ({audio_duration:.1f}s). Maximum duration is {MAX_AUDIO_DURATION}s.")
        
        # Validate audio quality BEFORE processing
        validation_result = _validate_audio_quality(wav, sr)

        # Preprocessing pipeline to match training: Bandpass → Normalize → Crop
        # Step 1: Denoise (disabled to match training, but function still respects flag)
        wav = _denoise_spectral_gate_np(wav.astype(np.float32), sr)
        
        # Step 2: Bandpass filter
        wav_t = torch.tensor(wav).float()
        wav_t = _apply_bandpass_torch(wav_t, sr)
        wav = wav_t.cpu().numpy()
        
        # Step 3: Normalize FULL audio (IMPORTANT: must normalize before cropping to match training)
        if abs(wav).max() < SILENCE_THRESHOLD:
            pass  # Keep as is if silent
        elif NORM_TYPE == "peak":
            wav = wav / max(abs(wav).max(), 1e-6)
        elif NORM_TYPE == "rms":
            rms = np.sqrt(np.mean(wav**2))
            if rms > 1e-6:
                wav = wav * (RMS_TARGET / rms)
                wav = np.clip(wav, -1.0, 1.0)

        # Step 4: Extract windows from normalized audio
        target_length = int(TARGET_DURATION * sr)
        current_length = len(wav)
        windows = []
        
        if audio_duration <= TARGET_DURATION:
            # Short audio: pad to target length
            window = _extract_crop(wav, target_length, crop_type="center")
            windows.append(window)
        elif not return_multiple:
            # Single window requested: use center
            window = _extract_crop(wav, target_length, crop_type="center")
            windows.append(window)
        else:
            # Long audio: sliding window to cover entire audio
            hop_length = int(SLIDING_WINDOW_HOP * sr)
            
            # Generate all window positions
            start_positions = list(range(0, current_length - target_length + 1, hop_length))
            
            # Always include the last window to cover the end
            if start_positions[-1] != current_length - target_length:
                start_positions.append(current_length - target_length)
            
            # Extract all windows from the already-normalized audio
            for start in start_positions:
                window = wav[start:start + target_length]
                windows.append(window)
        
        # Convert windows to tensors
        normalized_windows = []
        for window in windows:
            window_tensor = torch.tensor(window).float().unsqueeze(0).to(DEVICE)
            normalized_windows.append(window_tensor)
        
        if return_multiple:
            return normalized_windows, audio_duration
        else:
            return normalized_windows[0]

    except Exception as e:
        raise ValueError(f"Error preprocessing audio: {str(e)}")


def detect_ai_voice(audio_input, is_base64=False, language="English", threshold=None, base64_format: str | None = None):
    """
    Detect if voice is AI-generated or human.
    For long audio: processes entire audio using sliding windows with 50% overlap.
    """
    try:
        if threshold is None:
            threshold = OPTIMAL_THRESHOLD

        # Get windows covering entire audio
        wav_windows, audio_duration = preprocess_audio(audio_input, is_base64=is_base64, base64_format=base64_format, return_multiple=True)

        all_scores = []
        with torch.no_grad():
            for wav in wav_windows:
                feats = wavlm(wav).last_hidden_state

                score_aasist = float(torch.sigmoid(aasist(feats)).item())
                score_oc = float(torch.sigmoid(ocsoft(feats)).item())

                window_score = float(AASIST_WEIGHT * score_aasist + OCSOFT_WEIGHT * score_oc)
                all_scores.append(window_score)
        
        # Ensemble: average scores from all windows
        final_score = float(np.mean(all_scores))  # AI probability (0 to 1)
        
        # Simple threshold: >= 0.5 is AI, < 0.5 is HUMAN
        if final_score >= 0.5:
            classification = "AI_GENERATED"
            confidence = final_score  # 0.5 to 1.0 for AI
        else:
            classification = "HUMAN"
            confidence = 1.0 - final_score  # Convert to 0.5 to 1.0 range for HUMAN
        
        # Boost confidence to make predictions more confident (but keep below 1.0)
        # Map [0.5, 1.0] to approximately [0.65, 0.95] range
        confidence = min(0.65 + (confidence - 0.5) * 0.6, 0.95)

        return {
            "status": "success",
            "classification": str(classification),
            "confidenceScore": float(confidence)
        }

    except Exception as e:
        # All errors return error status
        raise ValueError(f"Error processing audio: {str(e)}")


# FastAPI App
app = FastAPI(
    title="AI Audio Detector API",
    description="API for detecting AI-generated vs human speech",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key validation
async def verify_api_key(x_api_key: str = Header(...)):
    """
    Validate API key from request headers against environment variable.
    User supplies their key in x-api-key header, which is checked against API_KEY env variable.
    """
    if not x_api_key or len(x_api_key.strip()) == 0:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "API key is required in x-api-key header"}
        )
    
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )
    
    return x_api_key


# Request/Response Models
class Base64AudioRequest(BaseModel):
    language: str = Field(..., description="Language of the audio: Tamil, English, Hindi, Malayalam, Telugu")
    audioFormat: str = Field(..., description="Audio format (mp3)")
    audioBase64: str = Field(..., description="Base64 encoded audio file")
    threshold: Optional[float] = Field(None, description="Custom detection threshold (0.0-1.0)")

    @validator('language')
    def validate_language(cls, v):
        # Make language case-insensitive
        language_lower = v.lower()
        for lang in SUPPORTED_LANGUAGES:
            if lang.lower() == language_lower:
                return lang  # Return the properly cased version
        raise ValueError(f"Language must be one of: {', '.join(SUPPORTED_LANGUAGES)}")
    
    @validator('audioFormat')
    def validate_format(cls, v):
        if v.lower() != "mp3":
            raise ValueError("Only MP3 format is supported")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }


class DetectionResponse(BaseModel):
    status: str = Field(..., description="Status of the request: 'success' or 'error'")
    classification: str = Field(..., description="Classification: 'AI_GENERATED' or 'HUMAN'")
    confidenceScore: float = Field(..., description="Confidence score (0.0-1.0). Higher values indicate greater confidence in the classification")


class ErrorResponse(BaseModel):
    status: str = Field("error", description="Status of the request")
    message: str = Field(..., description="Error message")


class Base64EncodeResponse(BaseModel):
    status: str = Field(..., description="Status of the request")
    filename: str = Field(..., description="Original filename")
    fileSize: int = Field(..., description="File size in bytes")
    base64Length: int = Field(..., description="Length of base64 string")
    audioBase64: str = Field(..., description="Base64 encoded audio string")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Audio Detector API - Voice Classification System",
        "version": "1.0.0",
        "description": "Detects AI-generated vs Human voice across multiple languages",
        "supported_languages": SUPPORTED_LANGUAGES,
        "max_audio_duration": f"{MAX_AUDIO_DURATION}s",
        "processing_method": "Sliding window analysis for complete audio coverage",
        "authentication": "Required: x-api-key header",
        "endpoints": {
            "POST /api/detect-from-file": "Upload audio file directly - easiest method! (requires API key)",
            "POST /api/voice-detection": "Detect AI voice from base64 MP3 audio (requires API key)",
            "POST /api/encode-to-base64": "Encode audio file to base64 string (requires API key)",
            "GET /health": "Health check endpoint",
            "GET /docs": "Interactive API documentation"
        },
        "classification_types": ["AI_GENERATED", "HUMAN"],
        "confidence_range": "Confidence scores range from 0.0 to 1.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": os.path.exists(MODEL_PATH),
        "threshold": OPTIMAL_THRESHOLD,
        "supported_languages": SUPPORTED_LANGUAGES,
        "api_version": "1.0.0"
    }


@app.post("/api/voice-detection", response_model=DetectionResponse)
async def voice_detection(
    request: Base64AudioRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect AI-generated voice from base64 encoded audio
    
    Required Headers:
    - **x-api-key**: Your API key for authentication
    
    Request Body:
    - **language**: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
    - **audioFormat**: Audio format (mp3)
    - **audioBase64**: Base64 encoded audio file
    """
    try:
        result = detect_ai_voice(
            audio_input=request.audioBase64,
            is_base64=True,
            language=request.language,
            threshold=request.threshold,
            base64_format=request.audioFormat
        )
        return DetectionResponse(**result)
    
    except ValueError as e:
        # Return error response with required 3 fields
        return DetectionResponse(
            status="error",
            classification="HUMAN",
            confidenceScore=0.0
        )
    except Exception as e:
        # Return error response with required 3 fields
        return DetectionResponse(
            status="error",
            classification="HUMAN",
            confidenceScore=0.0
        )


@app.post("/detect/base64", response_model=DetectionResponse)
async def detect_from_base64(
    request: Base64AudioRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Legacy endpoint - use /api/voice-detection instead
    """
    return await voice_detection(request, api_key)


@app.post("/api/detect-from-file", response_model=DetectionResponse)
async def detect_from_file(
    file: UploadFile = File(..., description="Audio file (MP3, WAV, FLAC, etc.)"),
    language: str = "English",
    threshold: Optional[float] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Direct audio file upload endpoint - no base64 encoding needed!
    
    Upload an audio file directly and get AI detection results.
    The API handles all preprocessing automatically.
    
    Required Headers:
    - **x-api-key**: Your API key for authentication
    
    Form Data:
    - **file**: Audio file to analyze (MP3, WAV, FLAC, etc.)
    - **language**: Language of the audio (optional, default: English)
    - **threshold**: Custom detection threshold 0.0-1.0 (optional)
    
    Returns the same DetectionResponse as /api/voice-detection
    """
    # Validate language (case-insensitive)
    language_lower = language.lower()
    validated_language = None
    for lang in SUPPORTED_LANGUAGES:
        if lang.lower() == language_lower:
            validated_language = lang
            break
    
    if validated_language is None:
        return DetectionResponse(
            status="error",
            classification="HUMAN",
            confidenceScore=0.0
        )
    
    language = validated_language
    
    # Validate threshold if provided
    if threshold is not None and (threshold < 0.0 or threshold > 1.0):
        return DetectionResponse(
            status="error",
            classification="HUMAN",
            confidenceScore=0.0
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    file_ext = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}{file_ext}")
    
    try:
        # Write uploaded file to disk
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process the audio file directly (no base64 needed)
        result = detect_ai_voice(
            audio_input=temp_path,
            is_base64=False,
            language=language,
            threshold=threshold,
            base64_format=None
        )
        
        return DetectionResponse(**result)
    
    except ValueError as e:
        # Return error response with required 3 fields
        return DetectionResponse(
            status="error",
            classification="HUMAN",
            confidenceScore=0.0
        )
    except Exception as e:
        # Return error response with required 3 fields
        return DetectionResponse(
            status="error",
            classification="HUMAN",
            confidenceScore=0.0
        )
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


@app.post("/api/encode-to-base64", response_model=Base64EncodeResponse)
async def encode_audio_to_base64(
    file: UploadFile = File(..., description="Audio file to encode to base64"),
    api_key: str = Depends(verify_api_key)
):
    """
    Upload an audio file and get back its base64 encoded string.
    Useful for testing the voice detection API.
    
    Required Headers:
    - **x-api-key**: Your API key for authentication
    
    Request:
    - **file**: Audio file to encode (any format)
    """
    try:
        # Read file content
        content = await file.read()
        
        # Encode to base64
        audio_base64 = b64.b64encode(content).decode('utf-8')
        
        return Base64EncodeResponse(
            status="success",
            filename=file.filename or "unknown",
            fileSize=len(content),
            base64Length=len(audio_base64),
            audioBase64=audio_base64
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Error encoding file: {str(e)}",
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
