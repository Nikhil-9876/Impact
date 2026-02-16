"""
Simple test script for the direct file upload endpoint
This is the easiest way to use the AI Audio Detector API!
"""

import requests
import sys
import os

# Configuration
API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "your-secret-api-key")  # Set your API key here or as environment variable

def test_file_upload(audio_file_path, language="English"):
    """
    Test the direct file upload endpoint
    
    Args:
        audio_file_path: Path to audio file (MP3, WAV, FLAC, etc.)
        language: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
    """
    print(f"\n{'='*60}")
    print(f"Testing AI Audio Detector - Direct File Upload")
    print(f"{'='*60}\n")
    
    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: File not found: {audio_file_path}")
        return
    
    print(f"📁 File: {audio_file_path}")
    print(f"🌍 Language: {language}")
    print(f"🔑 API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if len(API_KEY) > 4 else '****'}")
    
    # First, check API health
    print("\n1️⃣ Checking API health...")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        health_data = health_response.json()
        print(f"   ✅ API Status: {health_data['status']}")
        print(f"   🖥️  Device: {health_data['device']}")
        print(f"   📦 Model Loaded: {health_data['model_loaded']}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Error: Cannot connect to API. Is it running?")
        print(f"   💡 Start the API with: python api.py")
        return
    except Exception as e:
        print(f"   ❌ Error checking health: {e}")
        return
    
    # Upload and detect
    print(f"\n2️⃣ Uploading audio file and detecting...")
    headers = {"x-api-key": API_KEY}
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {"language": language}
            
            response = requests.post(
                f"{API_URL}/api/detect-from-file",
                files=files,
                data=data,
                headers=headers,
                timeout=30
            )
        
        # Check response
        if response.status_code == 401:
            print("   ❌ Authentication failed: Invalid API key")
            print("   💡 Set API_KEY environment variable or edit this script")
            return
        
        if response.status_code != 200:
            print(f"   ❌ Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        result = response.json()
        
        # Display results
        print("\n" + "="*60)
        print("🎯 DETECTION RESULTS")
        print("="*60)
        
        status = result.get('status', 'unknown')
        classification = result.get('classification', 'UNKNOWN')
        confidence = result.get('confidenceScore', 0.0)
        
        # Color coding based on classification (for terminal)
        if status == "success":
            if classification == "AI_GENERATED":
                label = "🤖 AI-GENERATED"
                confidence_label = f"🎯 Confidence: {confidence:.2%}"
            elif classification == "HUMAN":
                label = "👤 HUMAN"
                confidence_label = f"🎯 Confidence: {confidence:.2%}"
            elif classification == "UNCERTAIN":
                label = "❓ UNCERTAIN (Grey Zone)"
                confidence_label = f"⚠️  Ambiguity: {confidence:.2%} (lower = more uncertain)"
            else:
                label = f"❔ {classification}"
                confidence_label = f"🎯 Confidence: {confidence:.2%}"
            
            print(f"\n📊 Status: ✅ {status.upper()}")
            print(f"🔍 Classification: {label}")
            print(f"{confidence_label}")
            
            if result.get('audioDuration'):
                print(f"⏱️  Audio Duration: {result['audioDuration']:.1f}s")
            if result.get('segmentsAnalyzed'):
                print(f"📦 Segments Analyzed: {result['segmentsAnalyzed']}")
            
            print(f"\n💬 Explanation:")
            print(f"   {result.get('explanation', 'N/A')}")
            
        elif status == "rejected":
            print(f"\n📊 Status: ⚠️  {status.upper()}")
            print(f"🔍 Classification: {classification}")
            print(f"\n❌ Audio Rejected:")
            print(f"   {result.get('explanation', 'N/A')}")
            if result.get('validationError'):
                print(f"\n🔧 Technical Details:")
                print(f"   {result['validationError']}")
        
        print("\n" + "="*60)
        
    except requests.exceptions.Timeout:
        print("   ❌ Error: Request timed out (audio might be too long or processing is slow)")
    except Exception as e:
        print(f"   ❌ Error during detection: {e}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python test_file_upload.py <audio_file> [language]")
        print("\nExample:")
        print("  python test_file_upload.py audio.mp3 Tamil")
        print("\nSupported languages: Tamil, English, Hindi, Malayalam, Telugu")
        print("\nMake sure the API is running first:")
        print("  python api.py")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "English"
    
    test_file_upload(audio_file, language)
