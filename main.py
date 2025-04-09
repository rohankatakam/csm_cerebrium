# Sesame CSM-1B implementation for Cerebrium, following the tutorial closely
import os
import time
import base64
import torch
import torchaudio
from generator import load_csm_1b
import dotenv

# Disable HF Transfer to avoid dependency issues on Cerebrium
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# First, check if google-generativeai is actually available
GEMINI_AVAILABLE = False

# Add diagnostic information to help debug Cerebrium issues
print("=== Cerebrium Deployment Diagnostics ===")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Installed packages:")
import subprocess
subprocess.call([sys.executable, "-m", "pip", "list"])
print("=======================================")

try:
    print("Attempting to import google-generativeai...")
    import google.generativeai
    print("Successfully imported google.generativeai!")
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"ERROR importing google.generativeai: {e}")
    GEMINI_AVAILABLE = False
    try:
        # Try installing it directly as a last resort
        print("Attempting to install google-generativeai package directly...")
        subprocess.call([sys.executable, "-m", "pip", "install", "google-generativeai==0.8.1"])
        import google.generativeai
        print("Successfully installed and imported google.generativeai!")
        GEMINI_AVAILABLE = True
    except Exception as e2:
        print(f"Failed to install google-generativeai: {e2}")

# Check if we should use Gemini
USE_GEMINI = os.environ.get("USE_GEMINI", "false").lower() == "true" and GEMINI_AVAILABLE

# Make sure we have an API key if using Gemini
if USE_GEMINI and "GOOGLE_API_KEY" not in os.environ:
    print("WARNING: USE_GEMINI is set to true but GOOGLE_API_KEY is not set.")
    print("Gemini model will not work properly without an API key.")
    USE_GEMINI = False
    print("Falling back to Llama model.")

if USE_GEMINI and not GEMINI_AVAILABLE:
    print("WARNING: USE_GEMINI is set to true but google-generativeai package is not available.")
    USE_GEMINI = False
    print("Falling back to Llama model.")

# This device selection lets our code work on any Cerebrium hardware
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load the CSM-1B generator - this happens once when the service starts
try:
    if USE_GEMINI:
        print("Loading CSM-1B model with Gemini 2.0 Flash backbone...")
    else:
        print("Loading standard CSM-1B model with Llama backbone...")
        
    start_time = time.time()
    generator = load_csm_1b(device=device, use_gemini=USE_GEMINI)
    load_time = time.time() - start_time
    print(f"CSM-1B model loaded successfully in {load_time:.2f} seconds!")
    print(f"Model is ready for inference with sample rate: {generator.sample_rate} Hz")
    
    if USE_GEMINI:
        print("Using Gemini 2.0 Flash for text generation")
except Exception as e:
    import traceback
    print(f"Error loading CSM-1B model: {str(e)}")
    print(traceback.format_exc())
    # Don't raise, just log - the generator will be attempted to be loaded when needed

# The function Cerebrium will call - keep it simple with just the text parameter
def generate_audio(text: str, speaker=0, max_audio_length_ms=30000, temperature=0.7, topk=50):
    """
    Generate speech from text using the Sesame CSM-1B model.

    Args:
        text (str): The text to convert to speech
        speaker (int, optional): Speaker ID. Defaults to 0.
        max_audio_length_ms (int, optional): Maximum audio length in milliseconds. Defaults to 30000.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        topk (int, optional): Top-k sampling parameter. Defaults to 50.

    Returns:
        dict: Contains base64-encoded audio data and format information
    """
    try:
        # Load generator if it's not already loaded (global variable)
        global generator
        if 'generator' not in globals() or generator is None:
            print("Loading generator on demand...")
            generator = load_csm_1b(device=device, use_gemini=USE_GEMINI)
        
        # Log the request
        print(f"Generating audio for text: '{text}'")
        print(f"Parameters: speaker={speaker}, max_audio_length_ms={max_audio_length_ms}, temperature={temperature}, topk={topk}")
        start_time = time.time()
        
        # Validate input
        if not text or not isinstance(text, str):
            return {"error": "Text input is required and must be a string"}
        
        # Type validation and conversion
        try:
            speaker = int(speaker)
            max_audio_length_ms = int(max_audio_length_ms)
            temperature = float(temperature)
            topk = int(topk)
        except (ValueError, TypeError) as e:
            return {"error": f"Parameter type conversion error: {str(e)}"}
        
        # Make sure we have enough audio length for the text
        # Rough estimate: 100ms per character
        estimated_length = len(text) * 100
        if max_audio_length_ms < estimated_length:
            print(f"Warning: max_audio_length_ms may be too short for text. Increasing from {max_audio_length_ms} to {estimated_length}")
            max_audio_length_ms = max(max_audio_length_ms, estimated_length)
        
        # Generate audio with CSM-1B model
        print("Starting audio generation...")
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=[],  # No context
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )
        print(f"Generation complete. Audio tensor shape: {audio.shape}, device: {audio.device}")

        # Make sure we have valid audio data
        if audio is None or audio.numel() == 0:
            return {"error": "Failed to generate audio - empty result"}
            
        # Get audio duration
        duration_seconds = audio.shape[0] / generator.sample_rate if audio.ndim > 0 else 0
        print(f"Generated audio duration: {duration_seconds:.2f} seconds")
        
        # Safety check if tensor is on different device
        if audio.device != torch.device('cpu'):
            audio = audio.cpu()

        # Save to temporary WAV file, read it, and convert to base64
        output_file = f"audio_{int(time.time())}.wav"
        torchaudio.save(output_file, audio.unsqueeze(0), generator.sample_rate)
        
        with open(output_file, "rb") as f:
            wav_data = f.read()
        
        # File stats for debugging
        file_size = os.path.getsize(output_file)
        print(f"Audio file size: {file_size} bytes")
        
        os.remove(output_file)  # Clean up the temporary file
        encoded_data = base64.b64encode(wav_data).decode('utf-8')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Generated audio in {processing_time:.2f}s")

        return {
            "audio_data": encoded_data, 
            "format": "wav", 
            "encoding": "base64",
            "duration_seconds": duration_seconds,
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        # Return error information with traceback
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating speech: {str(e)}")
        print(error_details)
        return {"error": str(e), "traceback": error_details}

# We'll add advanced endpoints after the basic one works correctly