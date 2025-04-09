# Sesame CSM-1B implementation for Cerebrium, following the tutorial closely
import os
import time
import base64
import torch
import torchaudio
from generator import load_csm_1b, Segment

# Disable HF Transfer to avoid dependency issues on Cerebrium
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

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
    print("Loading CSM-1B model...")
    start_time = time.time()
    generator = load_csm_1b(device=device)
    load_time = time.time() - start_time
    print(f"CSM-1B model loaded successfully in {load_time:.2f} seconds!")
    print(f"Model is ready for inference with sample rate: {generator.sample_rate} Hz")
except Exception as e:
    import traceback
    print(f"Error loading CSM-1B model: {str(e)}")
    print(traceback.format_exc())
    # Don't raise, just log - the generator will be attempted to be loaded when needed

# The function Cerebrium will call - keep it simple with just the text parameter
def generate_audio(text: str):
    """
    Generate speech from text using the Sesame CSM-1B model.

    Args:
        text (str): The text to convert to speech

    Returns:
        dict: Contains base64-encoded audio data and format information
    """
    try:
        # Load generator if it's not already loaded (global variable)
        global generator
        if 'generator' not in globals() or generator is None:
            generator = load_csm_1b(device=device)
        
        # Log the request
        print(f"Generating audio for text: '{text}'")
        start_time = time.time()
        
        # Validate input
        if not text or not isinstance(text, str):
            return {"error": "Text input is required and must be a string"}
        
        # Generate audio with CSM-1B model - use conservative settings
        audio = generator.generate(
            text=text,
            speaker=0,  # Default speaker
            context=[],  # No context
            max_audio_length_ms=20_000,  # Limit to 20 seconds
            temperature=0.9,  # Standard temperature
        )

        # Make sure we have valid audio data
        if audio is None or audio.numel() == 0:
            return {"error": "Failed to generate audio - empty result"}
            
        # Safety check if tensor is on different device
        if audio.device != torch.device('cpu'):
            audio = audio.cpu()

        # Save to temporary WAV file, read it, and convert to base64
        output_file = f"audio_{int(time.time())}.wav"
        torchaudio.save(output_file, audio.unsqueeze(0), generator.sample_rate)
        with open(output_file, "rb") as f:
            wav_data = f.read()
        os.remove(output_file)  # Clean up the temporary file
        encoded_data = base64.b64encode(wav_data).decode('utf-8')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Generated audio in {processing_time:.2f}s")

        return {"audio_data": encoded_data, "format": "wav", "encoding": "base64"}
    except Exception as e:
        # Return error information with traceback
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating speech: {str(e)}")
        print(error_details)
        return {"error": str(e), "traceback": error_details}

# We'll add advanced endpoints after the basic one works correctly