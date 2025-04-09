"""
Test script for Gemini-powered CSM-1B model
"""
import os
import sys
import time
import torch
import torchaudio
import dotenv
from generator import load_csm_1b, Segment

# Load environment variables
dotenv.load_dotenv()

def test_gemini_csm():
    """Test Gemini-powered CSM-1B model for text-to-speech generation"""
    # Check if API key is set
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Please create a .env file with your API key or set it directly.")
        print("See .env.example for an example configuration.")
        sys.exit(1)
        
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load the CSM-1B model with Gemini backend
    print("Loading CSM-1B model with Gemini 2.0 Flash backbone...")
    start_time = time.time()
    
    try:
        # Select appropriate dtype based on device
        if device == "mps":
            # MPS doesn't fully support bfloat16 on some macOS versions
            print("Using float32 for MPS device")
            dtype = torch.float32
        else:
            # CUDA and CPU can use bfloat16
            dtype = torch.bfloat16
            
        # Use a custom function to load with specific dtype
        from models import Model, ModelArgs
        
        config = ModelArgs(
            backbone_flavor="gemini-1B",
            decoder_flavor="gemini-100M",
            text_vocab_size=128_256,
            audio_vocab_size=1024,
            audio_num_codebooks=32
        )
        
        model = Model(config)
        model.to(device=device, dtype=dtype)
        
        # Create generator
        from generator import Generator
        generator = Generator(model)
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
    
    # Generate audio without context
    text = "This is a test of the Gemini-powered CSM-1B model. It should convert this text to speech."
    speaker_id = 0
    
    print(f"Generating audio for: '{text}'")
    print(f"Parameters: speaker={speaker_id}, temperature=0.7, topk=50")
    
    # Time the generation
    gen_start_time = time.time()
    
    try:
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],  # No context
            max_audio_length_ms=10_000,
            temperature=0.7,
            topk=50,
        )
        gen_time = time.time() - gen_start_time
        
        # Audio statistics
        duration_seconds = audio.shape[0] / generator.sample_rate
        print(f"Generation successful in {gen_time:.2f} seconds!")
        print(f"Audio duration: {duration_seconds:.2f} seconds")
        
        # Save the audio
        output_file = "gemini_csm_output.wav"
        torchaudio.save(output_file, audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Audio saved to {output_file}")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to generate audio: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_gemini_csm()
