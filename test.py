import os
import time
import torch
import torchaudio
import soundfile as sf
from generator import load_csm_1b, Segment

# The text we want to convert to speech
test_text = "Cerebrium is a, uh, really great cloud platform for deploying your voice models. It's easy to use and the team is very helpful."

# Select device based on availability
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# Disable HF Transfer to avoid dependency issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Load the CSM generator
print("Loading CSM-1B generator...")
try:
    generator = load_csm_1b(device=device)
    print("CSM-1B generator loaded successfully!")
    
    # Time the request
    print(f"Sending text to be converted: \"{test_text}\"")
    start_time = time.time()
    
    # Generate audio with empty context
    audio = generator.generate(
        text=test_text,
        speaker=0,  # Using default speaker
        context=[],  # No context segments
        temperature=0.9,  # Controls randomness
        max_audio_length_ms=10_000,  # Limit to 10 seconds for faster testing
    )
    end_time = time.time()
    print(f"Generated audio in {end_time - start_time:.2f} seconds!")

    # Save to file
    output_file = "output.wav"
    torchaudio.save(output_file, audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {output_file}")
    
    # Get audio info
    audio_info = sf.info(output_file)
    print(f"Audio length: {audio_info.duration:.2f} seconds")
    print(f"Sample rate: {audio_info.samplerate} Hz")
    print(f"Channels: {audio_info.channels}")
    
    print("\nTest completed successfully!")
except Exception as e:
    print(f"Error: {str(e)}")