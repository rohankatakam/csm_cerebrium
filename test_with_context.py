#!/usr/bin/env python
# Test script for the CSM-1B model with context for better audio generation
import argparse
import base64
import os
import time
import torch
import torchaudio
from generator import load_csm_1b, Segment

# This device selection lets our code work on any hardware
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def create_empty_context():
    """Create an empty audio segment to provide minimal context"""
    # Create a short silent segment (0.5 seconds)
    sample_rate = 24000  # CSM's sample rate
    silent_audio = torch.zeros(int(0.5 * sample_rate))
    
    # Create a simple context segment
    context = [
        Segment(
            speaker=0,
            text="Hello.",
            audio=silent_audio
        )
    ]
    
    return context

def test_local_generation(text="Hello, what's up dog?", with_context=True):
    """Test the CSM-1B model locally with optional context"""
    print(f"Testing CSM-1B model locally with text: '{text}'")
    print(f"Device: {device}")
    
    try:
        # Load the generator
        print("Loading CSM-1B model...")
        start_time = time.time()
        generator = load_csm_1b(device=device)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Create context if requested
        context = create_empty_context() if with_context else []
        if with_context:
            print("Using minimal context (short silent segment)")
        else:
            print("Using empty context")
        
        # Generate audio
        print("Generating audio...")
        gen_start_time = time.time()
        audio = generator.generate(
            text=text,
            speaker=0,
            context=context,
            max_audio_length_ms=30000,
            temperature=0.7,
            topk=50
        )
        gen_time = time.time() - gen_start_time
        
        # Check if we got valid audio
        if audio is None or audio.numel() == 0:
            print("Error: Generated audio is empty")
            return False
        
        # Calculate duration
        duration = audio.shape[0] / generator.sample_rate if len(audio.shape) > 0 else 0
        print(f"Generated audio duration: {duration:.2f} seconds")
        
        # Save audio to file
        output_filename = "output_with_context.wav" if with_context else "output_no_context.wav"
        torchaudio.save(output_filename, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Print stats
        print(f"Audio saved to {output_filename}")
        print(f"Generation time: {gen_time:.2f} seconds")
        print(f"Audio file size: {os.path.getsize(output_filename)} bytes")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CSM-1B model locally")
    parser.add_argument("--text", type=str, default="Hello, what's up dog?", help="Text to convert to speech")
    parser.add_argument("--no-context", action="store_true", help="Don't use context (not recommended)")
    
    args = parser.parse_args()
    
    with_context = not args.no_context
    test_local_generation(args.text, with_context)
