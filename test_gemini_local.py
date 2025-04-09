"""
Simple test script for Gemini integration with CSM-1B on CPU
"""
import os
import torch
import dotenv
from generator import load_csm_1b

# Load environment variables
dotenv.load_dotenv()

# This test script forces CPU mode to avoid MPS/CUDA issues
os.environ["FORCE_CPU"] = "1"

def test_simple():
    """Test basic Gemini functionality"""
    print("Testing Gemini integration with CSM-1B in CPU mode")
    
    # Verify environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    use_gemini = os.environ.get("USE_GEMINI", "false").lower() == "true"
    
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        return False
    
    if not use_gemini:
        print("ERROR: USE_GEMINI not set to 'true'")
        return False
    
    print(f"API Key: {'*' * 8}{api_key[-4:]}")
    print(f"USE_GEMINI: {use_gemini}")
    
    # Force CPU mode
    device = "cpu"
    print(f"Using device: {device}")
    
    # Create generator with specific dtype
    try:
        print("Creating generator...")
        generator = load_csm_1b(
            device=device, 
            use_gemini=True,
            dtype=torch.float32  # Use float32 for compatibility
        )
        print("Generator created successfully")
        
        # Test tokenization
        text = "Hello, this is a test for the Gemini model."
        print(f"Tokenizing: '{text}'")
        tokens, masks = generator._tokenize_text_segment(text, speaker=0)
        print(f"Tokenization successful: {tokens.shape}")
        
        print("Test completed successfully!")
        return True
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_simple()
