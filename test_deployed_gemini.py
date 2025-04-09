"""
Test script for accessing the deployed Gemini-powered CSM-1B model on Cerebrium
"""
import os
import sys
import json
import base64
import requests
import time
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Get Cerebrium credentials from environment variables
# You'll set these in your .env file after deployment
CEREBRIUM_API_KEY = os.environ.get("CEREBRIUM_API_KEY", "")

# The URL for the deployed model - will be available in the Cerebrium dashboard
# Example: https://run.cerebrium.ai/v2/predict/10-sesame-voice-api
CEREBRIUM_URL = os.environ.get("CEREBRIUM_URL", "")

def test_deployed_tts():
    """Test the deployed TTS model with a text input"""
    if not CEREBRIUM_API_KEY:
        print("ERROR: CEREBRIUM_API_KEY environment variable is not set.")
        print("Please set it to your Inference Token from the Cerebrium dashboard.")
        return False
        
    if not CEREBRIUM_URL:
        print("ERROR: CEREBRIUM_URL environment variable is not set.")
        print("Please set it to your deployment URL from the Cerebrium dashboard.")
        return False
    
    # Prepare the request
    headers = {
        "Authorization": f"Bearer {CEREBRIUM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Input to test
    payload = {
        "text": "Hello there! This is a test of the Gemini-powered CSM-1B model on Cerebrium.",
        "speaker": 0,
        "max_audio_length_ms": 10000,
        "temperature": 0.7,
        "topk": 50
    }
    
    print(f"Testing deployed model at {CEREBRIUM_URL}")
    print(f"Input text: '{payload['text']}'")
    
    # Send the request
    try:
        start_time = time.time()
        response = requests.post(CEREBRIUM_URL, headers=headers, json=payload)
        request_time = time.time() - start_time
        
        # Check for success
        if response.status_code == 200:
            result = response.json()
            
            # Handle possible error in the response
            if "error" in result:
                print(f"API Error: {result['error']}")
                if "traceback" in result:
                    print(f"Traceback: {result['traceback']}")
                return False
                
            # Extract and save the audio
            # Check if we have the new Cerebrium API structure with a 'result' field
            if "result" in result and isinstance(result["result"], dict):
                result_data = result["result"]
            else:
                result_data = result
            
            if "audio_data" in result_data:
                # Decode base64 audio data
                audio_data = base64.b64decode(result_data["audio_data"])
                
                # Save to file
                output_file = "cerebrium_output.wav"
                with open(output_file, "wb") as f:
                    f.write(audio_data)
                
                # Stats
                print(f"Success! Audio saved to {output_file}")
                print(f"Audio format: {result_data.get('format', 'wav')}")
                print(f"Duration: {result_data.get('duration_seconds', 'unknown')} seconds")
                print(f"Processing time: {result_data.get('processing_time_seconds', 'unknown')} seconds")
                print(f"Request time: {request_time:.2f} seconds")
                
                return True
            else:
                print(f"API response missing audio_data: {json.dumps(result, indent=2)}")
                return False
        else:
            print(f"API Error: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_deployed_tts()
