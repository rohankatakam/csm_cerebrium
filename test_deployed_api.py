#!/usr/bin/env python
# Test script for the deployed Cerebrium CSM-1B API
import requests
import base64
import os
import time
import argparse

def test_cerebrium_api(text, endpoint_url, token):
    """Test the deployed Cerebrium CSM-1B API endpoint"""
    print(f"Testing Cerebrium API with text: '{text}'")
    start_time = time.time()
    
    # Set up the request
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Only send the text parameter for now (to match the deployed version)
    payload = {
        "text": text
    }
    
    # Make the request
    print("Sending request to Cerebrium...")
    try:
        response = requests.post(endpoint_url, json=payload, headers=headers)
        
        # Debug info
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        # Try to parse as JSON
        try:
            result = response.json()
            print(f"Response JSON: {result}")
        except Exception as e:
            print(f"Could not parse response as JSON: {e}")
            print(f"Raw response content: {response.text}")
            return False
        
        # Check for errors in the response
        if "error" in result:
            print(f"API returned error: {result['error']}")
            if "traceback" in result:
                print(f"Traceback: {result['traceback']}")
            return False
            
        # Check for Cerebrium's response format which includes 'result' wrapper
        if "result" in result:
            print("Found Cerebrium result wrapper, extracting inner result")
            # The actual function output is in the 'result' field
            inner_result = result["result"]
            if isinstance(inner_result, dict):
                result = inner_result
            else:
                print(f"Error: 'result' key doesn't contain a dictionary: {inner_result}")
                return False
        
        # Check if audio_data is present    
        if "audio_data" not in result:
            print(f"Error: 'audio_data' not found in response. Response keys: {list(result.keys())}")
            return False
            
        # Get the audio data
        audio_data = result["audio_data"]
        audio_format = result.get("format", "wav")
        
        # Save to file
        output_file = f"cerebrium_output.{audio_format}"
        with open(output_file, "wb") as f:
            f.write(base64.b64decode(audio_data))
            
        # Print stats
        print(f"Success! Audio saved to {output_file}")
        print(f"Audio duration: {result.get('duration_seconds', 'unknown')} seconds")
        print(f"Processing time: {result.get('processing_time_seconds', time.time() - start_time):.2f} seconds")
        print(f"Total round-trip time: {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        import traceback
        print(f"Exception during API request: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the deployed Cerebrium CSM-1B API")
    parser.add_argument("--text", type=str, 
                        default="Cerebrium is a really great cloud platform for deploying your voice models. It's easy to use and the team is very helpful.",
                        help="Text to convert to speech")
    parser.add_argument("--endpoint", type=str, 
                        help="Cerebrium endpoint URL (e.g., https://run.cerebrium.ai/v2/10-sesame-voice-api/predict)")
    parser.add_argument("--token", type=str,
                        help="Cerebrium inference token")
    
    args = parser.parse_args()
    
    # Check for environment variables if not provided as args
    endpoint_url = args.endpoint or os.environ.get("CEREBRIUM_ENDPOINT")
    token = args.token or os.environ.get("CEREBRIUM_TOKEN")
    
    if not endpoint_url:
        print("ERROR: Please provide the Cerebrium endpoint URL either via --endpoint or CEREBRIUM_ENDPOINT env variable")
        exit(1)
        
    if not token:
        print("ERROR: Please provide the Cerebrium inference token either via --token or CEREBRIUM_TOKEN env variable")
        exit(1)
    
    # Run the test
    test_cerebrium_api(args.text, endpoint_url, token)
