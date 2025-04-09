# CSM-1B with Gemini 2.0 Flash Integration

**2025/04/08 UPDATE** - This repository contains a working deployment of the Sesame CSM-1B model on Cerebrium with the **Gemini 2.0 Flash** backend replacing the original Llama-3.2-1B model. This implementation generates high-quality speech from text input through a scalable API endpoint.

**Original 2025/03/13** - Sesame released the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm-1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. Our modified implementation replaces the original Llama backbone with Google's **Gemini 2.0 Flash model**, while maintaining the smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

The original fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in the [Sesame blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) for the original model is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU (for local development)
* Python 3.10+ (local testing done with Python 3.12)
* For some audio operations, `ffmpeg` may be required
* Access to the [CSM-1B](https://huggingface.co/sesame/csm-1b) Hugging Face model
* Google API key for accessing the Gemini 2.0 Flash model
* Cerebrium account for cloud deployment

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B
huggingface-cli login

# Set up your .env file with required credentials
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

Create a `.env` file with the following variables:

```
# Google API key for Gemini access
GOOGLE_API_KEY=your_google_api_key_here

# Enable Gemini model (set to "true" to use Gemini, "false" to use Llama)
USE_GEMINI=true

# Gemini model to use
GEMINI_MODEL=gemini-2.0-flash-exp

# For Cerebrium deployment testing
CEREBRIUM_API_KEY=your_inference_token_here
CEREBRIUM_URL=your_deployment_url_here
```

## Using CSM-1B with Gemini

This implementation replaces the Llama-3.2-1B model with Google's Gemini 2.0 Flash model for text processing while maintaining the same audio generation pipeline.

### Key Files

- `gemini_model.py`: Implementation of the GeminiWrapper class that interfaces with the Gemini API
- `gemini_tokenizer.py`: Custom tokenizer for Gemini that's compatible with the CSM pipeline
- `main.py`: Entry point for the Cerebrium deployment with Gemini integration

### Quickstart

To test the model locally with the Gemini backend:

```bash
# Make sure your .env file is set up with GOOGLE_API_KEY and USE_GEMINI=true
python test_gemini_csm.py
```

## Usage

If you want to write your own applications with CSM + Gemini, the following examples show basic usage.

#### Generate a sentence

This will use a random speaker identity, as no prompt or context is provided.

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

#### Generate with context

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

NOTE: The following example is instructional and the audio files do not exist. It is intended as an example for using context with CSM.

```python
from generator import Segment

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.

---

# Cerebrium Deployment Guide

## Overview

Deploying the CSM-1B model with Gemini 2.0 Flash integration to Cerebrium provides a scalable, production-ready API for text-to-speech generation. This section explains how to deploy and test the model on Cerebrium.

This repository extends the CSM-1B model to provide a deployable API on the Cerebrium platform. The implementation enables text-to-speech generation through a simple HTTP endpoint. The deployed API accepts text input and returns base64-encoded WAV audio data.

**Key Features of Our Implementation:**

- **Llama-3.2-1B Tokenizer Integration**: Uses the official Llama tokenizer for optimal speech generation quality
- **Robust Error Handling**: Includes sophisticated fallback mechanisms with parameter adjustment when initial generation fails
- **Intelligent Retries**: Automatically attempts generation with different parameters if initial attempt produces no output
- **Comprehensive Testing**: Includes test scripts for both local development and remote API testing

## Setup & Deployment

### Prerequisites

- Cerebrium account - Sign up at [cerebrium.ai](https://cerebrium.ai)
- Cerebrium CLI - Install with `pip install cerebrium-cli`
- Python 3.10 or newer

### Deployment Steps

1. **Clone this repository**
   ```bash
   git clone https://github.com/rohankatakam/csm_cerebrium.git
   cd csm_cerebrium
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Deploy to Cerebrium**
   ```bash
   cerebrium deploy
   ```
   This uses the configuration in `cerebrium.toml` to set up the deployment with appropriate compute resources.

## Local Development vs. Cerebrium Deployment

### Local Development

1. **Environment Setup**
   ```bash
   git clone https://github.com/rohankatakam/csm_cerebrium.git
   cd csm_cerebrium
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   
   # Important: Disable lazy compilation in Mimi
   export NO_TORCH_COMPILE=1
   ```

2. **Hugging Face Authentication**
   ```bash
   huggingface-cli login
   ```
   You'll need access to both the [CSM-1B](https://huggingface.co/sesame/csm-1b) and [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) models.

3. **Local Testing**
   ```bash
   # Test with default phrase and minimal context
   python test_with_context.py
   
   # Test with custom phrase
   python test_with_context.py --text "Your custom phrase here"
   
   # Test without context (not recommended)
   python test_with_context.py --no-context
   ```

### Cerebrium Deployment

1. **Install Cerebrium CLI**
   ```bash
   pip install cerebrium-cli
   ```

2. **Deploy to Cerebrium**
   ```bash
   cerebrium deploy
   ```
   The deployment uses the configuration in `cerebrium.toml` which specifies GPU requirements, Python version, etc.

3. **Test Deployed API**
   ```bash
   # Set your Cerebrium inference token
   export CEREBRIUM_TOKEN=YOUR_INFERENCE_TOKEN
   
   # Test with the phrase "What up dog?"
   python test_phrase.py --text "What up dog?" --endpoint "https://api.cortex.cerebrium.ai/v4/{project_id}/{deployment_name}/generate_audio"
   ```

### When to Deploy to Cerebrium

- **Development Phase**: Use local testing with `test_with_context.py` for rapid iteration and debugging
- **Production Readiness**: Once your local tests are successful, deploy to Cerebrium for production use
- **Code Changes**: After significant code changes (like updating the tokenizer), redeploy to Cerebrium
- **Scaling Needs**: When you need to scale beyond your local resources or need high availability

### Known Issues & Solutions

- **Google Generative AI Package**: Ensure you have `google-genai>=0.1.7` in requirements.txt (not just `google-generativeai`)
- **Missing hf_transfer package**: Added to requirements.txt and disabled with environment variable `HF_HUB_ENABLE_HF_TRANSFER=0` in main.py
- **Model Compatibility**: The Gemini 2.0 Flash backend replaces the Llama 3.2 1B model, avoiding the need for Llama access
- **API Import Errors**: We've added error handling to gracefully degrade to standard models if API imports fail
- **API Authentication**: Requires an Inference Token from your Cerebrium dashboard, set as `CEREBRIUM_API_KEY` in your .env file

## Using the API

### API Endpoint

Once deployed, the API endpoint will be available at:
```
https://api.cortex.cerebrium.ai/v4/{project_id}/{deployment_name}/generate_audio
```

### Request Format

Make a POST request with the following JSON payload:
```json
{
  "text": "Your text to convert to speech goes here."
}
```

### Response Format

The API returns a JSON response with the following structure:
```json
{
  "run_id": "unique-request-id",
  "result": {
    "audio_data": "base64-encoded-wav-data",
    "format": "wav",
    "encoding": "base64"
  },
  "run_time_ms": 300.5
}
```

### Authentication

Include your Cerebrium Inference Token in the request headers:
```
Authorization: Bearer YOUR_INFERENCE_TOKEN
```

## Testing the Deployment

### Using the Test Script

This repository includes a `test_deployed_api.py` script to test the deployed API:

```bash
python test_deployed_api.py --endpoint https://api.cortex.cerebrium.ai/v4/{project_id}/{deployment_name}/generate_audio --token YOUR_INFERENCE_TOKEN
```

Replace `{project_id}` and `{deployment_name}` with your specific values, and `YOUR_INFERENCE_TOKEN` with the token from your Cerebrium dashboard.

### Sample Code

```python
import requests
import base64

# API endpoint and authentication
endpoint = "https://api.cortex.cerebrium.ai/v4/{project_id}/{deployment_name}/generate_audio"
token = "YOUR_INFERENCE_TOKEN"

# Request payload
payload = {"text": "Hello, this is a test of the CSM voice model."}

# Make the request
response = requests.post(
    endpoint,
    json=payload,
    headers={"Authorization": f"Bearer {token}"}
)

# Process the response
if response.status_code == 200:
    result = response.json()["result"]
    audio_data = result["audio_data"]
    
    # Save to file
    with open("output.wav", "wb") as f:
        f.write(base64.b64decode(audio_data))
    print("Success! Audio saved to output.wav")
else:
    print(f"Error: {response.text}")
```

## Performance Considerations

- The API typically responds in 5-10 seconds for short text inputs on Cerebrium
- For longer texts, response time scales approximately linearly with text length
- Local testing is faster but requires appropriate GPU hardware
- The deployment is configured to automatically scale based on demand

## Implementation Details

### Core Components

1. **Tokenization**: Uses the Llama-3.2-1B tokenizer for text processing
2. **Error Handling**: 
   - Detects empty sample cases and implements intelligent retry with modified parameters
   - Reduces temperature and narrows top-k sampling for more focused generation
   - Can simplify long prompts if needed
3. **Context Support**: Designed to work best with audio context (previous speech segments)

### Testing Scripts

- **test_phrase.py**: For testing the deployed Cerebrium API
- **test_with_context.py**: For local testing with proper context handling

## Implementation Details

### Gemini Integration

Our implementation makes several key modifications to integrate Gemini 2.0 Flash:

1. **GeminiWrapper Class**: Mimics the interface of the Llama transformer for seamless integration
2. **Custom Tokenization**: Implements a `GeminiTokenizer` class that interfaces with the Gemini API
3. **Fallback Mechanism**: Gracefully handles API errors and falls back to standard models if needed
4. **Deterministic Hashing**: Generates unique token IDs based on input text to ensure consistent tokenization

### Testing Scripts

- **test_deployed_gemini.py**: For testing the deployed Cerebrium API with Gemini
- **test_gemini_csm.py**: For local testing of the Gemini integration
- **test_with_context.py**: For local testing with proper context handling

## Best Practices

1. **Test Locally First**: Always test changes locally before deploying to Cerebrium
2. **Use Context When Possible**: The model works best with proper context
3. **Set Proper Environment Variables**: Ensure all API keys and configuration settings are correct
4. **Monitor Log Output**: Check for warnings about API errors or retries
5. **Verify Audio Quality**: Always verify the audio outputs match expectations

## Future Improvements

- Streaming audio response for real-time applications
- Better context handling for more natural conversational speech
- Voice cloning capabilities using Gemini's context handling
- Fine-tuning for specialized domains
- Integration with Gemini's multimodal capabilities for audio and text inputs
