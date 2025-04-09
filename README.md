# CSM Cerebrium Deployment

**2025/04/08 UPDATE** - This repository contains a working deployment of the Sesame CSM-1B model on Cerebrium with the Llama-3.2-1B tokenizer. Our implementation generates high-quality speech from text input through a scalable API endpoint.

**Original 2025/03/13** - Sesame released the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm-1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Quickstart

This script will generate a conversation between 2 characters, using a prompt for each character.

```bash
python run_csm.py
```

## Usage

If you want to write your own applications with CSM, the following examples show basic usage.

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

- **Missing hf_transfer package**: Added to requirements.txt and disabled with environment variable `HF_HUB_ENABLE_HF_TRANSFER=0` in main.py
- **Llama 3.2 1B model access**: Required for optimal performance - our implementation now uses the proper Llama-3.2-1B tokenizer
- **Silent Audio Output**: Fixed by using the correct Llama tokenizer and implementing robust fallback mechanisms
- **API Authentication**: Requires an Inference Token from your Cerebrium dashboard

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

## Best Practices

1. **Test Locally First**: Always test changes locally before deploying to Cerebrium
2. **Use Context When Possible**: The model works best with proper context
3. **Monitor Log Output**: Check for warnings about empty samples or retries
4. **Verify Audio Quality**: Always verify the audio outputs match expectations
5. **Track File Sizes**: Silent audio typically has a larger file size with repeating patterns

## Future Improvements

- Streaming audio response for real-time applications
- Better context handling for more natural conversational speech
- Voice cloning capabilities
- Fine-tuning for specialized domains
