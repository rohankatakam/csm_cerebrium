# Gemini 2.0 Flash Integration with CSM-1B

This document outlines the implementation details and usage instructions for the Gemini 2.0 Flash integration with the Sesame CSM-1B text-to-speech model.

## Overview

We've replaced the Llama 3.2 transformer architecture in CSM-1B with Gemini 2.0 Flash, Google's latest large language model. This integration leverages the Gemini API for text generation while maintaining compatibility with the existing audio generation components in CSM.

## Prerequisites

- A Google AI Studio account with access to Gemini 2.0 Flash
- A Gemini API key
- Python 3.9+ environment

## Setup

1. Create a `.env` file in the project root based on the provided `.env.example`:

```bash
cp .env.example .env
```

2. Edit the `.env` file and add your Gemini API key:

```
GOOGLE_API_KEY=your_api_key_here
USE_GEMINI=true
GEMINI_MODEL=gemini-2.0-flash-exp
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Testing

Run the provided test script:

```bash
python test_gemini_csm.py
```

This will generate a sample audio file using the Gemini-powered CSM-1B model.

### Cerebrium Deployment

The main Cerebrium entrypoint (`main.py`) has been updated to support both the original Llama-based model and the new Gemini-based model. The `USE_GEMINI` environment variable controls which model is used.

To deploy to Cerebrium:

1. Set the required environment variables in your Cerebrium dashboard:
   - `GOOGLE_API_KEY`: Your Gemini API key
   - `USE_GEMINI`: Set to "true" to use the Gemini model
   
2. Deploy using the existing deployment script.

## Implementation Details

The integration consists of the following components:

1. **GeminiTokenizer**: A wrapper class that provides a tokenizer interface compatible with the CSM-1B pipeline.

2. **GeminiWrapper**: A wrapper class that mimics the Llama 3.2 transformer interface while using the Gemini API.

3. **Modified Models**: The existing models.py file has been updated to include Gemini model configurations.

4. **Generator Adaptations**: The generator.py file has been updated to support both tokenizers.

## Limitations

1. **API Dependency**: This implementation relies on the Gemini API, so an internet connection is required for inference.

2. **Tokenization Approximation**: Since we don't have direct access to Gemini's internal tokenization, the tokenizer implementation uses a combination of API calls and approximations.

3. **Performance**: API calls add latency compared to local inference with Llama.

4. **Streaming**: Currently implemented as a non-streaming solution. Streaming capabilities can be added in a future update.

## Troubleshooting

If you encounter issues:

1. **API Key**: Ensure your Gemini API key is valid and correctly set in the environment variables.

2. **Model Availability**: Confirm you have access to the Gemini 2.0 Flash model in your Google AI Studio account.

3. **Rate Limits**: Be aware of the rate limits for the Gemini API and adjust your usage accordingly.

4. **Fallback**: The implementation includes fallback mechanisms to use the Llama model if Gemini initialization fails.
