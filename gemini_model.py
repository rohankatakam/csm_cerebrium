"""
Gemini 2.0 Flash model wrapper compatible with CSM-1B
"""
import os
from typing import Optional

import torch
import torch.nn as nn
from google import genai
from google.genai import types

class GeminiWrapper(nn.Module):
    """
    A wrapper class for Gemini 2.0 Flash that mimics the interface of Llama 3.2 transformer.
    
    This class provides an interface compatible with torchtune's TransformerDecoder to minimize
    changes needed in the existing CSM-1B code.
    """
    
    def __init__(
        self,
        vocab_size: int = 128_256,
        embed_dim: int = 2048,
        max_seq_len: int = 2048,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        """
        Initialize the GeminiWrapper.
        
        Args:
            vocab_size: Size of vocabulary (mostly for compatibility)
            embed_dim: Embedding dimension (mostly for compatibility)
            max_seq_len: Maximum sequence length (mostly for compatibility)
            api_key: Gemini API key, if not provided will look for GOOGLE_API_KEY in environment
            model_name: Name of the Gemini model to use
        """
        super().__init__()
        
        # Set up the Gemini client
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            
        if not api_key:
            raise ValueError("API key must be provided either directly or via GOOGLE_API_KEY environment variable")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Store configuration for compatibility
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Dummy embedding layer to be replaced with Identity in _prepare_transformer
        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Dummy output layer to be replaced with Identity in _prepare_transformer
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # Internal cache for generated content
        self._text_cache = {}
        self._last_input_text = None
        self._last_output_embedding = None
        
        # Add a compatibility layer to mimic transformer's hidden states
        self.hidden_proj = nn.Linear(1, embed_dim)
        
        # Whether KV caches have been set up (for compatibility)
        self._caches_enabled = False
        self._max_batch_size = 1
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that mimics a transformer's forward pass but uses Gemini API.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, embed_dim)
            input_pos: Positions for transformer (ignored, just for compatibility)
            mask: Attention mask (ignored, just for compatibility)
            
        Returns:
            Output tensor with same shape as input
        """
        # Get the batch size and sequence length
        batch_size, seq_len, _ = hidden_states.shape
        
        # We don't actually use the hidden states in this implementation
        # since we're using the Gemini API for generation
        
        # Create a random but deterministic output based on the input
        # This is a placeholder - in reality, we would use the Gemini API
        # But since we can't do that during a forward pass, we'll use this trick
        # and then replace it with real API calls during generation
        
        # Check if we have cached text and embeddings
        if self._last_input_text is not None and self._last_output_embedding is not None:
            # Use the cached embedding (will be replaced with actual Gemini output during generation)
            return self._last_output_embedding.expand(batch_size, seq_len, self.embed_dim)
        
        # Otherwise generate a placeholder output
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Create a deterministic but random-looking embedding based on the input
        # This is just a placeholder for the forward pass
        output = torch.sin(hidden_states + 0.1) * 0.1 + hidden_states
        
        return output
    
    def setup_caches(self, max_batch_size: int, dtype: torch.dtype = torch.bfloat16, decoder_max_seq_len: int = None) -> None:
        """
        Setup KV caches (dummy implementation for compatibility).
        
        Args:
            max_batch_size: Maximum batch size
            dtype: Data type for caches
            decoder_max_seq_len: Maximum sequence length for decoder (optional)
        """
        self._caches_enabled = True
        self._max_batch_size = max_batch_size
        self._decoder_max_seq_len = decoder_max_seq_len if decoder_max_seq_len is not None else self.max_seq_len
    
    def reset_caches(self) -> None:
        """Reset KV caches (dummy implementation for compatibility)."""
        # Clear any cached text or embeddings
        self._text_cache = {}
        self._last_input_text = None
        self._last_output_embedding = None
    
    def caches_are_enabled(self) -> bool:
        """Check if caches are enabled (for compatibility)."""
        return self._caches_enabled
    
    async def generate_with_gemini(self, text: str) -> str:
        """
        Generate text using the Gemini API.
        
        Args:
            text: Input text
            
        Returns:
            Generated text
        """
        try:
            # If we've already generated for this text, use the cached result
            if text in self._text_cache:
                return self._text_cache[text]
            
            # Prepare input for Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                )
            ]
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.2,  # Lower temperature for more deterministic outputs
                max_output_tokens=128,  # Limit output length
            )
            
            # Generate content
            response = await self.client.models.generate_content_async(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            # Extract generated text
            generated_text = response.text
            
            # Cache the result
            self._text_cache[text] = generated_text
            
            return generated_text
        except Exception as e:
            # Fallback in case of API error
            print(f"Gemini API error: {e}")
            return f"Error generating text: {str(e)[:50]}..."
    
    def generate_with_gemini_sync(self, text: str) -> str:
        """
        Generate text using the Gemini API (synchronous version).
        
        Args:
            text: Input text
            
        Returns:
            Generated text
        """
        try:
            # If we've already generated for this text, use the cached result
            if text in self._text_cache:
                return self._text_cache[text]
            
            # Prepare input for Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                )
            ]
            
            # Configure generation
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.2,  # Lower temperature for more deterministic outputs
                max_output_tokens=128,  # Limit output length
            )
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            # Extract generated text
            generated_text = response.text
            
            # Cache the result
            self._text_cache[text] = generated_text
            
            return generated_text
        except Exception as e:
            # Fallback in case of API error
            print(f"Gemini API error: {e}")
            return f"Error generating text: {str(e)[:50]}..."
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Create an embedding for text.
        This is a placeholder since we don't have direct access to Gemini's embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor
        """
        # Create a deterministic embedding based on the text
        # In a real implementation, you'd use the Gemini API to get embeddings
        # But since that's not exposed, we create a simplified version
        import hashlib
        
        # Hash the text to get a deterministic seed
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
        
        # Create a random but deterministic embedding
        generator = torch.Generator().manual_seed(text_hash)
        embedding = torch.randn(self.embed_dim, generator=generator, dtype=torch.float32)
        
        # Normalize the embedding
        embedding = embedding / embedding.norm()
        
        return embedding
    
    def set_last_io(self, input_text: str, output_embedding: torch.Tensor) -> None:
        """
        Set the last input text and output embedding for caching.
        
        Args:
            input_text: Input text
            output_embedding: Output embedding
        """
        self._last_input_text = input_text
        self._last_output_embedding = output_embedding


def gemini_flash_1B() -> GeminiWrapper:
    """
    Create a GeminiWrapper instance that mimics Llama 3.2 1B.
    
    Returns:
        GeminiWrapper instance
    """
    return GeminiWrapper(
        vocab_size=128_256,
        embed_dim=2048,
        max_seq_len=2048,
    )


def gemini_flash_100M() -> GeminiWrapper:
    """
    Create a smaller GeminiWrapper instance that mimics Llama 3.2 100M.
    
    Returns:
        GeminiWrapper instance
    """
    return GeminiWrapper(
        vocab_size=128_256,
        embed_dim=1024,
        max_seq_len=2048,
    )
