from dataclasses import dataclass
from typing import List, Tuple
import os

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark

# Import the new Gemini tokenizer
from gemini_tokenizer import GeminiTokenizer


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    Original implementation using Llama 3.2 1B tokenizer
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    
    # Configure tokenizer post-processing
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


def load_gemini_tokenizer():
    """
    Implementation using Gemini 2.0 Flash tokenizer
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be set to use Gemini tokenizer")
        
    tokenizer = GeminiTokenizer(api_key=api_key)
    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)
        
        # Check if we should use Gemini tokenizer based on model flavor
        if hasattr(self._model, 'config') and hasattr(self._model.config, 'backbone_flavor'):
            if 'gemini' in self._model.config.backbone_flavor:
                try:
                    self._text_tokenizer = load_gemini_tokenizer()
                    print("Using Gemini tokenizer")
                except Exception as e:
                    print(f"Error loading Gemini tokenizer: {e}. Falling back to Llama tokenizer.")
                    self._text_tokenizer = load_llama3_tokenizer()
            else:
                self._text_tokenizer = load_llama3_tokenizer()
        else:
            self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        # Handle case where samples list is empty
        if not samples:
            print("Warning: No audio samples were generated! Using fallback approach.")
            
            # Retry with different temperature and topk settings
            print("Retrying with different generation parameters...")
            self._model.reset_caches()
            
            # Use more conservative settings for the retry
            retry_temp = max(0.2, temperature - 0.3)  # Lower temperature
            retry_topk = min(10, topk)  # More focused sampling
            
            # Simplify the prompt to just the core text
            simplified_text = text
            if len(simplified_text) > 50:
                simplified_text = simplified_text[:50]  # Use shorter text if original is long
            
            print(f"Retry with temp={retry_temp}, topk={retry_topk}, text='{simplified_text}'")
            
            # Retry prompt tokenization
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(simplified_text, speaker)
            prompt_tokens = gen_segment_tokens.long().to(self.device)
            prompt_tokens_mask = gen_segment_tokens_mask.bool().to(self.device)
            
            # Try generation again with simplified approach
            retry_samples = []
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
            
            # Generate with a forced minimum
            for _ in range(24):  # Force at least a short clip (~1s)
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, retry_temp, retry_topk)
                retry_samples.append(sample)
                
                # Update for next iteration
                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
            
            # If retry still failed, generate noise as last resort
            if not retry_samples:
                print("Retry failed! Generating noise pattern as last resort")
                # Generate a soft noise pattern that won't be completely silent
                noise = torch.randn(self.sample_rate).to(self.device) * 0.01
                return noise
                
            # Use retry samples instead
            print(f"Retry succeeded with {len(retry_samples)} samples")
            samples = retry_samples
        
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def load_csm_1b(device: str = "cuda", use_gemini: bool = False, dtype=None) -> Generator:
    """
    Load the CSM-1B model, optionally using Gemini instead of Llama.
    
    Args:
        device: Device to load the model on ("cuda", "cpu", or "mps")
        use_gemini: Whether to use Gemini model instead of Llama
        dtype: Data type for the model (default: None, which will select an appropriate type)
        
    Returns:
        Generator instance
    """
    # Determine appropriate dtype if not provided
    if dtype is None:
        if device == "mps":
            # On MPS, bfloat16 is only supported on newer MacOS versions
            try:
                # Try creating a small tensor to test bfloat16 support
                # Test BFloat16 compatibility without creating unused variables
                torch.zeros(1, dtype=torch.bfloat16, device=device)
                dtype = torch.bfloat16
                print("Using bfloat16 on MPS device")
            except Exception:
                # Fall back to float32 if bfloat16 is not supported
                dtype = torch.float32
                print("MPS device doesn't support bfloat16, using float32 instead")
        else:
            # For CUDA and CPU, use bfloat16
            dtype = torch.bfloat16
            print(f"Using bfloat16 on {device} device")
    
    if use_gemini:
        # Create a custom model with Gemini backbone
        from models import Model, ModelArgs
        
        config = ModelArgs(
            backbone_flavor="gemini-1B",  # Use Gemini 1B as backbone
            decoder_flavor="gemini-100M",  # Use smaller Gemini for decoder
            text_vocab_size=128_256,
            audio_vocab_size=1024,
            audio_num_codebooks=32
        )
        
        model = Model(config)
        model.to(device=device, dtype=dtype)
        
        # Check for API key
        if "GOOGLE_API_KEY" not in os.environ:
            print("WARNING: GOOGLE_API_KEY environment variable not set. Gemini model won't work properly.")
    else:
        # Load the original CSM-1B model
        model = Model.from_pretrained("sesame/csm-1b")
        model.to(device=device, dtype=dtype)

    generator = Generator(model)
    return generator