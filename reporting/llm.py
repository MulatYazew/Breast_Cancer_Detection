"""Stage 6e: LLM backends: local transformers, Hugging Face Inference API.

Both expose ``generate(messages) -> str`` over OpenAI-style chat messages.
Heavy imports (transformers / huggingface_hub) happen inside the constructors,
so importing this module costs nothing when reporting is unused.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """The language model could not produce a response."""

    kind = "llm"


class LLMRateLimitError(LLMError):
    kind = "rate_limit"


class LLMTimeoutError(LLMError):
    kind = "timeout"


class ChatLLM(Protocol):
    model_id: str
    backend: str

    def generate(self, messages: list[dict]) -> str: ...


class LocalLLM:
    """transformers causal LM on cuda | mps | cpu (``utils.seed.get_device``)."""

    backend = "local"

    def __init__(self, model_id: str, device: str | None = None, max_new_tokens: int = 900,
                 temperature: float = 0.2) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from utils.seed import get_device

        self.model_id = model_id
        self.device = get_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()

    def generate(self, messages: list[dict]) -> str:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        sampling = {"do_sample": True, "temperature": self.temperature} if self.temperature > 0 else {"do_sample": False}
        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id, **sampling,
                )
        except RuntimeError as exc:  # e.g. MPS/CUDA out of memory
            raise LLMError(f"local generation failed: {exc}") from exc
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class HFApiLLM:
    """Hugging Face Inference Providers (router) chat completion."""

    backend = "hf_api"

    def __init__(self, model_id: str, token: str | None, timeout_s: float = 120, max_new_tokens: int = 900,
                 temperature: float = 0.2) -> None:
        from huggingface_hub import InferenceClient

        if not token:
            raise LLMError("hf_api backend needs a Hugging Face token: set HF_TOKEN in Streamlit secrets or the environment")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.client = InferenceClient(model=model_id, token=token, timeout=timeout_s, provider="auto")

    def generate(self, messages: list[dict]) -> str:
        import httpx
        from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError, OverloadedError

        try:
            response = self.client.chat_completion(
                messages=messages, max_tokens=self.max_new_tokens, temperature=max(self.temperature, 0.01),
            )
        except (InferenceTimeoutError, httpx.TimeoutException) as exc:
            raise LLMTimeoutError("The Hugging Face Inference API timed out.") from exc
        except OverloadedError as exc:
            raise LLMRateLimitError("The Hugging Face Inference API is overloaded right now.") from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                raise LLMRateLimitError("Hugging Face Inference API rate limit reached.") from exc
            if status in (502, 503, 504):
                raise LLMRateLimitError(f"The Hugging Face Inference API is unavailable (HTTP {status}).") from exc
            raise LLMError(f"Hugging Face Inference API error (HTTP {status}): {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Network error calling the Hugging Face Inference API: {exc}") from exc
        return response.choices[0].message.content or ""


def build_llm(reporting_cfg: dict, device: str | None = None, hf_token: str | None = None) -> ChatLLM:
    llm_cfg = reporting_cfg["llm"]
    backend = reporting_cfg["backend"]
    if backend == "local":
        return LocalLLM(llm_cfg["model_id"], device=device, max_new_tokens=llm_cfg["max_new_tokens"],
                        temperature=llm_cfg["temperature"])
    if backend == "hf_api":
        return HFApiLLM(llm_cfg["api_model_id"], token=hf_token, timeout_s=llm_cfg["timeout_s"],
                        max_new_tokens=llm_cfg["max_new_tokens"], temperature=llm_cfg["temperature"])
    raise ValueError(f"no LLM for reporting.backend={backend!r}")
