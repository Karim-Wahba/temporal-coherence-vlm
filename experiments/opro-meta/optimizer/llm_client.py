"""
optimizer/llm_client.py
-----------------------
Backend-agnostic text-LLM client. Three concrete backends:

  Qwen3LocalBackend   text-only Qwen3 model loaded with transformers
                      (kept as the default to mirror the rest of the project's
                       no-API-key, single-GPU-job preference)
  AnthropicBackend    Claude via the Anthropic SDK
  OpenAIBackend       GPT-4 family via the OpenAI SDK

Each backend exposes:
  generate(prompt: str, *, system: str | None = None) -> str
  generate_json(prompt: str, *, system: str | None = None) -> dict | None

A factory `make_client(backend, **kwargs)` builds the right one from config.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional


# ── helpers ────────────────────────────────────────────────────────────────────

def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models
    (DeepSeek-V4, R1-style, o1-style). Closes implicit blocks if generation
    was cut off before </think>."""
    # Closed block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Unclosed block (generation truncated mid-think): drop everything up to
    # the first '}' or '[' that could begin the JSON payload, falling back to
    # stripping the leading <think> tag only.
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        m = re.search(r"[\{\[]", text)
        if m:
            text = text[m.start():]
        else:
            text = re.sub(r"<think>", "", text, flags=re.IGNORECASE)
    return text


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _try_parse_json(text: str) -> Optional[dict]:
    text = _strip_thinking_tags(text)
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Greedy outermost {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# ── Generic HuggingFace local text-LLM backend ────────────────────────────────

class LocalHFBackend:
    """Text-only HuggingFace causal LM (e.g. Qwen3, DeepSeek-V4) via transformers.

    The factory exposes this under two names with different defaults:
      backend="qwen3"     -> default model_id Qwen/Qwen3-7B-Instruct
      backend="deepseek"  -> default model_id deepseek-ai/DeepSeek-V4-Pro
    Pass model_id explicitly to override.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
    ):
        print(f"[llm] loading text model: {model_id} on {device}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Qwen3 / DeepSeek-V4 default chat template adds a <think> block.
        # We disable it for the OPRO loop — it burns budget and the inner
        # loop only needs the final JSON.
        self.enable_thinking = enable_thinking

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        import torch
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        # Try the modern Qwen3 / R1-style chat template that accepts
        # enable_thinking; fall back gracefully if the tokenizer doesn't.
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-3),
            )
        trimmed = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(trimmed, skip_special_tokens=True)

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        return _try_parse_json(
            self.generate(prompt, system=system, max_new_tokens=max_new_tokens)
        )


# Backwards-compatible aliases. Both are the same generic backend.
Qwen3LocalBackend    = LocalHFBackend
DeepSeekLocalBackend = LocalHFBackend


# ── Anthropic backend ─────────────────────────────────────────────────────────

class AnthropicBackend:
    """Claude via the Anthropic Python SDK."""

    def __init__(
        self,
        model_id: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "AnthropicBackend requires the anthropic SDK: `pip install anthropic`"
            ) from e
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        kwargs = dict(
            model=self.model_id,
            max_tokens=max_new_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts)

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        return _try_parse_json(
            self.generate(prompt, system=system, max_new_tokens=max_new_tokens)
        )


# ── OpenAI backend ────────────────────────────────────────────────────────────

class OpenAIBackend:
    """GPT-4 family via the OpenAI Python SDK."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "OpenAIBackend requires the openai SDK: `pip install openai`"
            ) from e
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=msgs,
            max_tokens=max_new_tokens or self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        return _try_parse_json(
            self.generate(prompt, system=system, max_new_tokens=max_new_tokens)
        )


# ── Factory ───────────────────────────────────────────────────────────────────

_DEFAULT_MODEL_IDS = {
    "qwen3":    "Qwen/Qwen3-7B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
}


def make_client(backend: str, **kwargs):
    backend = backend.lower()
    if backend in ("qwen3", "qwen3-local"):
        kwargs.setdefault("model_id", _DEFAULT_MODEL_IDS["qwen3"])
        return LocalHFBackend(**kwargs)
    if backend in ("deepseek", "deepseek-v4", "deepseek-local"):
        kwargs.setdefault("model_id", _DEFAULT_MODEL_IDS["deepseek"])
        return LocalHFBackend(**kwargs)
    if backend in ("local", "hf"):
        # Generic local HF — caller must supply model_id
        if "model_id" not in kwargs:
            raise ValueError("backend='local' requires model_id=<huggingface repo>")
        return LocalHFBackend(**kwargs)
    if backend in ("anthropic", "claude"):
        return AnthropicBackend(**kwargs)
    if backend in ("openai", "gpt", "gpt-4"):
        return OpenAIBackend(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend}")
