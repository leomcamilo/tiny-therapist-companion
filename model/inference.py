"""
Model inference for Tiny Therapist Companion.
Supports llama.cpp (default) and transformers backends.
Lazy loading with thread-safe singleton pattern.
"""

import os
import threading
from typing import Optional

BACKEND = os.environ.get("BACKEND", "llama_cpp")

MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
GGUF_MODEL_ID = "bartowski/Llama-3.1-Nemotron-Nano-4B-v1.1-GGUF"

_lock = threading.Lock()
_model = None
_tokenizer = None
_loaded = False


def _load_llama_cpp():
    """Load model via llama-cpp-python (GGUF quantized)."""
    global _model, _tokenizer
    from llama_cpp import Llama

    _model = Llama.from_pretrained(
        repo_id=GGUF_MODEL_ID,
        filename="*Q4_K_M.gguf",
        n_ctx=2048,
        verbose=False,
    )
    _tokenizer = None  # not used with llama_cpp


def _load_transformers():
    """Load model via HuggingFace transformers."""
    global _model, _tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _tokenizer.pad_token_id = _tokenizer.eos_token_id


def ensure_model_loaded():
    """Lazy-load model on first request. Thread-safe."""
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        if BACKEND == "llama_cpp":
            _load_llama_cpp()
        else:
            _load_transformers()
        _loaded = True


def generate_response(message: str, history: list, system_prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate a response.

    Args:
        message: The user's current message.
        history: List of dicts with 'role' and 'content' keys
                 (Gradio 5 gr.ChatMessage format).
        system_prompt: System prompt to use.
        max_new_tokens: Max tokens to generate.

    Returns:
        The assistant's response text.
    """
    ensure_model_loaded()

    # Build messages list from history + current message
    messages = [{"role": "system", "content": system_prompt}]

    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    if BACKEND == "llama_cpp":
        return _generate_llama_cpp(messages, max_new_tokens)
    else:
        return _generate_transformers(messages, max_new_tokens)


def _generate_llama_cpp(messages: list, max_new_tokens: int) -> str:
    """Generate response using llama.cpp backend."""
    response = _model.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
    )
    return response["choices"][0]["message"]["content"]


def _generate_transformers(messages: list, max_new_tokens: int) -> str:
    """Generate response using transformers backend."""
    import torch

    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    outputs = _model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )

    response = _tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )
    return response