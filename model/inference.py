"""
Model inference for Tiny Therapist Companion.
Supports both transformers and llama.cpp backends.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
GGUF_MODEL_ID = "bartowski/Llama-3.1-Nemotron-Nano-4B-v1.1-GGUF"

_device = None
_model = None
_tokenizer = None


def load_model(backend="transformers", quantization=None):
    """Load the model. backend: 'transformers' or 'llama_cpp'."""
    global _model, _tokenizer, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "llama_cpp":
        from llama_cpp import Llama
        _model = Llama.from_pretrained(
            repo_id=GGUF_MODEL_ID,
            filename="*Q4_K_M.gguf",
        )
        _tokenizer = None
        return _model, _tokenizer

    # Transformers backend
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _tokenizer.pad_token_id = _tokenizer.eos_token_id

    return _model, _tokenizer


def generate_response(model, tokenizer, message, history, system_prompt, max_new_tokens=512):
    """Generate a response using the loaded model."""
    if tokenizer is None:
        # llama_cpp backend
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})
        messages.append({"role": "user", "content": message})

        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
        )
        return response["choices"][0]["message"]["content"]

    # Transformers backend
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h.get("user", "")})
        messages.append({"role": "assistant", "content": h.get("assistant", "")})
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response