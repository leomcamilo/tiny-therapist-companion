# 🧠 Tiny Therapist Companion

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live-green?logo=huggingface)](https://huggingface.co/spaces/leomcamilo/tiny-therapist-companion)
[![License](https://img.shields.io/badge/License-NVIDIA%20Open%20Model-blue)](./LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](./Dockerfile)

> _A small language model companion for empathetic listening, journaling, and gratitude practices — running 100% locally, no cloud APIs required._

Built for the **[Build Small Hackathon](https://huggingface.co/build-small-hackathon) 2026** — HuggingFace × Gradio.

---

## 🎯 What It Does

Tiny Therapist Companion is an AI-powered emotional-wellness companion that runs entirely on your machine. Choose from three interaction modes:

| Mode | What happens |
|---|---|
| 🗣️ **Talk** | Free-form empathetic conversation — organise scattered thoughts, reflect on your day, or just be heard |
| 📝 **Gratitude** | Guided gratitude practice — the companion helps you notice the good in your day with exercises like "3 good things" |
| 💭 **Journal** | Structured journaling — prompts, follow-up questions, and pattern recognition across your entries |

**This is NOT a replacement for professional therapy.** It's a companion tool for reflection and emotional wellness practices.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│              Gradio Web UI (port 7860)       │
│  ┌─────────┐  ┌───────────┐  ┌────────────┐  │
│  │  Talk   │  │ Gratitude │  │  Journal   │  │
│  └────┬────┘  └─────┬─────┘  └─────┬──────┘  │
│       │             │              │          │
│       └─────────────┼──────────────┘          │
│                     ▼                         │
│           ┌──────────────────┐                │
│           │  Mode Router     │                │
│           │  (system prompt) │                │
│           └────────┬─────────┘                │
│                    ▼                          │
│     ┌──────────────────────────────┐          │
│     │     Inference Backend        │          │
│     │  ┌──────────┐ ┌───────────┐  │          │
│     │  │llama.cpp │ │transformers│  │          │
│     │  │(GGUF/Q4) │ │ (full)    │  │          │
│     │  └────┬─────┘ └─────┬─────┘  │          │
│     └───────┼──────────────┼───────┘          │
│             ▼              ▼                  │
│  ┌─────────────────────────────────────┐      │
│  │  Nemotron-Nano-4B-v1.1 (4B params) │      │
│  │  nvidia/Llama-3.1-Nemotron-Nano-4B  │      │
│  └─────────────────────────────────────┘      │
└──────────────────────────────────────────────┘
         ▲              ▲
         │   CPU-only   │
         │  (free tier) │
         └──────────────┘
```

**Backend selection** is controlled by the `BACKEND` env var:
- `llama_cpp` (default) — loads a Q4_K_M quantised GGUF model via `llama-cpp-python`. Fast on CPU, low RAM (~3 GB).
- `transformers` — loads the full model via HuggingFace `transformers` + `torch`. Needs ~8 GB RAM.

---

## 🚀 Quick Start

### Option A — HuggingFace Space (one click)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-spaces-sm.svg)](https://huggingface.co/spaces/leomcamilo/tiny-therapist-companion)

Just click the badge above. The Space builds from the Dockerfile and runs on the free CPU tier.

### Option B — Local clone

```bash
# Clone
git clone https://github.com/leomcamilo/tiny-therapist-companion.git
cd tiny-therapist-companion

# (Recommended) Create a virtual env
python -m venv .venv && source .venv/bin/activate

# Install — llama-cpp-python needs CMake + a C++ compiler
# On macOS:  brew install cmake
# On Ubuntu:  sudo apt install build-essential cmake
pip install -r requirements.txt

# Run with llama.cpp backend (default)
BACKEND=llama_cpp python app.py

# …or run with transformers backend
BACKEND=transformers python app.py
```

### Option C — Docker

```bash
docker build -t tiny-therapist .
docker run -p 7860:7860 -e BACKEND=llama_cpp tiny-therapist
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `BACKEND` | `llama_cpp` | Inference backend: `llama_cpp` or `transformers` |
| `MODEL_ID` | `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | HuggingFace model ID (transformers backend) |
| `GGUF_MODEL_ID` | `bartowski/Llama-3.1-Nemotron-Nano-4B-v1.1-GGUF` | GGUF repo (llama_cpp backend) |
| `GGUF_FILENAME` | `*Q4_K_M.gguf` | GGUF file glob (llama_cpp backend) |
| `HF_SPACE` | `yes` | Set to `yes` on HuggingFace Spaces |

On HuggingFace Spaces, set these as **Space Secrets** in the Settings tab.

---

## 🎨 Fine-Tuning

Want to adapt the model to your own conversational style or domain?

### 1. Prepare your dataset

Create a JSONL file in the conversational (messages) format:

```jsonl
{"messages": [{"role": "system", "content": "You are an empathetic companion..."}, {"role": "user", "content": "I feel anxious today"}, {"role": "assistant", "content": "That sounds tough. Can you tell me more about what's making you feel anxious?"}]}
```

Place it at `fine-tuning/dataset/train.jsonl` (and optionally `val.jsonl`).

### 2. Configure

Edit `fine-tuning/config.yaml` to adjust epochs, learning rate, batch size, etc.

### 3. Train

```bash
# Using Unsloth (recommended — fast, memory-efficient)
pip install unsloth
python fine-tuning/train.py

# Or with standard HuggingFace Trainer
python fine-tuning/train.py --backend transformers
```

### 4. Push to Hub

After training, push your LoRA adapter or merged model:

```bash
huggingface-cli login
# Then follow the script's output to upload
```

---

## 🏆 Bonus Quests

| Quest | How we earned it |
|---|---|
| 🦙 **Llama Champion** | Model runs via `llama.cpp` / `llama-cpp-python` with GGUF quantisation |
| 🎨 **Off-Brand** | Custom Gradio Soft theme (violet + pink + Inter font), not the default look |
| 🔌 **Off the Grid** | Zero cloud API calls — model downloads once, inference is 100% local |

---

## 📦 Project Structure

```
tiny-therapist-companion/
├── app.py                  # Gradio application
├── model/
│   ├── __init__.py
│   ├── inference.py        # Model loading + dual-backend inference
│   └── prompts.py          # System prompts (Talk, Gratitude, Journal)
├── fine-tuning/
│   ├── config.yaml         # Training hyper-parameters
│   ├── train.py             # Fine-tuning script
│   └── dataset/             # Training data (JSONL)
├── requirements.txt        # Python dependencies
├── Dockerfile               # HF Spaces Docker build
├── .env.example             # Environment variable template
└── README.md
```

---

## 🧠 About the Model

**NVIDIA Llama-3.1-Nemotron-Nano-4B-v1.1** is a 4-billion-parameter model from NVIDIA's Nemotron family, built on Llama 3.1 architecture. It's designed for efficient inference on consumer hardware while maintaining strong conversational ability.

- **Parameters:** 4B
- **Context window:** 4,096 tokens
- **Quantisation:** Q4_K_M (GGUF) for llama.cpp — ~2.5 GB download
- **License:** NVIDIA Open Model License + Llama 3.1 Community License

---

## ⚠️ Crisis Resources Disclaimer

**Tiny Therapist Companion is NOT a substitute for professional mental health services.** It is a wellness companion for reflection, gratitude, and journaling.

If you or someone you know is experiencing a mental health crisis, please reach out to professionals:

- 🇧🇷 **Brazil:** CVV — dial **188** or chat at [cvv.org.br](https://www.cvv.org.br)
- 🇺🇸 **USA:** 988 Suicide & Crisis Lifeline — call or text **988**
- 🇬🇧 **UK:** Samaritans — call **116 123**
- 🌍 **International:** [findahelpline.com](https://findahelpline.com)

---

## 📄 License

This project uses the **NVIDIA Open Model License** (following the base model) and the **Llama 3.1 Community License**. See the model cards for full terms:

- [Nemotron-Nano-4B license](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1)
- [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)

Code in this repository is released under the MIT License unless otherwise stated.

---

Built with ❤️ by [Leo Camilo](https://github.com/leomcamilo) for the **Build Small Hackathon 2026**.