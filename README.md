# 🧠 Tiny Therapist Companion

> _A small language model companion for empathetic listening, journaling, and gratitude practices._

Built for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon) 2025 — HuggingFace × Gradio.

## 🎯 What It Does

Tiny Therapist Companion is an AI-powered companion that helps you:

- 🗣️ **Organize your thoughts** — Talk through what's on your mind, the companion listens and reflects back
- 📝 **Practice gratitude** — Guided gratitude journaling that helps you notice the good in your day
- 💭 **Journal your feelings** — A safe space to process emotions without judgment

**This is NOT a replacement for professional therapy.** It's a companion tool for reflection and emotional wellness practices.

## 🏗️ Architecture

- **Base Model:** NVIDIA Llama-3.1-Nemotron-Nano-4B-v1.1
- **Fine-tuned for:** Empathetic conversation, reflective listening, journaling prompts
- **Interface:** Gradio app running as a HuggingFace Space
- **Local-first:** Designed to run on consumer hardware (single GPU)

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/leomcamilo/tiny-therapist-companion.git
cd tiny-therapist-companion

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

## 📦 Project Structure

```
tiny-therapist-companion/
├── app.py                  # Gradio application
├── model/
│   ├── inference.py        # Model loading and inference
│   └── prompts.py          # System prompts and templates
├── fine-tuning/
│   ├── dataset/            # Training data
│   ├── train.py            # Fine-tuning script
│   └── config.yaml        # Training configuration
├── requirements.txt
├── Dockerfile              # For HF Space deployment
└── README.md
```

## 🏆 Hackathon Tracks

- **Track 1 (Backyard AI):** Solves a real problem — emotional wellness and self-reflection for people who may not have access to therapy
- **Track 2 (Adventure in Thousand Token Wood):** A delightful conversational experience that wouldn't exist without AI as the core

### Bonus Quests

- 🔌 **Off the Grid** — Runs 100% local, no cloud API needed
- 🎯 **Well-Tuned** — Fine-tuned model published on HF Hub
- 🎨 **Off-Brand** — Custom Gradio frontend going beyond default look
- 🦙 **Llama Champion** — Model runs via llama.cpp

## ⚠️ Disclaimer

This tool is **not** a substitute for professional mental health services. If you're experiencing a mental health crisis, please contact a professional or crisis hotline:

- 🇧🇷 Brazil: CVV — 188 or chat at cvv.org.br
- 🇺🇸 US: 988 Suicide & Crisis Lifeline

## 📄 License

NVIDIA Open Model License (following base model) + Llama 3.1 Community License

---

Built with ❤️ by [Leo Camilo](https://github.com/leomcamilo) for the Build Small Hackathon 2026.