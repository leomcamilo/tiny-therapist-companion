# Tiny Therapist Companion — Field Notes

**Build Small Hackathon 2026 · HuggingFace × Gradio**

*How we built a 4B-parameter emotional wellness companion that runs 100% locally*

---

## Day 1-2 (Jun 5-6): Choosing the Project

We started by listing ideas against the hackathon criteria. The key insight from reading the judging rubric:

- **Track 1 (Backyard AI)**: "Solve a real problem for someone you actually know" — the person must have REALLY USED the app
- **Track 2 (Adventure in Thousand Token Wood)**: "Build something delightful that wouldn't exist without AI" — AI must be load-bearing, not a chatbot skin

We picked **Tiny Therapist Companion** — a small model (4B params) fine-tuned for empathetic listening, gratitude practice, and guided journaling. It targets both tracks because:

1. Real problem: mental wellness tools are expensive and inaccessible
2. AI is core: without the model generating empathetic, context-aware responses, the app is just a journal template

**Model choice:** NVIDIA Llama-3.1-Nemotron-Nano-4B-v1.1 — small enough to run locally, strong enough for conversation.

---

## Day 3 (Jun 7): Architecture Decisions

We made three key decisions early:

### 1. llama.cpp as default backend (Llama Champion bonus 🦙)

Instead of loading the full model via HuggingFace transformers (~8GB RAM), we use a Q4_K_M quantized GGUF model via `llama-cpp-python`. This means:
- ~3GB RAM instead of ~8GB
- Runs on the free CPU tier of HuggingFace Spaces
- First inference is slower (model download), but subsequent runs are fast

The code supports both backends via `BACKEND` env var:
```bash
BACKEND=llama_cpp python app.py   # default
BACKEND=transformers python app.py  # fallback
```

### 2. Lazy model loading (Off the Grid bonus 🔌)

We don't load the model on import — we load it on first request. This makes the Space cold start faster:
```python
_lock = threading.Lock()
_model = None
_loaded = False

def ensure_model_loaded():
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        # load model...
        _loaded = True
```

Thread-safe singleton with double-checked locking. No cloud APIs, no external calls.

### 3. Three modes, three prompts

Instead of one generic chatbot, we built three distinct interaction modes:
- **Talk**: Free-form empathetic conversation
- **Gratitude**: Guided gratitude practice (3 good things, savoring, gratitude letters)
- **Journal**: Structured journaling with prompts and pattern recognition

Each mode has its own system prompt, optimized for a 4B model (concise, under 150 words per response).

---

## Day 4 (Jun 8): Custom Interface (Off-Brand bonus 🎨)

The default Gradio look screams "hackathon project." We wanted something that felt like a wellness app.

### What we built:
- **Color palette**: Lavender/sage/cream gradient background — calming, not clinical
- **Typography**: Nunito (Google Font) — rounded, friendly
- **Breathing animation**: The 🧠 icon pulses slowly (4s cycle) — it breathes with you
- **Pill selectors**: Mode buttons are rounded pills, not radio buttons
- **Rounded chat bubbles**: User messages in violet, bot messages in white with soft borders
- **Hidden footer**: No "Built with Gradio" branding
- **Crisis resources**: Prominent disclaimer with CVV 188 (Brazil) and 988 (US) always visible

The CSS is ~200 lines of custom overrides on top of Gradio's Soft theme. Key lesson: Gradio 5's `gr.Blocks(css=...)` makes this much easier than in v4.

---

## Day 5 (Jun 9): Fine-Tuning Pipeline (Well-Tuned bonus 🎯)

We built the full QLoRA fine-tuning pipeline with Unsloth:

### Dataset
- 52 training examples + 33 validation
- Split across 3 modes: Talk (empathetic listening), Gratitude (guided practice), Journal (prompts + reflection)
- All in English, responses under 150 words
- Includes crisis intervention examples where the model suggests professional help

### Training config
- **Base model**: Nemotron-Nano-4B-v1.1
- **Method**: QLoRA (4-bit quantization + LoRA rank 16)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Learning rate**: 2e-5, cosine scheduler
- **Epochs**: 3

The training script supports `--push_to_hub` to publish the fine-tuned model directly to HuggingFace, earning the Well-Tuned bonus.

---

## Day 6 (Jun 10): Agent Traces (Sharing is Caring bonus 📡)

We added agent trace logging — every conversation is optionally recorded to a JSONL file with:
- Mode (Talk/Gratitude/Journal)
- Timestamp
- Backend used
- Anonymized message exchange

Users opt in via a checkbox in the UI. Traces can be uploaded to a HuggingFace dataset:

```bash
python scripts/upload_traces.py --repo-id leomcamilo/tiny-therapist-traces
```

This satisfies the Sharing is Caring bonus quest — making agent behavior transparent and reproducible.

---

## Lessons Learned

### What worked
- **llama.cpp for deployment**: Q4_K_M quantization means the model runs on free CPU tier. Game changer for hackathons.
- **Concise prompts for small models**: Shorter system prompts = less token waste = better responses from 4B models.
- **Three modes instead of one**: Makes the app feel purposeful rather than "another chatbot."
- **Lazy loading**: Cold start is always painful on free tiers. Delaying model load until first request helps.

### What we'd improve
- **Dataset size**: 52 training examples is minimal. Ideally 500+ for a real fine-tune.
- **Evaluation**: No quantitative eval of fine-tuned vs base model yet.
- **Latency on CPU**: First response on free CPU tier takes ~10-15 seconds. Acceptable for a hackathon, needs optimization for production.
- **Multilingual**: All prompts are English-only. PT-BR would be a natural next step for the Brazilian market.

---

## Bonus Quests Earned

| Quest | How |
|---|---|
| 🔌 **Off the Grid** | Zero cloud APIs. Model runs 100% locally via llama.cpp. No API keys needed. |
| 🦙 **Llama Champion** | Model runs via llama-cpp-python with GGUF quantization (Q4_K_M). |
| 🎨 **Off-Brand** | Custom CSS: lavender/sage gradient, Nunito font, breathing animation, pill selectors, rounded bubbles. |
| 📡 **Sharing is Caring** | Agent traces logged to JSONL, uploadable as HF dataset. Users opt-in via checkbox. |
| 📓 **Field Notes** | This document. |
| 🎯 **Well-Tuned** | Fine-tuning pipeline ready. Model to be published on HF Hub after training. |

---

## Day 7 (Jun 12): Polish & UX Improvements

### System prompts overhaul
Rewrote all three prompts with more specific, directive language optimized for 4B models:

**Before:** Generic "Rules:" lists with vague guidance like "Listen, reflect, ask open-ended questions"
**After:** Each prompt now includes:
- A clear first-action directive (e.g., "Listen first, then reflect back what you heard")
- Explicit behavioral constraints ("Ask ONE open-ended question per response")
- Response length guidance ("2-4 sentences" instead of "under 150 words")
- Specific crisis intervention phrasing ("I care about you. Please reach out to...")
- Mode-specific opening questions (Gratitude: "What's one small thing that went well today?")

Key insight: 4B models need **explicit step-by-step instructions** rather than abstract rules. "Ask ONE follow-up question per response" > "Ask one question at a time."

### UI/UX Polish (Off-Brand bonus)

1. **Welcome suggestions** -- 4 clickable prompt buttons that fill the input box. Reduces friction for first-time users.

2. **Mode-aware placeholders** -- Input box placeholder changes based on selected mode.

3. **Typing indicator CSS** -- Pulsing dot animation ready for streaming.

4. **Enhanced CSS** -- Hover effects, custom scrollbar, backdrop-filter blur, prefers-reduced-motion, better mobile breakpoints, pill buttons, container max-width, subtle shadows.

5. **Accessibility** -- prefers-reduced-motion disables all animations.

### Dockerfile optimization
- Multi-stage build: builder + runtime. Cuts ~300MB from final image.
- Added HEALTHCHECK for HF Spaces.
- Runtime only needs libopenblas0.

### Bug fixes
- bubble_full_width=False in Chatbot for readability on wider screens

---

## Try It

```bash
git clone https://github.com/leomcamilo/tiny-therapist-companion.git
cd tiny-therapist-companion
pip install -r requirements.txt
BACKEND=llama_cpp python app.py
```

Open [http://localhost:7860](http://localhost:7860) and start talking.

---

*Built with ❤️ by [Leo Camilo](https://github.com/leomcamilo) for the Build Small Hackathon 2026.*