"""
Tiny Therapist Companion — Gradio App
Build Small Hackathon 2026

Uses Gradio 5 Blocks API with llama.cpp backend priority.
Custom CSS for Off-Brand bonus quest.
Agent tracing for Sharing is Caring bonus quest.
Welcome suggestions for delightful UX.
"""

import os
import json
import time
import gradio as gr
from model.inference import generate_response
from model.prompts import SYSTEM_PROMPT, GRATITUDE_PROMPT, JOURNAL_PROMPT

BACKEND = os.environ.get("BACKEND", "llama_cpp")

# ─── Agent Trace Logging — Sharing is Caring bonus quest ──────────────────────

TRACE_DIR = os.environ.get("TRACE_DIR", "/tmp/tiny-therapist-traces")
TRACE_ENABLED = os.environ.get("TRACE_ENABLED", "true").lower() == "true"


def log_trace(user_msg: str, bot_response: str, mode: str, metadata: dict | None = None):
    """Log an agent trace to a JSONL file for sharing on HuggingFace Hub."""
    if not TRACE_ENABLED:
        return

    trace = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "user_message": user_msg,
        "bot_response": bot_response,
        "backend": BACKEND,
        "model": os.environ.get(
            "GGUF_MODEL_ID",
            os.environ.get("MODEL_ID", "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"),
        ),
        "metadata": metadata or {},
    }

    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_file = os.path.join(TRACE_DIR, "traces.jsonl")
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


# ─── Mode metadata ────────────────────────────────────────────────────────────

MODE_INFO = {
    "Talk": {
        "icon": "\U0001f5e3\ufe0f",
        "prompt": SYSTEM_PROMPT,
        "greeting": "Hey there \U0001f49c I'm here to listen. How are you feeling today?",
        "placeholder": "What's on your mind?",
    },
    "Gratitude": {
        "icon": "\U0001f64f",
        "prompt": GRATITUDE_PROMPT,
        "greeting": "Let's find some light today \u2728 What's one small thing that went well?",
        "placeholder": "Something good happened today\u2026",
    },
    "Journal": {
        "icon": "\U0001f4d3",
        "prompt": JOURNAL_PROMPT,
        "greeting": "Ready to write it out? \U0001f4d3 Tell me what's swirling in your head.",
        "placeholder": "Start writing your thoughts\u2026",
    },
}

WELCOME_SUGGESTIONS = [
    "I'm feeling stressed",
    "I want to practice gratitude",
    "Help me organize my thoughts",
    "I need someone to listen",
]


# ─── Custom CSS — Off-Brand bonus quest ─────────────────────────────────────

CUSTOM_CSS = """
/* ── Base ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

:root {
  --bg-cream: #faf7f2;
  --bg-lavender: #e8dff5;
  --bg-sage: #c8d6c3;
  --text-dark: #3a3a3a;
  --text-muted: #6b6b6b;
  --text-light: #9a9a9a;
  --accent-violet: #7c5cbf;
  --accent-violet-light: #a78bda;
  --accent-violet-pale: #e8dff5;
  --accent-warm: #d4a373;
  --card-bg: rgba(255, 255, 255, 0.75);
  --border-soft: rgba(124, 92, 191, 0.15);
  --shadow-soft: 0 2px 12px rgba(124, 92, 191, 0.08);
}

body {
  font-family: 'Nunito', sans-serif !important;
  background: linear-gradient(135deg, var(--bg-cream) 0%, var(--bg-lavender) 50%, var(--bg-sage) 100%) !important;
  background-attachment: fixed !important;
  color: var(--text-dark) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent-violet-light); border-radius: 3px; }

/* ── Header ──────────────────────────────────────────────────── */
.app-header {
  text-align: center;
  padding: 1.5rem 1rem 0.5rem;
}
.app-header h1 {
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: var(--accent-violet) !important;
  margin-bottom: 0.15rem !important;
  letter-spacing: -0.02em;
}
.app-header .subtitle {
  color: var(--text-muted) !important;
  font-size: 0.95rem !important;
  font-weight: 400 !important;
}

/* Breathing animation */
@keyframes breathe {
  0%, 100% { transform: scale(1); opacity: 0.85; }
  50% { transform: scale(1.1); opacity: 1; }
}
.breathing-icon {
  display: inline-block;
  animation: breathe 4s ease-in-out infinite;
  font-size: 2.5rem;
}

/* ── Mode selector — pill style ──────────────────────────────── */
.mode-row {
  display: flex !important;
  justify-content: center;
  gap: 0.5rem;
  margin: 0.75rem 0;
}
.mode-row label {
  border-radius: 9999px !important;
  padding: 0.5rem 1.5rem !important;
  border: 2px solid var(--accent-violet-light) !important;
  background: var(--card-bg) !important;
  color: var(--text-dark) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  cursor: pointer;
  transition: all 0.25s ease;
  user-select: none;
  backdrop-filter: blur(4px);
}
.mode-row label:hover {
  border-color: var(--accent-violet) !important;
  background: var(--accent-violet-pale) !important;
}
.mode-row label:has(input:checked),
.mode-row label.selected {
  background: var(--accent-violet) !important;
  color: #fff !important;
  border-color: var(--accent-violet) !important;
  box-shadow: 0 2px 8px rgba(124, 92, 191, 0.3);
}

/* ── Welcome suggestions ─────────────────────────────────────── */
.welcome-suggestions {
  display: flex !important;
  flex-wrap: wrap !important;
  justify-content: center !important;
  gap: 0.5rem !important;
  margin: 0.75rem 0 0.5rem !important;
}
.suggestion-btn button,
button.suggestion-btn {
  background: rgba(255, 255, 255, 0.85) !important;
  border: 1.5px solid var(--accent-violet-light) !important;
  border-radius: 9999px !important;
  color: var(--accent-violet) !important;
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  padding: 0.35rem 0.9rem !important;
  min-width: 0 !important;
  cursor: pointer;
  transition: all 0.2s ease;
}
.suggestion-btn button:hover,
button.suggestion-btn:hover {
  background: var(--accent-violet) !important;
  color: #fff !important;
  border-color: var(--accent-violet) !important;
  transform: translateY(-1px);
  box-shadow: 0 3px 8px rgba(124, 92, 191, 0.25);
}

/* ── Chat area ───────────────────────────────────────────────── */
.chat-area {
  border-radius: 1rem !important;
  border: 1px solid var(--border-soft) !important;
  background: var(--card-bg) !important;
  backdrop-filter: blur(8px);
  box-shadow: var(--shadow-soft);
}
.message.user {
  background: var(--accent-violet) !important;
  color: #fff !important;
  border-radius: 1.25rem 1.25rem 0.25rem !important;
  margin-left: 2rem !important;
}
.message.bot, .message.assistant {
  background: rgba(255,255,255,0.92) !important;
  color: var(--text-dark) !important;
  border-radius: 1.25rem 1.25rem 1.25rem 0.25rem !important;
  margin-right: 2rem !important;
  border: 1px solid var(--border-soft) !important;
}

/* ── Input area ──────────────────────────────────────────────── */
.input-area textarea {
  border-radius: 1rem !important;
  border: 2px solid var(--border-soft) !important;
  background: rgba(255,255,255,0.9) !important;
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.95rem !important;
  min-height: 52px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.input-area textarea:focus {
  border-color: var(--accent-violet) !important;
  box-shadow: 0 0 0 3px rgba(124, 92, 191, 0.15) !important;
}
.input-area textarea::placeholder {
  color: var(--text-light) !important;
}

/* ── Disclaimer banner ───────────────────────────────────────── */
.disclaimer {
  background: linear-gradient(90deg, rgba(255,183,77,0.12), rgba(255,138,101,0.12));
  border: 1px solid rgba(255,152,0,0.25);
  border-radius: 0.75rem;
  padding: 0.75rem 1rem;
  text-align: center;
  font-size: 0.82rem;
  color: var(--text-muted);
  margin: 0.5rem 0 0.25rem;
  line-height: 1.5;
}
.disclaimer strong {
  color: #c62828;
}

/* ── Trace toggle ────────────────────────────────────────────── */
.trace-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin: 0.25rem 0;
  font-size: 0.78rem;
  color: var(--text-light);
}

/* ── Footer ──────────────────────────────────────────────────── */
.app-footer {
  text-align: center;
  font-size: 0.78rem;
  color: var(--text-light);
  padding: 0.75rem 0 0.5rem;
  border-top: 1px solid var(--border-soft);
  margin-top: 0.5rem;
}
.app-footer a {
  color: var(--accent-violet);
  text-decoration: none;
  font-weight: 600;
}
.app-footer a:hover {
  text-decoration: underline;
}

/* ── Hide default Gradio footer ─────────────────────────────── */
footer.svelte-1rjn5q2,
footer.svelte-1pp78pt,
#components-__gradio_footer__,
.gradio-container > footer {
  display: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.primary-btn button,
button.primary {
  background: var(--accent-violet) !important;
  color: #fff !important;
  border-radius: 9999px !important;
  font-weight: 600 !important;
  padding: 0.5rem 1.5rem !important;
  transition: all 0.2s ease !important;
  font-family: 'Nunito', sans-serif !important;
}
.primary-btn button:hover,
button.primary:hover {
  background: var(--accent-violet-light) !important;
  transform: translateY(-1px);
  box-shadow: 0 3px 8px rgba(124, 92, 191, 0.3);
}
.secondary-btn button,
button.secondary {
  background: rgba(255,255,255,0.85) !important;
  color: var(--text-dark) !important;
  border: 1.5px solid var(--border-soft) !important;
  border-radius: 9999px !important;
  font-weight: 600 !important;
  padding: 0.5rem 1.5rem !important;
  font-family: 'Nunito', sans-serif !important;
  transition: all 0.2s ease !important;
}
.secondary-btn button:hover,
button.secondary:hover {
  background: var(--accent-violet-pale) !important;
  border-color: var(--accent-violet-light) !important;
}

/* ── Gradio container polish ──────────────────────────────────── */
.gradio-container {
  max-width: 720px !important;
  margin: 0 auto !important;
}
.contain {
  max-width: 720px !important;
}

/* ── Responsive ───────────────────────────────────────────────── */
@media (max-width: 640px) {
  .app-header h1 { font-size: 1.5rem !important; }
  .breathing-icon { font-size: 2rem; }
  .mode-row label { padding: 0.35rem 1rem !important; font-size: 0.8rem !important; }
  .welcome-suggestions { gap: 0.35rem !important; }
  .suggestion-btn button, button.suggestion-btn { padding: 0.3rem 0.75rem !important; font-size: 0.78rem !important; }
  .message.user { margin-left: 0.5rem !important; }
  .message.bot, .message.assistant { margin-right: 0.5rem !important; }
}

/* ── Reduced motion preference ─────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  .breathing-icon { animation: none; }
  .suggestion-btn button:hover, button.suggestion-btn:hover { transform: none; }
  .primary-btn button:hover, button.primary:hover { transform: none; }
}
"""


# ─── Chat handler ──────────────────────────────────────────────────────────────

MODE_PROMPTS = {
    "Talk": SYSTEM_PROMPT,
    "Gratitude": GRATITUDE_PROMPT,
    "Journal": JOURNAL_PROMPT,
}


def respond(message: str, history: list, mode: str):
    """Handle a chat message. History is list of gr.ChatMessage dicts."""
    system_prompt = MODE_PROMPTS.get(mode, SYSTEM_PROMPT)
    response = generate_response(
        message=message,
        history=history,
        system_prompt=system_prompt,
    )
    # Log agent trace for Sharing is Caring bonus quest
    log_trace(message, response, mode, metadata={"system_prompt_mode": mode})
    return response


# ─── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    css=CUSTOM_CSS,
    title="Tiny Therapist Companion",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="pink",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Nunito"),
    ),
) as app:

    # Header
    gr.HTML(
        """
        <div class="app-header">
            <span class="breathing-icon">\U0001f9e0</span>
            <h1>Tiny Therapist Companion</h1>
            <p class="subtitle">A safe space to reflect, journal, and practice gratitude</p>
        </div>
        """
    )

    # Mode selector
    mode = gr.Radio(
        choices=["Talk", "Gratitude", "Journal"],
        value="Talk",
        label="",
        show_label=False,
        elem_classes=["mode-row"],
    )

    # Disclaimer
    gr.HTML(
        """
        <div class="disclaimer">
            \u26a0\ufe0f <strong>This is NOT a substitute for professional therapy.</strong><br>
            If you're in crisis: \U0001f1e7\U0001f1f7 <strong>CVV 188</strong> | cvv.org.br &nbsp;\u00b7&nbsp; \U0001f1fa\U0001f1f8 <strong>988</strong> Suicide & Crisis Lifeline
        </div>
        """
    )

    # Welcome suggestions — quick-start buttons
    with gr.Row(elem_classes=["welcome-suggestions"]):
        suggestion_btns = []
        for suggestion_text in WELCOME_SUGGESTIONS:
            btn = gr.Button(suggestion_text, elem_classes=["suggestion-btn"])
            suggestion_btns.append((btn, suggestion_text))

    # Chat — using Chatbot with type="messages"
    chatbot = gr.Chatbot(
        type="messages",
        elem_classes=["chat-area"],
        height=450,
        show_copy_button=True,
        avatar_images=(None, "\U0001f9e0"),
        placeholder="Start a conversation\u2026 How are you feeling today?",
        bubble_full_width=False,
    )

    msg_input = gr.Textbox(
        placeholder="What's on your mind?",
        show_label=False,
        elem_classes=["input-area"],
        scale=4,
    )

    with gr.Row():
        send_btn = gr.Button("Send \u2728", variant="primary", elem_classes=["primary-btn"])
        clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["secondary-btn"])

    # Agent trace toggle — Sharing is Caring bonus quest
    trace_checkbox = gr.Checkbox(
        value=TRACE_ENABLED,
        label="\U0001f4e1 Share anonymized traces for research",
        visible=True,
        elem_classes=["trace-row"],
    )

    # Footer
    gr.HTML(
        """
        <div class="app-footer">
            Built for <a href="https://huggingface.co/build-small-hackathon" target="_blank"><strong>Build Small Hackathon 2026</strong></a> &middot;
            Powered by <a href="https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1" target="_blank">Nemotron-Nano-4B</a> &middot;
            <a href="https://github.com/leomcamilo/tiny-therapist-companion" target="_blank">GitHub</a> &middot;
            \U0001f4d3 <a href="FIELD_NOTES.md" target="_blank">Field Notes</a>
        </div>
        """
    )

    # ─── Event handlers ──────────────────────────────────────────────

    def chat_fn(user_msg: str, chat_history: list, mode_val: str, trace_on: bool):
        """Full chat cycle: add user msg, generate, add bot response."""
        global TRACE_ENABLED
        TRACE_ENABLED = trace_on

        if not user_msg.strip():
            return chat_history or [], ""

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_msg})

        bot_response = respond(user_msg, chat_history[:-1], mode_val)
        chat_history.append({"role": "assistant", "content": bot_response})

        return chat_history, ""

    def clear_chat():
        return [], ""

    def on_mode_change(new_mode: str):
        """Update placeholder when mode changes."""
        info = MODE_INFO.get(new_mode, MODE_INFO["Talk"])
        return gr.Textbox(placeholder=info["placeholder"])

    send_btn.click(
        fn=chat_fn,
        inputs=[msg_input, chatbot, mode, trace_checkbox],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=chat_fn,
        inputs=[msg_input, chatbot, mode, trace_checkbox],
        outputs=[chatbot, msg_input],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_input],
    )

    # When mode changes, update placeholder text
    mode.change(
        fn=on_mode_change,
        inputs=[mode],
        outputs=[msg_input],
    )

    # Wire suggestion buttons to fill input
    for btn, suggestion_text in suggestion_btns:
        btn.click(
            fn=lambda s=suggestion_text: s,
            inputs=[],
            outputs=[msg_input],
        )


if __name__ == "__main__":
    app.launch()