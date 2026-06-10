"""
Tiny Therapist Companion — Gradio App
Build Small Hackathon 2026

Uses Gradio 5 Blocks API with llama.cpp backend priority.
Custom CSS for Off-Brand bonus quest.
"""

import os
import gradio as gr
from model.inference import generate_response
from model.prompts import SYSTEM_PROMPT, GRATITUDE_PROMPT, JOURNAL_PROMPT

BACKEND = os.environ.get("BACKEND", "llama_cpp")

# ─── Custom CSS — Off-Brand bonus quest ─────────────────────────────────────

CUSTOM_CSS = """
/* ── Base ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

:root {
  --bg-cream: #faf7f2;
  --bg-lavender: #e8dff5;
  --bg-sage: #c8d6c3;
  --text-dark: #3a3a3a;
  --text-muted: #6b6b6b;
  --accent-violet: #7c5cbf;
  --accent-violet-light: #a78bda;
  --accent-warm: #d4a373;
  --card-bg: rgba(255, 255, 255, 0.75);
  --border-soft: rgba(124, 92, 191, 0.15);
}

body {
  font-family: 'Nunito', sans-serif !important;
  background: linear-gradient(135deg, var(--bg-cream) 0%, var(--bg-lavender) 50%, var(--bg-sage) 100%) !important;
  background-attachment: fixed !important;
  color: var(--text-dark) !important;
}

/* ── Header ──────────────────────────────────────────────────── */
.app-header {
  text-align: center;
  padding: 1.5rem 1rem 0.5rem;
}
.app-header h1 {
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--accent-violet) !important;
  margin-bottom: 0.25rem !important;
}
.app-header p {
  color: var(--text-muted) !important;
  font-size: 0.95rem !important;
}

/* Breathing animation */
@keyframes breathe {
  0%, 100% { transform: scale(1); opacity: 0.85; }
  50% { transform: scale(1.08); opacity: 1; }
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
  padding: 0.45rem 1.4rem !important;
  border: 2px solid var(--accent-violet-light) !important;
  background: var(--card-bg) !important;
  color: var(--text-dark) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  cursor: pointer;
  transition: all 0.2s ease;
  user-select: none;
}
.mode-row label:has(input:checked),
.mode-row label.selected {
  background: var(--accent-violet) !important;
  color: #fff !important;
  border-color: var(--accent-violet) !important;
}

/* ── Chat area ───────────────────────────────────────────────── */
.chat-area {
  border-radius: 1rem !important;
  border: 1px solid var(--border-soft) !important;
  background: var(--card-bg) !important;
  backdrop-filter: blur(8px);
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
}
.input-area textarea:focus {
  border-color: var(--accent-violet) !important;
  box-shadow: 0 0 0 3px rgba(124, 92, 191, 0.15) !important;
}

/* ── Disclaimer banner ───────────────────────────────────────── */
.disclaimer {
  background: linear-gradient(90deg, rgba(255,183,77,0.15), rgba(255,138,101,0.15));
  border: 1px solid rgba(255,152,0,0.3);
  border-radius: 0.75rem;
  padding: 0.75rem 1rem;
  text-align: center;
  font-size: 0.85rem;
  color: var(--text-muted);
  margin: 0.5rem 0 0.25rem;
}
.disclaimer strong {
  color: #c62828;
}

/* ── Footer ───────────────────────────────────────────────────── */
.app-footer {
  text-align: center;
  font-size: 0.8rem;
  color: var(--text-muted);
  padding: 0.75rem 0 0.5rem;
  border-top: 1px solid var(--border-soft);
  margin-top: 0.5rem;
}

/* ── Hide default Gradio footer ──────────────────────────────── */
footer.svelte-1rjn5q2,
footer.svelte-1pp78pt,
#components-__gradio_footer__,
.gradio-container > footer {
  display: none !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.primary-btn button,
button.primary {
  background: var(--accent-violet) !important;
  color: #fff !important;
  border-radius: 9999px !important;
  font-weight: 600 !important;
  transition: background 0.2s ease;
}
.primary-btn button:hover,
button.primary:hover {
  background: var(--accent-violet-light) !important;
}

/* ── Responsive ──────────────────────────────────────────────── */
@media (max-width: 640px) {
  .app-header h1 { font-size: 1.5rem !important; }
  .breathing-icon { font-size: 2rem; }
  .mode-row label { padding: 0.35rem 1rem !important; font-size: 0.8rem !important; }
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
            <span class="breathing-icon">🧠</span>
            <h1>Tiny Therapist Companion</h1>
            <p>A safe space to reflect, journal, and practice gratitude</p>
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
            ⚠️ <strong>This is NOT a substitute for professional therapy.</strong><br>
            If you're in crisis: 🇧🇷 <strong>CVV 188</strong> | cvv.org.br &nbsp;·&nbsp; 🇺🇸 <strong>988</strong> Suicide & Crisis Lifeline
        </div>
        """
    )

    # Chat — using Chatbot with type="messages"
    chatbot = gr.Chatbot(
        type="messages",
        elem_classes=["chat-area"],
        height=450,
        show_copy_button=True,
        avatar_images=(None, "🧠"),
        placeholder="Start a conversation… How are you feeling today?",
    )

    msg_input = gr.Textbox(
        placeholder="Type your message…",
        show_label=False,
        elem_classes=["input-area"],
        scale=4,
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

    # Footer
    gr.HTML(
        """
        <div class="app-footer">
            Built for <strong>Build Small Hackathon 2026</strong> · Powered by Nemotron-Nano-4B
        </div>
        """
    )

    # ─── Event handlers ──────────────────────────────────────────────

    def submit_message(user_msg: str, chat_history: list):
        """Process a user message and return updated history."""
        if not user_msg.strip():
            return "", chat_history or []

        chat_history = chat_history or []
        # Append user message
        chat_history.append({"role": "user", "content": user_msg})

        # Generate response
        mode_value = mode.value if hasattr(mode, "value") else "Talk"
        # Re-read mode from the radio component isn't directly accessible here,
        # so we pass it via a wrapper
        return "", chat_history  # will be handled by .then()

    # Chain: user msg → clear input → generate → append response
    def chat_fn(user_msg: str, chat_history: list, mode_val: str):
        """Full chat cycle: add user msg, generate, add bot response."""
        if not user_msg.strip():
            return chat_history or [], ""

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_msg})

        bot_response = respond(user_msg, chat_history[:-1], mode_val)
        chat_history.append({"role": "assistant", "content": bot_response})

        return chat_history, ""

    def clear_chat():
        return [], ""

    send_btn.click(
        fn=chat_fn,
        inputs=[msg_input, chatbot, mode],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=chat_fn,
        inputs=[msg_input, chatbot, mode],
        outputs=[chatbot, msg_input],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_input],
    )


if __name__ == "__main__":
    app.launch()