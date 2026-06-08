"""
Tiny Therapist Companion — Gradio App
Build Small Hackathon 2026
"""

import gradio as gr
from model.inference import load_model, generate_response
from model.prompts import SYSTEM_PROMPT, GRATITUDE_PROMPT, JOURNAL_PROMPT

# Load model on startup
model, tokenizer = load_model()


def chat_message(message, history, mode):
    """Handle chat interaction with mode switching."""
    if mode == "Gratitude":
        system_prompt = GRATITUDE_PROMPT
    elif mode == "Journal":
        system_prompt = JOURNAL_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT

    response = generate_response(model, tokenizer, message, history, system_prompt)
    return response


# Custom theme — Off-Brand bonus quest
theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="pink",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(theme=theme, title="Tiny Therapist Companion") as app:
    gr.Markdown(
        """
        # 🧠 Tiny Therapist Companion
        A safe space to reflect, journal, and practice gratitude.
        """
    )

    with gr.Row():
        mode = gr.Radio(
            choices=["Talk", "Gratitude", "Journal"],
            value="Talk",
            label="Mode",
            info="Choose how you'd like to interact",
        )

    gr.ChatInterface(
        fn=chat_message,
        additional_inputs=[mode],
        type="messages",
        title="",
        description="⚠️ This is NOT a substitute for professional therapy. If you're in crisis, please contact a professional.",
    )

if __name__ == "__main__":
    app.launch()