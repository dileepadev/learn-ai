---
title: Introduction to Gradio
description: Learn Gradio — the Python library for rapidly building interactive ML demos and web applications. Covers the Interface API for quick demos, the Blocks API for complex multi-component layouts, custom components, sharing on Hugging Face Spaces, and best practices for showcasing models to non-technical audiences.
---

**Gradio** is a Python library that lets you build interactive web demos for machine learning models in minutes, with no JavaScript required. A researcher can wrap any Python function — a model inference call, a data processing pipeline, a generative AI endpoint — in a Gradio interface and share it as a public URL or embed it in a web page.

Gradio is particularly well-integrated with the Hugging Face ecosystem: every Hugging Face Space runs Gradio or Streamlit, and the Hub hosts tens of thousands of Gradio demos for everything from image generation to code completion to medical diagnosis. Its combination of rapid prototyping, rich component library, and zero-friction sharing makes it the dominant tool for ML demo development.

## Quick Start: The Interface API

The simplest Gradio usage wraps a Python function with typed inputs and outputs:

```python
import gradio as gr
import numpy as np
from PIL import Image

def classify_image(img: np.ndarray) -> dict[str, float]:
    """
    Placeholder image classifier — replace with your actual model.
    Returns a dict mapping class name → confidence score (0–1).
    """
    # Simulate model predictions
    classes = ["cat", "dog", "bird", "car", "person"]
    scores = np.random.dirichlet(np.ones(len(classes)))
    return dict(zip(classes, scores.tolist()))


# Single-line demo: inputs and outputs inferred from type hints
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),           # webcam or file upload → numpy array
    outputs=gr.Label(num_top_classes=3),  # bar chart of top predictions
    title="Image Classifier",
    description="Upload an image to classify it.",
    examples=["cat.jpg", "dog.jpg"],   # pre-loaded example inputs
    theme=gr.themes.Soft()
)

demo.launch(
    share=True,       # generates a public gradio.live URL
    server_port=7860
)
```

## Core Input/Output Components

Gradio provides over 30 built-in components covering all common ML modalities:

```python
import gradio as gr

# Text
text_input = gr.Textbox(label="Input text", lines=3, placeholder="Type here...")
text_output = gr.Textbox(label="Output", interactive=False)

# Numbers and sliders
temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7,
                         step=0.1, label="Temperature")
max_tokens = gr.Number(value=256, minimum=1, maximum=4096,
                        label="Max tokens", precision=0)

# Dropdowns and checkboxes
model_choice = gr.Dropdown(
    choices=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
    value="gpt-4o",
    label="Model"
)
streaming = gr.Checkbox(value=True, label="Stream output")

# Media
image_input = gr.Image(type="pil", label="Upload image")
audio_input = gr.Audio(type="numpy", label="Record or upload audio",
                        sources=["microphone", "upload"])
video_input = gr.Video(label="Upload video")

# File I/O
file_upload = gr.File(file_types=[".pdf", ".txt", ".csv"],
                       label="Upload document")
file_download = gr.DownloadButton(label="Download result")

# Structured data
dataframe_output = gr.Dataframe(
    headers=["Name", "Score", "Label"],
    datatype=["str", "number", "str"],
    interactive=False
)

# Rich display
json_output = gr.JSON(label="Model response")
markdown_output = gr.Markdown()
html_output = gr.HTML()
plot_output = gr.Plot()   # matplotlib, plotly, altair figures
```

## The Blocks API: Complex Layouts

`gr.Blocks` gives full control over layout, event handling, and multi-step interactions:

```python
import gradio as gr
import openai
from typing import Iterator

client = openai.OpenAI()

def stream_chat_response(
    message: str,
    history: list[list[str]],
    system_prompt: str,
    temperature: float,
    max_tokens: int
) -> Iterator[str]:
    """Stream a chat response token by token."""
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    partial = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial


with gr.Blocks(title="Chat Interface", theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# AI Chat Assistant")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                bubble_full_width=False,
                show_copy_button=True
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type a message...",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            system_prompt = gr.Textbox(
                label="System prompt",
                value="You are a helpful assistant.",
                lines=4
            )
            temperature = gr.Slider(0.0, 2.0, value=0.7,
                                     step=0.05, label="Temperature")
            max_tokens = gr.Slider(64, 4096, value=512,
                                    step=64, label="Max tokens")
            clear_btn = gr.Button("Clear conversation", variant="secondary")

    def respond(message, chat_history, sys_prompt, temp, max_tok):
        # Generator function enables token-by-token streaming in UI
        bot_message = ""
        for partial in stream_chat_response(message, chat_history,
                                             sys_prompt, temp, max_tok):
            bot_message = partial
            yield chat_history + [[message, bot_message]], ""
        # Final yield with complete message
        yield chat_history + [[message, bot_message]], ""

    send_btn.click(
        respond,
        inputs=[msg, chatbot, system_prompt, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    msg.submit(   # also trigger on Enter key
        respond,
        inputs=[msg, chatbot, system_prompt, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    clear_btn.click(lambda: [], outputs=chatbot)

demo.launch()
```

## Multi-Tab Interfaces

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Modal AI Demo")

    with gr.Tab("Text Generation"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=3)
            output = gr.Textbox(label="Generated text", lines=8, interactive=False)
        gen_btn = gr.Button("Generate", variant="primary")
        gen_btn.click(fn=generate_text, inputs=prompt, outputs=output)

    with gr.Tab("Image Generation"):
        with gr.Row():
            img_prompt = gr.Textbox(label="Image prompt")
            img_style = gr.Dropdown(
                choices=["realistic", "anime", "oil painting", "sketch"],
                value="realistic", label="Style"
            )
        gen_img_btn = gr.Button("Generate image", variant="primary")
        image_out = gr.Image(label="Generated image", width=512, height=512)
        gen_img_btn.click(fn=generate_image,
                           inputs=[img_prompt, img_style], outputs=image_out)

    with gr.Tab("Document Q&A"):
        file_in = gr.File(file_types=[".pdf", ".txt"], label="Upload document")
        question = gr.Textbox(label="Question about the document")
        answer = gr.Textbox(label="Answer", interactive=False)
        qa_btn = gr.Button("Ask", variant="primary")
        qa_btn.click(fn=answer_question, inputs=[file_in, question], outputs=answer)

demo.launch()
```

## State Management

```python
import gradio as gr

def add_to_history(new_item: str, history: list) -> tuple[list, list, str]:
    """Add item to session-specific history."""
    updated_history = history + [new_item]
    display = "\n".join(f"{i+1}. {item}" for i, item in enumerate(updated_history))
    return updated_history, updated_history, display


with gr.Blocks() as demo:
    # gr.State: persists across interactions within a session (server-side)
    history_state = gr.State(default_value=[])

    item_input = gr.Textbox(label="Add item")
    add_btn = gr.Button("Add")
    history_display = gr.Textbox(label="History", interactive=False)

    add_btn.click(
        add_to_history,
        inputs=[item_input, history_state],
        outputs=[history_state, history_state, history_display]
    )
```

## Deploying on Hugging Face Spaces

Every Hugging Face Space is a Gradio (or Streamlit) app. Deployment is a single git push:

```python
# app.py — entry point for HF Spaces
import gradio as gr
import torch
from transformers import pipeline

# Load model (cached by HF infrastructure)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_sentiment(text: str) -> dict:
    result = classifier(text)[0]
    return {result["label"]: result["score"],
            ("POSITIVE" if result["label"] == "NEGATIVE" else "NEGATIVE"): 1 - result["score"]}

demo = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(label="Review text", placeholder="Enter text to analyze..."),
    outputs=gr.Label(num_top_classes=2),
    title="Sentiment Classifier",
    description="Classify text as POSITIVE or NEGATIVE sentiment.",
    examples=["I loved this movie!", "The service was terrible.", "It was okay."],
    cache_examples=True   # pre-compute example outputs to speed up cold start
)

# HF Spaces reads launch() parameters from README.md hardware config
demo.launch()
```

The `requirements.txt` in the Space repo lists dependencies; hardware (CPU/T4/A10G) is configured in the Space settings UI or `README.md` metadata block.

## Gradio vs. Alternatives

| Tool | Best for | Requires | Sharing |
|---|---|---|---|
| **Gradio** | ML demos, rapid prototyping | Python only | HF Spaces, gradio.live |
| **Streamlit** | Data dashboards, app-like UIs | Python only | Streamlit Cloud |
| **Panel** | Complex dashboards, notebooks | Python only | Self-host |
| **Dash** | Production analytics apps | Python + config | Self-host |
| **FastAPI + React** | Full custom web app | Python + JS | Self-host |

Gradio's combination of near-zero setup, deep HuggingFace integration, and growing component library makes it the fastest path from a trained model to a shareable interactive demo — a capability that has fundamentally changed how ML researchers communicate results and gather feedback from users.
