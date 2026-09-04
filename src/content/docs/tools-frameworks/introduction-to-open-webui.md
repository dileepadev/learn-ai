---
title: Introduction to Open WebUI
description: Learn how to set up Open WebUI, the self-hosted extensible chat interface for Ollama, OpenAI-compatible APIs, local RAG pipelines, and multi-user role-based access control.
---

While local AI execution runtimes (like Ollama, vLLM, and llama.cpp) have made running open-weight LLMs straightforward, developers and enterprise teams need a polished, modern, and secure graphical user interface.

**Open WebUI** (formerly Ollama WebUI) is an extensible, self-hosted AI interface that delivers a **complete ChatGPT-like web experience** running 100% locally or inside a private enterprise VPC. Built with a responsive SvelteKit frontend and a Python FastAPI backend, Open WebUI supports multi-model chat, integrated document RAG, web search grounding, voice synthesis, and multi-user role-based access control (RBAC).

---

## Architecture Overview

```
                                [ Web Browser / Mobile Device ]
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Open WebUI (Docker Container / SvelteKit + FastAPI)                                         │
│                                                                                             │
│  ┌────────────────────────┐  ┌─────────────────────────┐  ┌──────────────────────────────┐ │
│  │ Multi-User Auth & RBAC │  │ Native RAG Engine       │  │ Pipelines / Middleware       │ │
│  │ Admin, User, Quotas    │  │ ChromaDB, BM25, Hybrid   │  │ Custom python pre/post      │ │
│  │ OAuth / LDAP / SSO     │  │ PDF, DOCX, CSV parsing  │  │ filters, moderation, routing │ │
│  └────────────────────────┘  └─────────────────────────┘  └──────────────────────────────┘ │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ (HTTP REST / Streaming API)
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Execution Backends                                                                          │
│ • Local Ollama Instance: `http://localhost:11434`                                           │
│ • Remote OpenAI-Compatible Endpoints: vLLM, TGI, LM Studio, Azure, Anthropic                 │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities

### 1. Dual-Model Chat & Side-by-Side Comparison
Users can prompt two models concurrently (e.g., LLaMA-3.1-8B vs. Mistral-NeMo) and view responses side-by-side in real time, making model evaluation, speed benchmarking, and output comparison seamless.

### 2. Built-In Document RAG Pipeline
Open WebUI includes an out-of-the-box **Retrieval-Augmented Generation (RAG)** engine:
- Users can drag-and-drop PDFs, Word documents, text files, and spreadsheets directly into the chat box.
- The backend chunks documents, embeds them using a local embedding model (or ChromaDB), and executes hybrid search.
- When answering questions, the model references exact page citations and excerpts.

### 3. Real-Time Web Search Grounding
Connects seamlessly to open-source search engines (SearXNG) or commercial APIs (Brave Search, Google Search). When a user asks about current events, Open WebUI executes a search, scrapes candidate pages, injects relevant snippets into the prompt, and supplies footnotes.

### 4. Custom Modelfiles & Community Hub
Users can create specialized custom AI personas (Modelfiles) with customized system prompts, temperature settings, and attached document knowledge bases, or download community-curated assistants directly from the Open WebUI Community Hub.

### 5. Multi-User Administration & Role-Based Access Control (RBAC)
Unlike single-user desktop apps, Open WebUI was engineered for team collaboration:
- Administrators can manage user registrations, assign roles (`admin` or `user`), and configure per-user model permissions.
- Integrates with enterprise Single Sign-On (SSO) via OAuth2, OpenID Connect, and LDAP.

---

## Getting Started: Docker Deployment

### 1. Running Alongside a Local Ollama Instance

If Ollama is already running on your host machine:

```bash
docker run -d -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui-data:/app/backend/data \
    --name open-webui \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

Navigate to `http://localhost:3000` in your web browser. The first account created automatically receives administrative privileges.

### 2. All-in-One Bundled Deployment with GPU Acceleration

If you do not have Ollama installed, you can launch a single container that bundles both Ollama and Open WebUI with NVIDIA GPU support:

```bash
docker run -d -p 3000:8080 --gpus all \
    -v ollama-models:/root/.ollama \
    -v open-webui-data:/app/backend/data \
    --name open-webui \
    --restart always \
    ghcr.io/open-webui/open-webui:ollama
```

---

## Connecting External OpenAI-Compatible Endpoints

Open WebUI is not limited to Ollama; it connects to any OpenAI-compatible serving engine:

1. In Open WebUI, navigate to **Settings > Admin Settings > Connections**.
2. Under **OpenAI API**, set the target base URL:
   - For vLLM: `http://vllm-server:8000/v1`
   - For TGI: `http://tgi-server:8080/v1`
   - For OpenRouter: `https://openrouter.ai/api/v1`
3. Enter the corresponding API Key. All models hosted on the remote cluster will automatically populate in the model selection dropdown.

---

## Custom Extensibility: The Pipelines Framework

Open WebUI features **Pipelines**, a lightweight modular framework allowing developers to inject custom Python logic before or after generation:

```python
"""
title: Custom PII Redaction Filter Pipeline
description: Redacts email addresses and phone numbers before routing to LLM
"""
import re
from typing import List, Dict, Any

class Pipeline:
    def __init__(self):
        pass

    async def inlet(self, body: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        # Pre-process prompt before it reaches the language model
        messages = body.get("messages", [])
        for message in messages:
            # Mask email addresses
            message["content"] = re.sub(
                r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                "[REDACTED_EMAIL]",
                message["content"]
            )
        return body

    async def outlet(self, body: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        # Post-process response before sending to user interface
        return body
```

---

## Key Takeaways

- Open WebUI provides a polished, self-hosted web interface for local LLMs, rivaling commercial consumer platforms like ChatGPT.
- Out-of-the-box document RAG, web search grounding, and dual-model comparison simplify multi-modal workflows.
- Built-in multi-user management, OAuth integration, and the Python Pipelines framework make it production-ready for enterprise team deployments.
