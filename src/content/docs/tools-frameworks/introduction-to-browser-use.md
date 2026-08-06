---
title: "Introduction to browser-use: Agentic Web Automation Framework"
description: Learn how browser-use enables LLM agents to interact with web pages, parse visual DOM trees, execute actions, and perform multi-step web task automation.
---

**`browser-use`** is an open-source Python library designed to make web browsers accessible to LLM agents. Built on top of Playwright and vision-capable language models, `browser-use` allows AI agents to interact with websites naturally — clicking elements, filling forms, extracting tabular data, and navigating complex multi-step web applications just like a human user.

## Why `browser-use`?

Traditional web scraping and web automation tools (like raw Playwright, Selenium, or BeautifulSoup) rely on rigid XPath selectors or static CSS rules. When a website redesigns its layout or changes dynamic class names, automation scripts break immediately.

`browser-use` solves this by giving LLMs **direct visual and structural perception** of the DOM:
- **DOM Tree Simplification:** Translates complex HTML into a clean, minimal tree containing only interactable elements.
- **Visual Bounding Boxes:** Highlights interactable elements on screen with numbered visual tags so vision-capable models can verify target click coordinates accurately.
- **Self-Healing Automation:** If a click fails or opens an unexpected modal, the agent senses the state change and adapts dynamically.

## Architecture Overview

```
+-------------------+
|  LLM Agent Policy | <--- (State: Screenshot + Interactive DOM)
+-------------------+
          |
   Action Decision (e.g. click_element(idx=5), input_text(idx=2, "San Francisco"))
          v
+-------------------+
| Browser-Use Controller |
+-------------------+
          |
   Playwright API Commands
          v
+-------------------+
| Chromium / Browser|
+-------------------+
```

## Quickstart Code Example

Installing `browser-use` via pip:

```bash
pip install browser-use playwright
rf -rf ~/.cache/ms-playwright && playwright install
```

Automating a flight search or product research task in Python:

```python
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

async def main():
    # Initialize vision-capable LLM model
    llm = ChatOpenAI(model="gpt-4o")

    # Create agent with natural language goal
    agent = Agent(
        task="Navigate to news.ycombinator.com, find the top story about AI, click into it, and extract the main takeaway.",
        llm=llm,
    )

    # Run execution loop
    history = await agent.run()
    
    print("\nFinal Result:")
    print(history.final_result())

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Capabilities & Features

### 1. Vision & Element Tagging
`browser-use` injects an interactive visual overlay into the browser. Every clickable button, link, input box, and dropdown is tagged with a highlighted numerical index. The agent receives both the cropped screenshot and the index map, eliminating coordinate misclicks.

### 2. Multi-Tab & Popup Management
The framework automatically tracks tab handles, new window popups, and alert dialogs, maintaining state history across multi-tab workflows.

### 3. Agentic Memory & Trajectory Logging
Every step — including action inputs, DOM states, console outputs, and screenshots — is logged into structured JSON trajectories for debugging and evaluation.

### 4. Custom Tool Extensions
You can register custom Python functions (e.g., saving output to Postgres, triggering webhook alerts, or solving CAPTCHAs) directly into the agent's controller:

```python
from browser_use import Controller

controller = Controller()

@controller.action("Save product price to database")
def save_price(product_name: str, price: float):
    # Custom business logic
    print(f"Database Record Created: {product_name} - ${price}")
```

## Comparison: Web Scraping vs. Agentic Browser Automation

| Feature | Traditional Scraper (Selenium / BS4) | LLM Agent Automation (`browser-use`) |
|---|---|---|
| **Selector Dependency** | Static XPath / CSS Selectors | Semantic Perception & Vision Overlay |
| **Handling Layout Changes** | Fails on DOM edits | Self-heals & finds target visually |
| **Authentication & CAPTCHAs** | Complex manual scripting | Handles interactive prompts & session state |
| **Multi-Step Logic** | Hardcoded conditional flows | Natural Language Instruction Driven |
| **Speed** | Extremely Fast (Raw HTTP) | Medium (LLM Inference Loop Latency) |

## Best Practices for Production Deployment

1. **Session Persistence:** Store browser cookies and local storage profiles to avoid repeated login prompts across runs.
2. **Stealth & Anti-Detection:** Configure Playwright with real user-agent strings and viewport dimensions to prevent bot detection blocks.
3. **Headless Execution:** Run browsers in headless mode on server instances with virtual framebuffers (`xvfb`).

## Summary

`browser-use` transforms web automation by replacing brittle selector scripts with intelligent visual agents. Whether building market intelligence agents, automated QA testing workflows, or enterprise web task assistants, `browser-use` provides a robust foundation for browser interaction.

## Further Reading

- browser-use GitHub Repository: `github.com/browser-use/browser-use`
- Playwright Python Documentation: `playwright.dev/python`
- WebArena Benchmark for Browser Agents
