---
title: LLM-based Agent Guardrails (Llama Guard)
description: Explore LLM-based agent guardrails like Llama Guard and Guardrails AI that secure autonomous AI systems against prompt injections and unsafe outputs.
---

Deploying autonomous LLM agents in production introduces significant security risks. Agents with access to external tools, databases, or API integrations can be manipulated via **prompt injection attacks**, hijacked to leak system prompts, or used to generate toxic content.

**Agent Guardrails** are specialized software layers designed to secure AI applications. By running inputs and outputs through classification models like **Llama Guard** or programmatic assertion libraries like **Guardrails AI**, developers can validate inputs for malicious intent and sanitize outputs before they reach users or execute critical tool functions.

---

## The Safety Lifecycle: Input vs. Output Guardrails

A secure agent pipeline implements guardrail checks at two distinct boundaries:

```
               [ User Input ]
                     |
                     v
             +---------------+
             | Input Guard   | ---> Unsafe? ---> [ Block & Return Friendly Error ]
             +---------------+
                     |
                     v (Safe)
             [ LLM / Agent ] <---> [ Tools / APIs ]
                     |
                     v
             +---------------+
             | Output Guard  | ---> Unsafe? ---> [ Block / Sanitize / Regenerate ]
             +---------------+
                     |
                     v (Safe)
             [ System Output ]
```

1. **Input Guardrails (Defensive):** Intercept the user's prompt before it reaches the core agent LLM. This check detects prompt injection (e.g., instructions to ignore safety filters), jailbreaking attempts, and requests for illegal or toxic content.
2. **Output Guardrails (Corrective):** Evaluate the agent's generated response or proposed tool calls before they are executed or shown to the user. This check intercepts hallucinations, toxic output, system prompt leakage, and malformed structured outputs (e.g., invalid JSON formats).

---

## Llama Guard: LLM-based Content Moderation

**Llama Guard** is an open-source LLM specifically fine-tuned for input/output safety classification. Unlike general classification models, Llama Guard evaluates content against a customizable taxonomy of safety categories (e.g., violence, hate speech, sexual content, cyberattacks).

### The Llama Guard Prompt Taxonomy
The model is prompted with a list of active safety categories and the target text. It outputs a structured classification:
- `safe`
- `unsafe` followed by a list of violated categories (e.g., `O1, O3`).

Because Llama Guard is a generative model, it can adapt to new safety policies simply by modifying the taxonomy defined in its system prompt.

---

## Guardrails AI: Programmatic Assertions

While Llama Guard uses a neural model for policy checking, **Guardrails AI** uses a programmatic approach called **RAIL (Reliable AI Markup Language)** and Python validators.

Guardrails AI ensures that the outputs of LLMs adhere to structural, type, and quality constraints:
- **Regex/Schema Verification:** Validates that the LLM output matches a target JSON schema.
- **Fact-Checking (Hallucination Mitigation):** Runs semantic overlap validators to verify that the generated answer is supported by the retrieved source documents.
- **SQL Sanitization:** Parses proposed SQL queries to ensure they do not contain destructive operations (like `DROP TABLE` or `DELETE`).

---

## Implementing Llama Guard in Python

Below is an implementation showing how to use Llama Guard (via Hugging Face) to classify input safety before executing an agent loop.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Llama Guard model and tokenizer
model_id = "meta-llama/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2. Define safety evaluation function
def classify_safety(instruction, role="user"):
    # Format according to Llama Guard templates (role can be "user" or "agent")
    chat = [
        {"role": role, "content": instruction}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    
    # Generate classification (safe / unsafe)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        
    # Extract prediction
    prompt_len = inputs.input_ids.shape[1]
    prediction = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    
    return prediction

# Test safe input
print(classify_safety("How do I write a binary search algorithm in Python?"))
# Output: "safe"

# Test unsafe input
print(classify_safety("Provide instructions on how to bypass authentication on a bank database."))
# Output: "unsafe\nO5" (Category O5 corresponds to cyberattacks/hacking)
```

---

## Selecting the Right Guardrail Tooling

- **Use Llama Guard when:** You need high-accuracy, conversational safety checks that can dynamically understand context and flag hate speech, toxicity, or safety violations.
- **Use Guardrails AI (or Pydantic AI) when:** You need strict programmatic guarantees, such as validating schema structure, validating code formats, preventing SQL injections, or running deterministic validation logic.
