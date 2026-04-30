---
title: LLM Routing and Model Cascades
description: Learn how to intelligently route queries to the most appropriate LLM based on complexity, cost, and capability — covering classifier-based routing, LLM cascade strategies, FrugalGPT patterns, embedding similarity routing, and production architecture for cost-quality optimization.
---

**LLM routing** is the practice of directing each query to the most appropriate model — rather than sending every request to the same, typically most expensive, model. A query asking "what is the capital of France?" does not warrant the same model (or cost) as a query requiring multi-step legal reasoning. **Model cascades** extend this idea: try a cheaper model first, evaluate the output quality, and escalate to a more capable model only when needed.

Together, routing and cascades address one of the most practical problems in production LLM deployment: **the cost-quality-latency triangle**. Using a powerful model for everything is safe but expensive; using a cheap model for everything is fast and cheap but degrades quality. Routing makes it possible to optimize all three simultaneously.

## The Economic Case

A frontier model API call (e.g., GPT-4o, Claude Opus) costs roughly 10–100× more than a capable mid-tier model. In high-volume production systems, 60–80% of queries are often straightforward enough for a cheaper model. Routing can therefore reduce inference costs by 50–70% with minimal quality degradation — a finding demonstrated in the **FrugalGPT** paper (Chen et al., Stanford, 2023).

$$\text{cost savings} = \sum_{q} \mathbf{1}[\text{query } q \text{ routes to cheap model}] \times (c_{\text{expensive}} - c_{\text{cheap}})$$

## Routing Strategies

### Classifier-Based Routing

Train a lightweight classifier (BERT-small, DistilBERT, or even a linear model on TF-IDF features) to predict which model tier a query should go to:

```python
from transformers import pipeline
from dataclasses import dataclass
from enum import Enum

class ModelTier(Enum):
    FAST = "gpt-4o-mini"          # ~$0.15/1M tokens
    BALANCED = "gpt-4o"           # ~$2.50/1M tokens  
    POWERFUL = "claude-opus-4"    # ~$15/1M tokens

@dataclass
class RoutingDecision:
    tier: ModelTier
    confidence: float
    reasoning: str

# Lightweight classifier trained on labeled query-tier pairs
router = pipeline(
    "text-classification",
    model="your-org/query-complexity-classifier",  # fine-tuned DistilBERT
    return_all_scores=True
)

TIER_MAP = {
    "simple": ModelTier.FAST,
    "moderate": ModelTier.BALANCED,
    "complex": ModelTier.POWERFUL
}

def classify_and_route(query: str) -> RoutingDecision:
    scores = router(query)[0]
    best = max(scores, key=lambda x: x["score"])
    tier = TIER_MAP[best["label"]]
    return RoutingDecision(
        tier=tier,
        confidence=best["score"],
        reasoning=f"Classified as '{best['label']}' with {best['score']:.2%} confidence"
    )
```

Training data for the classifier is bootstrapped by labeling historical queries with the minimum model tier needed to produce an acceptable answer — evaluated by a judge model or human annotators.

### Embedding Similarity Routing

Route based on semantic similarity to prototypical queries for each model tier:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Representative examples for each tier (manually curated or bootstrapped)
tier_prototypes = {
    ModelTier.FAST: [
        "What is the capital of France?",
        "Convert 50 fahrenheit to celsius",
        "Translate 'hello' to Spanish",
        "What year was Python created?",
    ],
    ModelTier.BALANCED: [
        "Write a professional email declining a meeting",
        "Summarize this 500-word article",
        "Explain the difference between TCP and UDP",
        "Review this SQL query for performance issues",
    ],
    ModelTier.POWERFUL: [
        "Analyze the constitutional implications of this legal clause",
        "Debug this complex async race condition in Rust",
        "Design a distributed system for 10M concurrent users",
        "Write a research proposal for novel cancer treatment approaches",
    ]
}

# Precompute prototype embeddings
prototype_embeddings = {
    tier: encoder.encode(examples, normalize_embeddings=True)
    for tier, examples in tier_prototypes.items()
}

def route_by_similarity(query: str) -> RoutingDecision:
    query_emb = encoder.encode([query], normalize_embeddings=True)[0]
    
    tier_scores = {}
    for tier, proto_embs in prototype_embeddings.items():
        # Mean cosine similarity to tier prototypes
        similarities = np.dot(proto_embs, query_emb)
        tier_scores[tier] = float(similarities.mean())
    
    best_tier = max(tier_scores, key=tier_scores.get)
    return RoutingDecision(
        tier=best_tier,
        confidence=tier_scores[best_tier],
        reasoning=f"Highest similarity to {best_tier.name} prototypes"
    )
```

### Rule-Based Routing

For structured applications where query types are well-defined, explicit rules are often simpler and more reliable than learned classifiers:

```python
import re

def rule_based_router(query: str, context: dict) -> ModelTier:
    """
    Route based on explicit signals: query length, keywords,
    user tier, task type flags.
    """
    # Long queries likely need more capable models
    if len(query.split()) > 300:
        return ModelTier.POWERFUL
    
    # Code review and debugging: use balanced or powerful
    code_keywords = ["debug", "error", "stack trace", "segfault", "race condition"]
    if any(kw in query.lower() for kw in code_keywords):
        return ModelTier.BALANCED
    
    # Legal, medical, financial reasoning: always use powerful
    high_stakes = ["legal", "medical", "diagnosis", "contract", "compliance"]
    if any(kw in query.lower() for kw in high_stakes):
        return ModelTier.POWERFUL
    
    # Simple factual lookups: fast model
    factual_patterns = [r"^what is", r"^who is", r"^when was", r"^convert \d+"]
    if any(re.match(p, query.lower()) for p in factual_patterns):
        return ModelTier.FAST
    
    # User tier override (premium users get better models)
    if context.get("user_tier") == "enterprise":
        return ModelTier.BALANCED
    
    # Default: balanced
    return ModelTier.BALANCED
```

## Model Cascades

In a **cascade**, the output from a cheap model is evaluated by a quality judge before deciding whether to escalate. This is fundamentally different from routing — routing decides *before* generation; cascading decides *after* seeing the cheap model's output.

```python
import openai

client = openai.OpenAI()

def cascade_query(query: str, max_escalations: int = 2) -> dict:
    """
    FrugalGPT-style cascade: try cheapest model first,
    evaluate quality, escalate if needed.
    """
    model_cascade = [
        "gpt-4o-mini",   # cheapest, fastest
        "gpt-4o",        # mid-tier
        "o3",            # most capable (most expensive)
    ]
    
    history = []
    
    for i, model in enumerate(model_cascade[:max_escalations + 1]):
        # Generate with current model
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=0
        )
        answer = response.choices[0].message.content
        
        history.append({"model": model, "answer": answer})
        
        # Don't evaluate after last model
        if i == len(model_cascade) - 1:
            break
        
        # Quality judgment: is this answer good enough?
        judge_prompt = f"""Query: {query}

Answer: {answer}

Is this answer complete, accurate, and sufficient? 
Respond with exactly: SUFFICIENT or ESCALATE"""
        
        judgment = client.chat.completions.create(
            model="gpt-4o-mini",  # cheap judge
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        ).choices[0].message.content.strip()
        
        if "SUFFICIENT" in judgment:
            return {"answer": answer, "model_used": model, "escalations": i, "history": history}
    
    # Return last answer after all escalations
    return {"answer": history[-1]["answer"], "model_used": model_cascade[i], 
            "escalations": i, "history": history}
```

## The LLM Router as a Model

**RouteLLM** (Ong et al., 2024) trains a specialized router model that learns to predict which model will produce a better response — without requiring labels. The key insight: generate responses from both models on training queries, then use a preference model to label which response is better. The router learns from these preference labels.

```python
# Conceptual: RouteLLM-style router training
# (actual implementation would use their library)

# 1. Collect training queries
# 2. Generate responses from cheap_model and strong_model
# 3. Label with preference model: which response is better?
# 4. Train binary classifier: strong_model_preferred (yes/no)
# 5. At inference: threshold the probability
#    - p(strong_model_preferred) > threshold → route to strong model
#    - threshold tunable to hit quality/cost tradeoffs

def routellm_style_router(query: str, threshold: float = 0.5) -> ModelTier:
    """
    Threshold on router's predicted probability that
    strong model is preferred over weak model.
    """
    # Router is a lightweight model (e.g., BERT fine-tuned)
    p_strong_preferred = router_model.predict_proba([query])[0][1]
    
    if p_strong_preferred > threshold:
        return ModelTier.POWERFUL
    else:
        return ModelTier.FAST
```

Increasing the threshold → more queries to cheap model (lower cost, some quality loss). Decreasing the threshold → more queries to strong model (higher cost, higher quality). This single knob enables **Pareto-optimal cost-quality tradeoffs**.

## Production Architecture

```python
import asyncio
from typing import AsyncGenerator

class LLMRouter:
    def __init__(self, router_strategy="classifier"):
        self.strategy = router_strategy
        self._cost_tracker = {"total_tokens": 0, "total_cost": 0.0}
    
    async def route_and_generate(
        self, 
        query: str, 
        context: dict = None
    ) -> AsyncGenerator[str, None]:
        """Route query and stream response from appropriate model."""
        
        # Select routing strategy
        if self.strategy == "classifier":
            decision = classify_and_route(query)
        elif self.strategy == "similarity":
            decision = route_by_similarity(query)
        else:
            decision = RoutingDecision(
                tier=rule_based_router(query, context or {}),
                confidence=1.0, reasoning="Rule-based"
            )
        
        model = decision.tier.value
        
        # Stream response
        async with openai.AsyncOpenAI() as async_client:
            stream = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
```

## Routing vs. Mixture of Experts vs. Mixture of Agents

These are often confused but are distinct:

| Concept | Decision point | Unit of routing | Who decides |
|---|---|---|---|
| **LLM Routing** | Pre-generation | Entire query → one model | Router model/rules |
| **Mixture of Experts** | Within model forward pass | Token → expert FFN | Learned gating |
| **Mixture of Agents** | Post-generation | Multiple answers → aggregated | Aggregator LLM |
| **Cascade** | Post-generation | Answer quality → next model | Quality judge |

Routing is the only approach that operates entirely outside the model — making it applicable to any combination of black-box API models. It is the most practical and deployable approach for production cost optimization today.
