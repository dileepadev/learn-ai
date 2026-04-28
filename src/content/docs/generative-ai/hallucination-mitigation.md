---
title: Hallucination Mitigation in Large Language Models
description: Understand why large language models hallucinate and learn practical techniques to reduce fabricated facts — including retrieval augmentation, uncertainty estimation, self-consistency decoding, factuality training, and production-grade guardrails.
---

**Hallucination** in large language models (LLMs) refers to the generation of fluent, confident-sounding content that is factually incorrect, unsupported by sources, or entirely fabricated. Unlike a model that says "I don't know," a hallucinating model produces plausible-sounding but false information — inventing citations, misquoting statistics, fabricating events, or generating non-existent product names.

Hallucination is not a bug to be patched but an emergent property of how language models are trained: they learn to produce likely next tokens given context, not to verify claims against a ground truth. Reducing hallucination requires a combination of architectural choices, training strategies, inference techniques, and application-level safeguards — no single technique eliminates it entirely.

## Why LLMs Hallucinate

Understanding the root causes guides mitigation strategy:

**Parametric knowledge limitations**: LLMs store world knowledge in weights, which have a training cutoff and finite capacity. Facts the model never encountered, changed since training, or only appeared rarely are hallucination risks. A model trained on web data will hallucinate answers to questions whose answers are absent from the training corpus.

**Overconfident language modeling objective**: Standard training maximizes log-likelihood of the next token — rewarding fluent output regardless of factual accuracy. There is no explicit penalty for generating false statements confidently.

**Sycophancy and reward hacking**: RLHF training optimizes for human preference — and humans often prefer fluent, confident-sounding answers over hedged or uncertain ones, inadvertently reinforcing hallucination.

**Attention dilution in long contexts**: With long input contexts, attention becomes diluted — the model may attend poorly to critical evidence and fill gaps with plausible but incorrect content.

**Reasoning chain errors**: In chain-of-thought reasoning, an error in an early reasoning step compounds downstream, leading to confident but incorrect conclusions that are internally consistent with the flawed premise.

## Retrieval-Augmented Generation (RAG)

The most effective general-purpose hallucination mitigation is **Retrieval-Augmented Generation** — providing the model with relevant, verified external knowledge at inference time rather than relying on parametric memory.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Explicit grounding prompt: instruct the model to only use provided context
GROUNDED_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using ONLY the information provided in the context below.
If the context does not contain enough information to answer the question, 
say "I don't have enough information to answer this."
Do not use any external knowledge.

Context:
{context}

Question: {question}

Answer:"""
)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": GROUNDED_PROMPT}
)

response = qa_chain.invoke("What were the Q3 revenue figures?")
```

Key practices for RAG-based grounding:

- Use explicit instructions: tell the model to refuse if context is insufficient.
- Retrieve enough context (k=5-10 chunks) to cover the answer.
- Use reranking to prioritize the most relevant chunks.
- Return source citations to enable human verification.

## Self-Consistency Decoding

**Self-consistency** (Wang et al., 2022) samples multiple independent reasoning paths and takes the majority vote:

```python
import openai
from collections import Counter

def self_consistent_answer(question: str, num_samples: int = 10) -> str:
    """
    Generate multiple answers and return the most common one.
    Effective for factual questions with deterministic answers.
    """
    client = openai.OpenAI()
    answers = []
    
    for _ in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the question step by step, then state your final answer clearly."},
                {"role": "user", "content": question}
            ],
            temperature=0.7  # Non-zero temperature for diversity
        )
        # Extract the final answer from the response
        answer = extract_final_answer(response.choices[0].message.content)
        answers.append(answer)
    
    # Return most common answer
    most_common = Counter(answers).most_common(1)[0][0]
    confidence = Counter(answers).most_common(1)[0][1] / num_samples
    return most_common, confidence

answer, conf = self_consistent_answer("What is the capital of the country with the most UNESCO World Heritage Sites?")
print(f"Answer: {answer} (confidence: {conf:.0%})")
```

Self-consistency improves factual accuracy by approximately 10-20% on knowledge-intensive QA benchmarks. Low confidence scores (majority vote < 50%) signal uncertain questions that benefit from human review.

## Uncertainty Estimation and Selective Prediction

A model that knows when it doesn't know can abstain rather than hallucinate. Approaches to uncertainty estimation:

**Token-level probability thresholding**: Flag answers where the model assigns low probability to generated tokens — indicating low confidence in specific claims.

**Semantic entropy** (Farquhar et al., 2023): Sample multiple generations and compute the entropy over semantic equivalence classes (clustering generations by meaning rather than exact string match). High semantic entropy indicates the model generates contradictory answers — a reliable signal of hallucination risk.

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def semantic_entropy(question: str, model, n_samples: int = 20) -> float:
    """
    Estimate hallucination risk via semantic entropy.
    High entropy = model generates contradictory answers = likely uncertain.
    """
    # Sample multiple answers
    answers = [model.generate(question, temperature=1.0) for _ in range(n_samples)]
    
    # Embed answers and cluster by semantic similarity
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(answers)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.3,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    
    # Compute entropy over cluster distribution
    cluster_counts = np.bincount(labels)
    probs = cluster_counts / cluster_counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    return entropy

# Low entropy: model consistently gives same answer → likely correct
# High entropy: model gives many different answers → likely hallucinating
risk_score = semantic_entropy("What were the exact terms of the 1947 Indian Independence Act?", model)
```

## Factuality-Oriented Training

**Training-time** approaches that explicitly optimize for factual accuracy:

### Factuality Reinforcement Learning

**FactScore** and similar metrics decompose responses into atomic claims and verify each against a knowledge base. The factuality score can be used as a reward signal in RLHF:

- High-factuality responses receive high reward, reinforcing grounded generation.
- Low-factuality (hallucinated) responses receive low reward, discouraging fabrication.

This approach requires a reliable factuality evaluator — often another LLM prompted to check claims against retrieved sources.

### Direct Preference Optimization for Factuality

Curate preference datasets where:

- **Preferred**: Responses that accurately reflect source documents, include appropriate hedging, and cite uncertainty.
- **Rejected**: Responses that confidently assert falsehoods, over-claim certainty, or fabricate citations.

DPO training on such datasets shifts the model toward more calibrated, source-faithful generation.

### RLHF with Calibration Objectives

In addition to helpfulness rewards, incorporate **calibration rewards** that penalize confident wrong answers more than uncertain right answers. This discourages the overconfident generation that is a key hallucination failure mode.

## Chain-of-Verification (CoVe)

**CoVe** (Dhuliawala et al., 2023) reduces hallucination by having the model generate verification questions and independently answer them before finalizing a response:

1. **Draft**: Generate an initial response to the user question.
2. **Plan**: Generate a list of factual verification questions about claims in the draft.
3. **Verify**: Answer each verification question independently (without seeing the draft).
4. **Revise**: Revise the draft based on any inconsistencies found in verification.

```python
def chain_of_verification(question: str, llm) -> str:
    # Step 1: Draft initial response
    draft = llm.generate(f"Answer this question: {question}")
    
    # Step 2: Generate verification questions about the draft's claims
    verification_questions = llm.generate(
        f"Question: {question}\nDraft answer: {draft}\n\n"
        f"List 3-5 factual questions that would verify the key claims "
        f"in this answer. Output as a numbered list."
    )
    
    # Step 3: Answer each verification question independently
    verification_answers = llm.generate(
        f"Answer each of these questions independently, without referring "
        f"to the draft answer:\n{verification_questions}"
    )
    
    # Step 4: Revise draft based on verification
    final_answer = llm.generate(
        f"Original question: {question}\n"
        f"Draft answer: {draft}\n"
        f"Verification Q&A: {verification_answers}\n\n"
        f"Based on the verification, revise the answer to correct any "
        f"inaccuracies. If the draft was correct, restate it."
    )
    
    return final_answer
```

## Hallucination Detection at Inference Time

For production deployments, a **hallucination detection layer** can evaluate LLM outputs before serving them:

```python
def check_hallucination_risk(question: str, answer: str, sources: list[str], llm) -> dict:
    """
    Evaluate whether an answer is grounded in provided sources.
    Returns a risk score and specific unsupported claims.
    """
    source_text = "\n\n".join(sources)
    
    evaluation = llm.generate(f"""
    Evaluate whether the answer is supported by the provided sources.
    
    Question: {question}
    Answer: {answer}
    Sources: {source_text}
    
    For each factual claim in the answer:
    1. State the claim
    2. State whether it is SUPPORTED, PARTIALLY SUPPORTED, or UNSUPPORTED by the sources
    3. If unsupported, quote what the sources actually say (or note if the topic is absent)
    
    Finally, provide an overall hallucination risk: LOW, MEDIUM, or HIGH.
    """)
    
    return parse_hallucination_evaluation(evaluation)
```

## Practical Mitigation Strategy

A layered approach provides the best practical results:

1. **Use RAG** whenever the domain has a defined knowledge base — this is the single highest-impact intervention.
2. **Lower temperature** (0.0-0.3) for factual tasks — reduces diversity but improves consistency.
3. **Use system prompts** that explicitly instruct the model to express uncertainty and refuse when unsatisfied: "If you are not confident in an answer, say so rather than guessing."
4. **Apply self-consistency** for high-stakes factual questions where compute allows.
5. **Implement output verification** using a separate LLM-as-judge step for critical applications.
6. **Fine-tune on calibrated data** for domain-specific applications where hallucination in specific topics is unacceptable.
7. **Track hallucination metrics** (FactScore, RAGAS faithfulness) in production — hallucination rates drift as user query patterns change.

No combination of techniques eliminates hallucination entirely; the goal is reducing its rate and severity to acceptable levels for the specific application risk tolerance.
