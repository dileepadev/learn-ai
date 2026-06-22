---
title: Prompt-Lookup Decoding & Draft-Free Speculative Decoding
description: Learn how Prompt-Lookup Decoding accelerates LLM generation by reusing repeating n-grams from the prompt context, skipping draft models entirely for copy-paste-heavy tasks.
---

Speculative decoding is a popular technique to accelerate Large Language Model (LLM) inference by using a small "draft model" to generate candidate tokens that a larger "target model" validates in a single forward pass. However, maintaining and running a draft model introduces engineering complexity and extra memory consumption.

**Prompt-Lookup Decoding** is a draft-free speculative decoding technique. Instead of using a neural network to generate candidate tokens, it uses a simple string-matching heuristic to find repeating patterns (*n-grams*) directly in the prompt or context window, achieving up to 2x speedups on tasks like editing, summarization, and retrieval.

---

## The Heuristic: Repetitive Language Patterns

Many LLM workloads are copy-paste-heavy. For example:
- **Summarization:** The output heavily copies names, key phrases, and facts from the source text.
- **RAG / Q&A:** Answers repeat passages from the retrieved documents.
- **Code Editing:** Modified code blocks share 90% of their structure with the original code.

Prompt-Lookup Decoding exploits this repetition. If the target model generates the token sequence `"the Capital of France is"`, the system searches the prompt for this sequence. If it finds `"the Capital of France is Paris"` in the prompt, it guesses that the next token will be `"Paris"`.

---

## How Prompt-Lookup Decoding Works

The system maintains a history of generated tokens and the original prompt context.

```
Prompt: "The capital of France is Paris. The capital of Germany is Berlin."
Model generated so far: "I went to France. The capital of France is..."
```

1. **N-gram Match Search:** The decoding engine looks at the last $K$ tokens generated (e.g., $K=3$, representing `"capital of France"`). It searches the prompt for occurrences of this 3-gram.
2. **Draft Projection:** It finds a match in the prompt: `"...[capital of France] is Paris..."`.
3. **Speculative Candidate Generation:** It copies the next $N$ tokens that followed that match in the prompt (e.g., `"is"`, `"Paris"`) and proposes them as candidates.
4. **Target Model Validation:** The target model processes the candidates in a single parallelized forward pass:
   
   $$\text{Verify}([\text{"is"}, \text{"Paris"}], \text{given context})$$

5. **Acceptance:** If the target model agrees with all or some of the candidates, those tokens are accepted instantly, skipping multiple sequential generation steps. If a candidate is rejected, the system falls back to the target model's actual prediction and adjusts the search index.

---

## Comparison: Standard Speculative Decoding vs. Prompt-Lookup Decoding

| Aspect | Standard Speculative Decoding | Prompt-Lookup Decoding |
|---|---|---|
| **Draft Generator** | Smaller Neural Network (e.g., Llama-68M) | Exact Substring Matcher (CPU-bound) |
| **Memory Overhead** | High (must fit two models in VRAM) | Zero (uses existing KV cache) |
| **Applicability** | General-purpose generation | Tasks with high context overlap |
| **Speedup Source** | Neural model approximation | Context-based duplication |

---

## Code Concept: Simulating Prompt-Lookup

Here is a simplified Python representation of the lookup algorithm.

```python
def find_speculative_candidates(prompt_tokens, generated_tokens, lookback=3, draft_len=4):
    """
    Finds candidates from prompt_tokens based on the end of generated_tokens.
    """
    if len(generated_tokens) < lookback:
        return []
        
    # Get the last K tokens generated
    query = generated_tokens[-lookback:]
    
    # Simple search in prompt_tokens for the query pattern
    for i in range(len(prompt_tokens) - lookback - draft_len):
        if prompt_tokens[i:i+lookback] == query:
            # Found a match! Return the subsequent tokens as candidates
            candidates = prompt_tokens[i+lookback : i+lookback+draft_len]
            return candidates
            
    return []

# Example:
prompt = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
gen    = ["I", "saw", "the", "quick", "brown"]

candidates = find_speculative_candidates(prompt, gen, lookback=2, draft_len=3)
print(f"Proposed Candidates: {candidates}")
# Output: Proposed Candidates: ['fox', 'jumps', 'over']
```

---

## Integration in Production

Prompt-Lookup Decoding is supported natively in popular inference engines:
- **vLLM:** Configurable via speculative decoding arguments (`--speculative-model=draft`).
- **Hugging Face Transformers:** Activated by passing a custom candidate generator to the `.generate()` method:
  
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from transformers.generation import PromptLookupCandidateGenerator
  
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
  
  # Initialize candidate generator
  candidate_generator = PromptLookupCandidateGenerator(max_matching_ngram_size=3)
  
  # Generate with speculative lookup
  outputs = model.generate(
      **inputs, 
      candidate_generator=candidate_generator,
      max_new_tokens=100
  )
  ```
