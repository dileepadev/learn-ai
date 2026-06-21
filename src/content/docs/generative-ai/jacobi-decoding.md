---
title: Jacobi Decoding and Parallel Speculative Decoding
description: Learn how Jacobi Decoding reformulates autoregressive generation into a system of non-linear equations, enabling parallel token prediction.
---

Autoregressive language models generate tokens sequentially: predicting token $x_t$ requires waiting for tokens $x_{<t}$ to be computed. This serial dependency limits GPU utilization, as modern hardware is optimized for parallel workloads.

**Jacobi Decoding** (sometimes called **Parallel Decoding** or **Speculative Jacobi Decoding**) reformulates autoregressive generation as solving a system of non-linear equations. By predicting and updating all tokens in a sequence simultaneously using a fixed-point iteration algorithm, Jacobi Decoding enables parallel token generation, accelerating inference on multi-token outputs.

---

## The Mathematical Reformulation

Standard decoding evaluates tokens one-by-one:

$$x_{t} = f(x_{<t})$$

In contrast, Jacobi Decoding views a sequence of length $L$ as a fixed-point problem. Let $X = (x_1, x_2, \dots, x_L)$ be a sequence. We define the system of equations:

$$x_i = g_i(x_1, \dots, x_{i-1}) \quad \text{for } i = 1, \dots, L$$

Where $g_i$ is the model's greedy prediction at position $i$ given the preceding tokens. 

We can solve this system using **Jacobi Fixed-Point Iteration**:
1. **Initialize:** Start with a random or heuristic sequence estimate $X^{(0)} = (x_1^{(0)}, \dots, x_L^{(0)})$.
2. **Iterate:** In parallel, update all tokens in the sequence using the values from the previous iteration step:
   
   $$x_i^{(k+1)} = g_i\left(x_1^{(k)}, \dots, x_{i-1}^{(k)}\right) \quad \text{for all } i \text{ simultaneously}$$

3. **Convergence:** Repeat step 2 until the sequence stabilizes ($X^{(k+1)} = X^{(k)}$). 

In practice, the sequence often converges in significantly fewer than $L$ steps, allowing the model to generate $L$ tokens using only a fraction of $L$ forward passes.

---

## The Jacobi Iteration Step Visualized

Consider generating the sentence `"The cat sat on the mat"` (length $L=6$).

```
Iteration 0 (Initialization):
[ "The", "dog", "slept", "under", "a", "tree" ]

Iteration 1 (Parallel Forward Pass):
- Feed [ "The", "dog", "slept", "under", "a" ] to the model.
- The model predicts next-tokens for every prefix position in parallel:
  Position 1: "The" -> predicts "cat"
  Position 2: "The dog" -> predicts "sat"
  Position 3: "The dog slept" -> predicts "on"
  ...
Update sequence to:
[ "The", "cat", "sat", "on", "the", "mat" ]

Iteration 2:
- Repeat parallel pass. If the predictions match the current tokens, the system has converged.
```

If the sequence converges in 2 iterations, the model has generated 6 tokens using only 2 forward passes.

---

## Speculative Jacobi Decoding

Standard Jacobi decoding struggles with long sequences because the probability of converging across a long window drops exponentially.

To address this, **Speculative Jacobi Decoding** combines Jacobi decoding with a draft model:
- A smaller, fast draft model generates an initial estimate sequence $X^{(0)}$ of length $K$.
- The large target model runs a single parallel Jacobi iteration step over $X^{(0)}$ to verify and update all $K$ tokens.
- This hybrid approach achieves higher convergence rates while maintaining the speed of draft-free Jacobi iterations.

---

## Comparison: Autoregressive vs. Jacobi Decoding

| Feature | Autoregressive Decoding | Jacobi Decoding (Parallel) |
|---|---|---|
| **Forward Passes** | Exactly $L$ passes | $K$ passes (where $K \ll L$ upon convergence) |
| **GPU Utilization** | Low (computes one token at a time) | High (processes entire sequence in parallel) |
| **Computations per Step** | Small (processes single-token query) | Large (processes entire sequence batch) |
| **Mathematical Approach** | Greedy progression | Fixed-point system solution |
| **Output Type** | Exact greedy match | Exact greedy match (guaranteed mathematically) |

---

## Python Concept: Jacobi Iteration Loop

Below is a conceptual PyTorch snippet demonstrating how Jacobi Decoding processes sequences in parallel.

```python
import torch

def jacobi_decode(model, prompt_tokens, target_length=8, max_iters=15):
    # Initialize sequence with prompt + padding/random tokens
    # seq shape: [1, prompt_len + target_length]
    prompt_len = len(prompt_tokens)
    seq = torch.tensor(prompt_tokens + [0] * target_length).unsqueeze(0)
    
    for iteration in range(max_iters):
        # 1. Parallel forward pass over the entire sequence
        # We compute logits for all positions [0, N-1] at once
        logits = model(seq) # [1, seq_len, vocab_size]
        
        # 2. Extract greedy predictions for every position
        predictions = torch.argmax(logits, dim=-1) # [1, seq_len]
        
        # 3. Construct the next iteration state
        # For position i, the new token is the prediction made at position i-1
        new_seq = seq.clone()
        new_seq[0, prompt_len:] = predictions[0, prompt_len - 1 : -1]
        
        # 4. Check for convergence
        if torch.equal(new_seq, seq):
            print(f"Converged at iteration {iteration}")
            return new_seq[0].tolist()
            
        seq = new_seq
        
    print("Reached max iterations without full convergence")
    return seq[0].tolist()
```
