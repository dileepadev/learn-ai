---
title: Recurrent Neural Networks - Processing Sequential Data
description: Understanding RNNs, LSTMs, and GRUs for time series and language tasks.
---

While CNNs excel at spatial data like images, Recurrent Neural Networks (RNNs) are designed for sequential data: time series, text, speech, or any data where order matters. This post explores how they work and their variants.

## The Sequential Data Problem

Standard neural networks treat all inputs as independent. But sequential data has dependencies:

**Examples:**
- Text: "the cat sat on the ___" → next word depends on previous words
- Stock prices: Tomorrow's price depends on past prices
- Weather: Forecasts depend on recent conditions
- Speech: Understanding words depends on context

**Key Insight:** Need memory to use past information.

## RNN Architecture

RNNs maintain hidden state that gets updated as they process sequences.

### The Recurrent Principle

**Key Idea:** Apply same weights recursively across time steps

```
At time t:
h_t = f(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

Where:
- h_t: Hidden state at time t
- x_t: Input at time t
- W_hh: Weights for previous hidden state
- W_xh: Weights for current input
- W_hy: Weights for output

### Unfolded RNN

```
Time:      t-1          t          t+1
          ────────────────────────────────
          │            │            │
Input:  x_{t-1} → x_t → x_{t+1}
          │            │            │
          ↓            ↓            ↓
Hidden: h_{t-1} → h_t → h_{t+1}
          │            │            │
          ↓            ↓            ↓
Output: y_{t-1}    y_t        y_{t+1}
```

**Unfolding:** Shows connections between time steps

### Memory and Context

**Hidden State as Memory:**
- Contains compressed information from all previous inputs
- Passed forward to next time step
- Updated based on current input and previous state

**Example: Sentiment Analysis**

```
Sentence: "The movie was great but too long"

T=0: Input "The", h_0 → some state
T=1: Input "movie", h_0 → h_1 (more context)
T=2: Input "was", h_1 → h_2 (building understanding)
T=3: Input "great", h_2 → h_3 (positive signal)
T=4: Input "but", h_3 → h_4 (but wait...)
T=5: Input "too", h_4 → h_5 (contradiction)
T=6: Input "long", h_5 → h_6 (final state encodes: good movie, too long)
Output: Mixed sentiment based on final h_6
```

## Backpropagation Through Time (BPTT)

Training RNNs involves backpropagating through time steps.

**Process:**
1. Unfold network across time steps
2. Forward pass: Compute all hidden states
3. Backward pass: Propagate gradients backward through time
4. Update weights

**Computational Cost:** Proportional to sequence length

## The Vanishing Gradient Problem

RNNs face a critical challenge: gradients shrink exponentially over many time steps.

### Why It Happens

**Chain Rule Across Many Steps:**
```
dL/dW ∝ ∂y_T/∂y_{T-1} × ∂y_{T-1}/∂y_{T-2} × ... × ∂y_1/∂W

Many multiplications of small numbers → very small gradient
```

**Consequence:**
- Early time steps can't update weights
- Long-term dependencies not learned
- RNNs forget distant past

### Example: Identifying Subject-Verb Agreement

```
"The cats that were sleeping in the room were"
                                              ↑
Subject "cats" far away, but determines "were"
```

Standard RNN might fail to remember subject.

## LSTM: Long Short-Term Memory

LSTMs solve vanishing gradient problem through gating mechanisms.

### Key Innovation: The Cell State

**Instead of just hidden state, maintain:**
- Hidden state (h_t): Short-term memory, output
- Cell state (c_t): Long-term memory, detailed history

### LSTM Gates

**Three gates control information flow:**

#### 1. Forget Gate

**Question:** What to forget from cell state?

**Formula:** f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)

**Output:** Values between 0 (forget) and 1 (keep)

#### 2. Input Gate

**Question:** What new information to add?

**Formula:** 
- i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)
- C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)

**Output:** What to add (i_t) and candidate values (C̃_t)

#### 3. Output Gate

**Question:** What to output?

**Formula:** o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)

**Output:** Controls which parts of cell state become output

### Cell State Update

```
c_t = (f_t ⊙ c_{t-1}) + (i_t ⊙ C̃_t)
      └─────────────┘   └──────────┘
      What to keep      What to add

h_t = o_t ⊙ tanh(c_t)
      └────────────┘
      Controlled output
```

Where ⊙ is element-wise multiplication

### Why LSTMs Work

**Additive Connection:**
```
c_t = f_t × c_{t-1} + i_t × C̃_t
      ↑
Additive not multiplicative!
```

**Benefit:** Gradients can flow unchanged through addition

**Result:** Gradients don't vanish over many steps

### LSTM Analogy

Think of cell state as tape:
- **Forget gate:** Erase parts of tape
- **Input gate:** Write new information
- **Output gate:** What to communicate

The tape carries long-term information; gates control access.

## GRU: Gated Recurrent Unit

GRU is simpler alternative to LSTM with fewer parameters.

### Differences from LSTM

**LSTM:** 3 gates + cell state
**GRU:** 2 gates + hidden state

**Gates:**
- Reset gate: How much past to forget
- Update gate: How much to keep vs update

**Cell State Computation:**
```
r_t = sigmoid(W_r × [h_{t-1}, x_t])
z_t = sigmoid(W_z × [h_{t-1}, x_t])
h̃_t = tanh(W_h × [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
```

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 | 2 |
| **Parameters** | More | Fewer |
| **Computation** | Slower | Faster |
| **Training Data** | More needed | Less needed |
| **Typical Use** | Large data | Limited data |

**Rule of Thumb:**
- GRU: Faster, good for smaller datasets
- LSTM: More expressive, better for large datasets

## Bidirectional RNNs

Use information from both past and future.

**Architecture:**
```
Forward RNN:  → → → (left to right)
Backward RNN: ← ← ← (right to left)

Combined: Concatenate forward and backward outputs
```

**When to Use:**
- When full sequence available (not real-time prediction)
- NLP tasks (translation, tagging)
- Not suitable for live/streaming data

## Sequence-to-Sequence Models

**Encoder-Decoder Architecture:**

1. **Encoder RNN:** Processes input sequence, produces context vector
2. **Context Vector:** Compressed representation of input
3. **Decoder RNN:** Uses context to generate output sequence

**Applications:**
- Machine translation: French → English
- Summarization: Long text → Short summary
- Question answering: Question → Answer
- Speech recognition: Audio → Text

## Attention Mechanism

Context vector bottleneck: Can't compress long sequences into single vector.

**Solution: Attention**

Instead of single context vector, use weighted combination of all encoder states.

**How It Works:**
1. For each decoder step, compute attention weights
2. Weights indicate which encoder outputs are relevant
3. Combine weighted encoder outputs
4. Use combined vector for decoder

**Benefit:**
- Model learns what to focus on
- Better for long sequences
- Interpretable (see which input tokens matter)

## Practical RNN Applications

### Text Generation

- Input: "Once upon a"
- Output: "time there was..."

Training: Learn to predict next character/word

### Machine Translation

- Input: "Bonjour, comment allez-vous?"
- Output: "Hello, how are you?"

Encoder-decoder with attention

### Sentiment Analysis

- Input: Sequence of words
- Output: Sentiment score

Process sequence, final hidden state predicts sentiment

### Time Series Forecasting

- Input: Past stock prices
- Output: Next price prediction

Learn temporal patterns in data

## Training Considerations

### Gradient Clipping

Prevent exploding gradients:
```
if ||gradient|| > threshold:
    gradient = gradient × (threshold / ||gradient||)
```

### Sequence Batching

Pad shorter sequences or batch similar lengths together

### Truncated BPTT

For very long sequences, only backprop a few steps back

## Modern Alternatives

**Transformer Models:**
- Attention-based (no recurrence)
- Faster training (parallelizable)
- Better for long sequences
- Became dominant in NLP

**RNNs Still Used For:**
- Stream/real-time processing
- Time series with variable length
- Simpler tasks with limited data

## Conclusion

RNNs introduced recurrence to handle sequential data. Standard RNNs suffer from vanishing gradients for long sequences. LSTMs solve this through gating mechanisms that control information flow. GRUs offer simpler, more efficient alternatives. Bidirectional RNNs and attention mechanisms further improve capabilities. While Transformers have gained prominence, RNNs remain valuable for many sequential learning tasks, especially those requiring real-time processing or working with variable-length sequences.
