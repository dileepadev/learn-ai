---
title: "Grammar-Constrained Decoding: Beyond Outlines"
description: Explore grammar-constrained decoding for LLMs — how finite-state automata, EBNF grammars, and token masking enforce structured output without the Outlines library, including implementations in llama.cpp, vLLM, and SGLang.
---

Grammar-constrained decoding is a technique that restricts an LLM's token generation to sequences that conform to a formal grammar. Instead of sampling freely from the full vocabulary at each step, the model can only produce tokens that keep the output in a valid partial parse state.

The Outlines library popularized this idea in Python, but grammar-constrained decoding is now natively implemented in **llama.cpp**, **vLLM**, **SGLang**, **Guidance**, **LMQL**, and several other inference engines. Understanding how it works at the level of automata and token masking gives you the ability to use, debug, and extend these systems without relying on any specific library.

## Why Constrain Generation?

LLMs generate syntactically invalid or schema-violating outputs even when explicitly prompted to produce JSON, SQL, or other structured formats. The failure modes include:

- Extra trailing commas in JSON
- Missing closing brackets
- SQL keywords with incorrect casing or order
- Python indentation errors
- Hallucinated field names not present in the requested schema

Grammar-constrained decoding eliminates these failures entirely: invalid tokens are assigned zero probability **before** sampling, making non-compliant output structurally impossible.

## Formal Grammars and Their Representations

A **context-free grammar (CFG)** defines a language via production rules. For structured output, we typically work with:

- **JSON Schema** — defining which keys, types, and structures are valid
- **EBNF (Extended Backus-Naur Form)** — a notation for defining grammars used in llama.cpp's grammar support
- **Regular expressions** — for simpler cases like phone numbers, dates, or enum values
- **Pydantic schemas** — compiled to JSON Schema and then to a grammar

### EBNF Example: Simple JSON Object

```ebnf
root   ::= object
object ::= "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair   ::= string ":" ws value
value  ::= string | number | "true" | "false" | "null" | object | array
array  ::= "[" ws (value ("," ws value)*)? "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n\r]*
```

## From Grammar to Finite-State Automaton

The core algorithm converts a grammar into an automaton that can be advanced token-by-token:

1. **Parse the grammar** into a structured representation (rules, terminals, non-terminals)
2. **Compile to a pushdown automaton (PDA)** or, for regular grammars, a **finite-state automaton (FSA)**
3. **Pre-compute allowed characters** at each FSA state
4. **Map FSA states to token masks** using the model's vocabulary

### Incremental Token Masking

At each decoding step, given the current FSA state $s_t$, compute the set of valid next tokens:

$$\text{valid\_tokens}(s_t) = \{ v \in V : \exists s' \text{ s.t. } (s_t, \text{prefix}(v)) \vdash^* s' \}$$

Where $\text{prefix}(v)$ is the byte string of token $v$ and $\vdash^*$ means "transitions to" in the automaton.

In practice, this is implemented as a **token mask** — a boolean tensor of size $|V|$ — applied to the logits before softmax:

```python
def apply_grammar_mask(logits: torch.Tensor, valid_token_ids: list[int]) -> torch.Tensor:
    """
    Zero out all tokens not permitted by the current grammar state.
    logits: (vocab_size,)
    """
    mask = torch.full_like(logits, float('-inf'))
    mask[valid_token_ids] = 0.0
    return logits + mask
```

### Pre-Computing Vocabulary Masks per State

Naively computing `valid_tokens(s_t)` at every decoding step is expensive. The key optimization is **pre-computing a mapping from FSA states to token masks** before generation begins:

```python
def precompute_state_token_masks(
    fsm: FiniteStateMachine,
    tokenizer_vocab: dict[str, int],
    byte_encoder: dict[int, str],
) -> dict[int, list[int]]:
    """
    For each FSM state, find all token IDs whose byte strings are
    valid continuations from that state.
    Returns: {state_id: [valid_token_id, ...]}
    """
    state_masks = {}
    for state in fsm.states:
        valid = []
        for token_str, token_id in tokenizer_vocab.items():
            byte_str = decode_token_bytes(token_str, byte_encoder)
            if fsm.can_advance(state, byte_str):
                valid.append(token_id)
        state_masks[state] = valid
    return state_masks
```

This precomputation happens once and is cached. At inference time, looking up `state_masks[current_state]` is O(1).

## Handling Byte-Level Tokenizers

Modern LLMs like LLaMA, Mistral, and GPT-4 use byte-pair encoding (BPE) tokenizers that operate on UTF-8 bytes, not characters. This complicates grammar constraints because:

1. A single grammar terminal (e.g., the `{` character) may correspond to many different tokens depending on context
2. Token boundaries may fall in the middle of a multi-byte character
3. Whitespace handling in grammars (e.g., optional spaces) creates combinatorial token paths

The solution is to work at the **byte level** — convert the grammar to a byte-level DFA and check which token byte strings are valid transitions from the current state. The `interegular` and `lark` libraries provide building blocks for this in Python.

## Implementation in llama.cpp

llama.cpp has native EBNF grammar support via the `llama_grammar` API:

```bash
# Run with a grammar file
./llama-cli \
  -m model.gguf \
  --grammar-file json.gbnf \
  -p "Extract the person's name and age as JSON:"
```

The `.gbnf` (GGML BNF) format is a subset of EBNF. llama.cpp compiles it to a stack-based parser and applies token masking during sampling:

```c
// llama.cpp internal — simplified
struct llama_grammar_candidate {
    size_t index;
    const uint32_t * code_points;
    llama_partial_utf8 partial_utf8;
};

// At each token position, filter candidates by grammar state
static void llama_grammar_advance_stack(
    const llama_grammar_rules & rules,
    const llama_grammar_stacks & stacks,
    llama_grammar_stacks & new_stacks
);
```

## Implementation in vLLM

vLLM 0.4+ supports structured output via the `guided_decoding` parameter:

```python
from vllm import LLM, SamplingParams
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
params = SamplingParams(
    temperature=0.0,
    guided_decoding={"type": "json_schema", "json_schema": Person.model_json_schema()}
)

outputs = llm.generate(["Extract person info: John Smith, 34, software engineer"], params)
print(outputs[0].outputs[0].text)
# {"name": "John Smith", "age": 34, "occupation": "software engineer"}
```

Internally, vLLM uses the `xgrammar` library or the `outlines` backend to compile schemas to FSAs and compute token masks, applying them during the sampling step.

## Limitations and Tradeoffs

Grammar-constrained decoding is not free:

**Compute overhead:** Pre-computing token masks for large grammars with many states can take seconds to minutes, though this cost is amortized over the generation. At inference time, the overhead per token is typically under 1ms.

**Grammar coverage vs. model capability:** Constraining generation to a grammar cannot fix a model that does not understand the requested content — it only ensures structural validity. A grammar-constrained model can still produce semantically incorrect outputs (wrong values, wrong field semantics).

**Recursive grammars:** Full context-free grammars (which allow unbounded nesting) require pushdown automata, not just FSAs. This makes the token mask computation more complex and less efficient.

**Ambiguous tokenization:** Some grammar terminals can be tokenized multiple ways (e.g., `true` vs. `t` + `rue`). The mask must account for all valid tokenizations of each terminal, not just the canonical one.

**Very large vocabularies:** Vocabularies of 128K+ tokens (common in recent models) make the precomputation step heavier. Sparse mask representations and batched FSA transitions are needed for efficiency.

## When to Use Grammar Constraints

Grammar-constrained decoding is most valuable when:

- Output must be machine-parseable (API responses, database records, code)
- The grammar is fixed or changes infrequently (amortizing precomputation)
- The model would otherwise produce valid-looking but syntactically invalid output in 5–20% of cases
- You need deterministic validation without a retry loop

It is less necessary when:
- The model is large and well-instruction-tuned (e.g., GPT-4 producing JSON is usually valid)
- The schema changes per request (high precomputation cost)
- You are generating prose or semi-structured output where strict grammar is inappropriate

## The Bigger Picture: Structured Generation as a First-Class LLM Feature

Grammar-constrained decoding represents a broader shift: treating LLMs not as black-box text generators but as components in typed, structured data pipelines. Libraries like Instructor, Marvin, and LangChain's structured output parsers all use variants of this idea — either via grammar constraints or post-hoc validation with retries.

The ongoing convergence of inference engine support (llama.cpp, vLLM, SGLang, TGI) means grammar-constrained generation is becoming a standard feature rather than a specialized technique. Understanding the underlying automata theory positions you to use these tools effectively and extend them when your use case requires it.
