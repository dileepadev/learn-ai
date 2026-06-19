---
title: Pre-training Data Deduplication (MinHash & SemDeDup)
description: Explore pre-training data deduplication techniques like MinHash LSH and SemDeDup that clean web-scale datasets to reduce training times and prevent memorization.
---

Large Language Models are trained on massive datasets containing trillions of tokens extracted from the web (e.g., Common Crawl). However, raw web data is highly repetitive: news articles are syndicated across websites, blogs republish the same documentation, and boilerplate code is copied across repositories. 

Training on duplicate data causes models to memorize text verbatim, increases training costs, and can degrade performance. **Pre-training Data Deduplication** addresses these issues. By leveraging algorithms like **MinHash LSH** for syntactic matches and **SemDeDup** for semantic duplicates, researchers can remove up to 50% of web tokens without affecting model accuracy.

---

## The Impact of Duplicate Data

1. **Over-training and Memorization:** Models are highly prone to memorizing duplicate sequences, leading to privacy issues (e.g., leaking PII) and plagiarism.
2. **Degraded Generalization:** Repetitive training data skews the token distribution, causing the model to over-index on common templates at the expense of rare, high-quality information.
3. **Compute Waste:** Processing identical text blocks consumes millions of FLOPS that could otherwise be spent learning new domains.

---

## Syntactic Deduplication: MinHash LSH

Syntactic deduplication identifies documents that share identical or highly similar sequences of words. Standard pairwise string matching is computationally unfeasible at web scale ($\mathcal{O}(N^2)$ for billions of documents). Instead, researchers use **Locality-Sensitive Hashing (LSH)** with **MinHash**.

```
Document A ---> Shingling (n-grams) ---> MinHashing (Signature Vector) ---\
                                                                           +---> LSH Bucketing ---> Candidate Matches
Document B ---> Shingling (n-grams) ---> MinHashing (Signature Vector) ---/
```

### 1. Shingling
Documents are split into overlapping character-level or token-level $n$-grams (called shingles). For example, 3-shingling `"the quick brown"` yields `{"the quick", "quick brown"}`.

### 2. MinHashing
To estimate the Jaccard similarity between shingle sets without storing them, we apply a set of $H$ hash functions. The MinHash value for a document under hash function $h_i$ is the minimum hash value generated over all shingles in the document:

$$\text{MinHash}_{h_i}(D) = \min_{s \in D} h_i(s)$$

A signature vector of length $H$ (typically $H=100$ or $200$) is constructed. The probability that two documents share the same MinHash value equals their Jaccard similarity.

### 3. Locality-Sensitive Hashing (LSH)
Signature vectors are divided into $b$ bands of $r$ rows ($b \times r = H$). If two documents share an identical signature within any single band, they are hashed to the same bucket and flagged as duplicate candidates for verification.

---

## Semantic Deduplication: SemDeDup

Syntactic deduplication misses documents that are semantically identical but phrased differently (e.g., two news articles summarizing the same press release using different vocabulary).

**SemDeDup (Semantic Deduplication)** targets these instances by operating in embedding space:
1. **Embedding Generation:** Source documents are passed through a lightweight encoder model (like a small BERT or sentence-transformer) to generate semantic embeddings.
2. **K-Means Clustering:** The embedding vectors are partitioned into $C$ clusters.
3. **Intra-cluster Similarity:** Within each cluster, the cosine similarity between all pairs is computed. If the cosine similarity exceeds a threshold (e.g., $\text{sim}(u, v) \ge 0.93$), the document with the shorter sequence length is evicted.

By restricting pairwise comparisons to individual cluster subsets, SemDeDup scales linearly with dataset size.

---

## Comparison: MinHash vs. SemDeDup

| Metric | MinHash LSH | SemDeDup |
|---|---|---|
| **Similarity Type** | Syntactic (word-for-word overlap) | Semantic (meaning and content overlap) |
| **Representational Space** | Text character shingle hashes | Dense embedding vectors |
| **Computational Footprint** | Low (integer hash operations) | High (requires model inference + clustering) |
| **Scope of Removal** | Near-duplicate web pages, boilerplate | Paraphrases, redundant articles, translations |
| **Typical Target** | Initial pipeline filtering stage | Fine-tuning/curation stage |

---

## Python Concept: MinHash Signature Computation

Below is a Python demonstration of how to calculate MinHash signatures for a document to estimate similarity.

```python
import hashlib

class MinHash:
    def __init__(self, num_hashes=128):
        self.num_hashes = num_hashes
        # Generate random hash coefficients (a, b) for: (a * x + b) % prime
        # Using simple hashlib permutations for demonstration
        self.salts = [f"salt_{i}".encode('utf-8') for i in range(num_hashes)]

    def compute_signature(self, text, k=5):
        # 1. Generate k-shingles (word level)
        words = text.lower().split()
        shingles = set(" ".join(words[i:i+k]) for i in range(len(words) - k + 1))
        
        if not shingles:
            return [float('inf')] * self.num_hashes
            
        # 2. Compute signature
        signature = []
        for salt in self.salts:
            min_val = float('inf')
            for shingle in shingles:
                # Hash shingle with salt
                h = hashlib.sha256(salt + shingle.encode('utf-8')).hexdigest()
                int_val = int(h, 16)
                if int_val < min_val:
                    min_val = int_val
            signature.append(min_val)
            
        return signature

# Example Usage:
doc1 = "The quick brown fox jumps over the lazy dog."
doc2 = "A quick brown fox jumped over that lazy dog."

m = MinHash(num_hashes=64)
sig1 = m.compute_signature(doc1)
sig2 = m.compute_signature(doc2)

# Estimate Jaccard similarity by counting matching signature elements
matches = sum(1 for s1, s2 in zip(sig1, sig2) if s1 == s2)
similarity_estimate = matches / len(sig1)
print(f"Estimated Syntactic Similarity: {similarity_estimate:.2f}")
```
