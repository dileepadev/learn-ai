---
title: "Private Set Intersection for Machine Learning"
description: Understand Private Set Intersection (PSI) and how this cryptographic primitive enables privacy-preserving machine learning — allowing parties to compute on shared data without revealing what they each hold, with applications in healthcare, finance, and federated training.
---

Two hospitals each have patient records. Hospital A wants to know which of its patients is also a patient at Hospital B, so they can study outcomes for patients with overlapping care. But neither hospital can share its patient list with the other — patient privacy laws, competitive concerns, and data governance policies forbid it.

This is the canonical setup for **Private Set Intersection (PSI)**: a cryptographic protocol that allows two parties to compute the intersection of their datasets without revealing anything beyond the intersection itself. PSI and related multi-party computation (MPC) primitives are increasingly important in machine learning, where training on richer data requires combining datasets that cannot be shared openly.

## The Private Set Intersection Problem

Formally: Alice holds a set $A = \{a_1, \ldots, a_m\}$ and Bob holds a set $B = \{b_1, \ldots, b_n\}$. After running a PSI protocol:

- Alice (and/or Bob, depending on the variant) learns $A \cap B$
- Neither party learns anything about elements not in the intersection — Alice learns nothing about Bob's elements outside $A \cap B$, and vice versa

The security guarantee is cryptographic, not heuristic. PSI protocols rely on number-theoretic hardness assumptions (discrete logarithm, RSA, etc.) or symmetric crypto primitives, and their security is formally proven.

### Variants

The basic PSI setting has several variants that matter for ML applications:

**Standard PSI:** Both parties learn the intersection.

**One-sided PSI (receiver-only):** Only Alice learns the intersection. Bob learns nothing — not even the size of the intersection.

**PSI-Cardinality:** Both parties learn only the *size* of the intersection, not the elements themselves.

**PSI-Sum / PSI-Statistics:** Parties jointly compute a statistic (sum, average, count) over values associated with intersecting elements, without revealing the individual values.

**Labeled PSI:** Elements in $B$ have associated labels or features. Alice learns the features corresponding to intersecting elements, without Bob revealing the features of non-intersecting elements.

## Core PSI Protocols

### Diffie-Hellman Based PSI

The simplest PSI protocol uses the Diffie-Hellman assumption. Let $H: \{0,1\}^* \to \mathbb{G}$ be a hash-to-group function mapping elements to a cyclic group $\mathbb{G}$ of order $p$ where the DH problem is hard.

**Protocol:**

1. Alice samples random $a \in \mathbb{Z}_p^*$, computes $\{H(x)^a : x \in A\}$, sends to Bob.
2. Bob samples random $b \in \mathbb{Z}_p^*$, computes $\{H(x)^b : x \in B\}$ and $\{H(x)^{ab} : x \in \text{received from Alice}\}$, sends both sets to Alice.
3. Alice computes $\{H(x)^{ab} : x \in A\}$ by exponentiating each received-from-step-1 element.
4. Alice computes the intersection by comparing her set from step 3 with Bob's set from step 2.

Bob learns nothing because he only sees $\{H(x)^a\}$ — without knowing $a$, he cannot invert the exponentiation. Alice only learns intersection elements because she can only evaluate the matching function on elements she owns.

**Complexity:** $O(m + n)$ exponentiations and $O(m + n)$ communication. This is relatively efficient but not optimal.

### Oblivious PRF (OPRF) Based PSI

More efficient modern PSI protocols are built on **Oblivious Pseudorandom Functions (OPRFs)**. An OPRF is a two-party protocol where:
- Alice holds input $x$ and learns $F_k(x)$ (the PRF output)
- Bob holds key $k$ and learns nothing about $x$
- Alice learns nothing about $k$

PSI using OPRF:
1. For each $b_j \in B$, Bob evaluates the OPRF to get $F_k(b_j)$, then hashes them to build a lookup table.
2. For each $a_i \in A$, Alice uses the OPRF to learn $F_k(a_i)$ without revealing $a_i$ to Bob.
3. Alice checks which $F_k(a_i)$ values appear in Bob's table — these correspond to intersection elements.

State-of-the-art OPRF-based PSI (KKRT, 2016; PRTY, 2019) achieves millions of elements intersected per second in practice.

### Circuit-Based PSI

For PSI-Sum and more complex statistics, protocols based on garbled circuits or secret sharing compute arbitrary functions over the intersection:

```
PSI-Sum Example:
  Alice: patient_ids = {p1, p2, p3, ...}, no labels
  Bob:   patient_ids = {p2, p4, p5, ...}, labels = {revenue, cost, ...}
  
  Goal: compute average revenue for patients in intersection
  
  Without PSI: requires sharing all data
  With PSI-Sum: parties jointly compute the average via MPC,
                neither learns which specific patients intersected
```

These protocols are more expensive — circuit complexity grows with the complexity of the function being computed — but enable richer computations.

## PSI in Machine Learning Pipelines

### Record Linkage Across Organizations

The most direct application: discovering which records in dataset A correspond to records in dataset B, so features from both datasets can be combined for training.

**Example:** A bank (features: transaction history, credit score) and a healthcare provider (features: diagnoses, medications) want to train a model predicting disease risk in patients who also have financial stress. Neither can share its raw data. PSI identifies the overlapping individuals; then techniques like Federated Learning or Secure Multi-Party Computation compute gradients over the linked records without centralization.

### Federated Learning with PSI

In standard federated learning, all parties know which samples they hold but train local models and share gradients. Adding PSI enables:

1. **Vertical federated learning alignment:** Two parties hold different features for the same individuals. PSI first aligns records (finds who appears in both datasets), then vertical FL trains on aligned samples without either party learning the other's raw features.

2. **Training set filtering:** A model owner wants to ensure it doesn't train on data for which it has no license. PSI against a licensed data registry identifies which training samples are permissible.

### Private Inference and Feature Lookup

After training, production ML inference sometimes needs to check whether a query item appears in a sensitive reference set. For example:

- **Fraud detection:** Does this transaction pattern appear in a known-fraudulent set maintained by another institution?
- **Content moderation:** Does this image hash appear in a hash database of known harmful content (PhotoDNA-style)?

PSI protocols allow this lookup without exposing the reference set or the query. One-sided PSI is particularly useful here — the database owner learns only whether there was a match (and optionally the matched item), not the full query context.

### Differential Privacy + PSI

PSI tells you *who* is in the intersection but not associated sensitive values. When the goal is statistical: "what is the average income of people who appear in both datasets?", adding differential privacy noise to the computed statistic protects individuals within the intersection:

```
Protocol:
1. PSI: identify intersection
2. MPC: jointly compute statistic over intersection values
3. DP: add calibrated noise before releasing the result
4. Release: sanitized aggregate statistic
```

This combination — PSI + MPC + DP — is the current gold standard for privacy-preserving analytics across institutional boundaries.

## Computational Costs

PSI is no longer prohibitively expensive for practical ML use cases:

| Protocol | Set Size | Runtime | Communication |
|---------|----------|---------|---------------|
| DH-PSI | 1M × 1M | ~30s | ~800 MB |
| KKRT-OPRF | 1M × 1M | ~2s | ~80 MB |
| Labeled-PSI (bitset) | 1M × 1B | ~10s | ~100 MB |

The Labeled PSI variant (where Bob has a large database and Alice queries individual items) can handle billion-item server-side sets efficiently using batched computations and Bloom filter representations.

For ML-scale datasets (millions to tens of millions of records), modern PSI protocols run in seconds to minutes on commodity hardware, making them feasible for use in data preparation pipelines.

## Real-World Deployments

**Private Join and Compute (Google):** An open-source implementation for private data joining at scale, used internally to compute aggregate statistics across datasets held by different Google teams without centralizing sensitive data.

**Meta's FBPCS (Private Computation Service):** Used for privacy-preserving ad measurement — computing ad conversion statistics across Meta's ad server data and advertisers' purchase data using PSI and MPC, without Meta seeing purchase data or advertisers seeing click data.

**Privacy-Preserving Record Linkage (PPRL) in Healthcare:** Multiple health systems have piloted PSI-based patient matching to enable research on multi-institutional cohorts without violating HIPAA. The All of Us Research Program and PCORnet have both explored PSI-based cohort building.

**Regulatory Compliance:** GDPR's restrictions on data sharing have accelerated PSI adoption in Europe. PSI allows joint analysis while staying compliant — the data never leaves the originating party's infrastructure.

## Implementation Libraries

Several production-quality libraries implement PSI protocols:

**OpenMined PSI:** Python library wrapping highly optimized C++ PSI implementations. Supports ECDH-PSI and KKRT-based protocols. Easy integration with PySyft for federated learning.

**Private-ID (Meta):** Implements multiple PSI protocols with an emphasis on production-scale deployment. Available on GitHub.

**APSI (Asymmetric PSI):** Microsoft's implementation for the labeled/asymmetric PSI case, handling very large server-side sets efficiently using lattice-based encryption.

**TF-Encrypted:** TensorFlow integration for secure multi-party computation including PSI, designed for federated learning pipelines.

```python
# Example using OpenMined PSI
import openmined_psi as psi

# Server (Bob) setup
server = psi.Server.CreateWithNewKey(reveal_intersection=True)
setup = server.CreateSetupMessage(0.001, len(server_data), server_data)

# Client (Alice) query
client = psi.Client.CreateWithNewKey(reveal_intersection=True)
request = client.CreateRequest(client_data)

# Server response
response = server.ProcessRequest(request)

# Client learns intersection
intersection = client.GetIntersection(setup, response)
```

## Limitations and Challenges

**Scalability to very large sets.** PSI with sets of billions of elements requires specialized protocols and significant infrastructure. Naive OPRF-based PSI doesn't scale directly.

**Multi-party PSI.** Standard PSI is two-party. Extending to three or more parties requires more complex MPC protocols and is significantly more expensive.

**Approximate matching.** PSI computes exact set intersection — elements must match exactly. Record linkage with typos, abbreviations, or format differences requires fuzzy matching, which is much harder to do privately (private fuzzy PSI is an active research area).

**Intersection size leakage.** Most PSI protocols reveal the size of the intersection to both parties, which can itself be sensitive information. PSI-cardinality-hiding protocols exist but are more expensive.

**Key management.** PSI protocols require parties to manage cryptographic keys carefully. Key compromise or improper rotation can undermine security guarantees.

## The Future of PSI in AI

As AI systems increasingly draw on data from multiple institutions — healthcare networks, financial consortia, research collaborations — PSI and its variants will become a standard component of the data pipeline. The combination of PSI for record linkage, differential privacy for output sanitization, and federated learning for model training defines a privacy-preserving ML stack that enables collaboration that would otherwise be legally and ethically impossible.

The technical progress in PSI over the past decade has been remarkable: what once required hours of computation now runs in seconds. The remaining challenges are as much about integration, tooling, legal clarity, and organizational adoption as they are about cryptography.
