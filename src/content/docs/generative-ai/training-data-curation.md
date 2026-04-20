---
title: Training Data Curation for LLMs
description: How frontier AI labs build, filter, and manage the massive text datasets that power large language models — covering data sourcing, quality filtering, deduplication, domain mixing, and landmark datasets like The Pile, FineWeb, and Dolma.
---

**Training data curation** is one of the most consequential — and least discussed — aspects of building large language models. While model architecture and training algorithms receive significant research attention, the quality and composition of training data often determines model capability more than any other single factor. "Garbage in, garbage out" applies at a trillion-token scale.

## The Scale of LLM Training Data

Modern frontier LLMs are trained on datasets ranging from hundreds of billions to tens of trillions of tokens:

- **GPT-3** (2020): ~300 billion tokens.
- **LLaMA 2** (2023): 2 trillion tokens.
- **LLaMA 3** (2024): 15 trillion tokens.
- **Estimated frontier models** (2025+): 10–100 trillion tokens.

At these scales, the entire publicly indexed web — Common Crawl contains roughly 250 trillion raw tokens — is the primary data source. But raw web data is noisy, repetitive, and often low quality.

## Primary Data Sources

### Common Crawl

**Common Crawl** is a non-profit that has continuously crawled and archived the public web since 2008. It releases monthly snapshots containing petabytes of HTML and extracted text. It is the foundational data source for virtually every open LLM.

**Raw quality**: Very low. Common Crawl includes spam, boilerplate, SEO-stuffed content, adult material, and machine-generated text.

### Curated Web Datasets

Several research groups have published cleaned subsets of Common Crawl:

- **C4** (Google, 2020): Used to train T5. Applied heuristic filters to Common Crawl: removed pages with fewer than 3 sentences, filtered pages containing profanity, deduplicated aggressively.
- **The Pile** (EleutherAI, 2021): A 825 GB curated dataset combining Common Crawl with 22 high-quality sources: academic papers (arXiv, PubMed), code (GitHub), books (Books3), legal documents (FreeLaw), Wikipedia, and more.
- **RedPajama** (Together AI, 2023): An open reproduction of LLaMA's training data recipe at full scale.
- **Dolma** (AI2, 2024): A fully open, documented 3T-token dataset with careful provenance tracking and data cards.
- **FineWeb** (HuggingFace, 2024): A 15T-token filtered Common Crawl dataset that significantly outperforms C4 and prior web datasets on benchmarks.

### High-Quality Sources

Certain data sources punch well above their size in training impact:

- **Books**: Long-form coherent text that teaches structure, narrative, and multi-paragraph reasoning. OpenAI's Books1/Books3, Project Gutenberg.
- **Code**: GitHub, The Stack. Code trains logical reasoning, structured output, and step-by-step thinking.
- **Academic papers**: arXiv, S2ORC, PubMed Central. Science and math text.
- **Wikipedia**: Factual encyclopedic text with consistent style and reliable citations.
- **Stack Exchange / Reddit**: Q&A format, diverse topics, informal register.

### Data Mixture

The proportion of each source in the training mix matters enormously. LLaMA's data recipe (Touvron et al., 2023) revealed that heavily weighting code and academic text — even though they represent a small fraction of the raw web — dramatically improved reasoning capabilities.

## The Data Curation Pipeline

A typical LLM data curation pipeline involves several stages:

### 1. URL and Domain Filtering

Before text extraction, URL-level and domain-level filters block known low-quality domains:

- Adult content domains.
- Known spam and SEO farms.
- Blacklisted domains maintained by organizations like the Colossal Clean Crawled Corpus (C4) team.

### 2. Language Identification

Most LLMs target primarily English (or a specific language mix). Language classifiers (fastText-langdetect, CLD3) identify and filter pages by language. Precision matters: mislabeled non-English text degrades model quality.

### 3. Text Extraction and Normalization

Converting raw HTML to clean text involves:

- Removing HTML tags, JavaScript, CSS.
- Extracting main content vs. navigation, ads, and sidebars (trafilatura, jusText, Boilerplate detection).
- Unicode normalization.
- Removing non-printable characters and encoding artifacts.

### 4. Quality Filtering

Quality filtering is where curation most diverges from raw data processing. Key heuristics:

**Perplexity filtering**: Train a small reference language model (e.g., KenLM trigram model on Wikipedia). Filter out documents with perplexity too high (incoherent/random text) or too low (templated/repetitive content).

**Length filters**: Minimum word count per document, minimum line count.

**Repetition filters**: Documents where many lines or paragraphs are repeated (common in SEO spam and aggregator sites).

**Word list filters**: Block documents containing too many profanity, adult-content, or hate-speech terms.

**Classifier-based filtering**: Train a classifier to distinguish high-quality text (Wikipedia, books) from low-quality text (spam). Use the classifier score as a filter. **FineWeb** uses this approach, training a quality classifier on samples rated by GPT-4.

### 5. Deduplication

Deduplication is essential — web crawls contain massive redundancy from scraped content, SEO duplicates, and mirror sites. Training on duplicates causes the model to memorize specific content and reduces diversity.

**Exact deduplication**: Hash-based matching of identical documents or paragraphs (MD5/SHA256 hashes).

**Fuzzy/near-deduplication**: **MinHash LSH** (Locality-Sensitive Hashing) approximates Jaccard similarity to find near-duplicate documents — catching reworded or partially duplicated content. Used by LLaMA, The Pile, FineWeb.

**Substring deduplication**: Suffix array-based methods that find long repeated substrings across the entire corpus, catching memorized boilerplate even when documents differ otherwise.

**Impact**: Aggressive deduplication can remove 20–50% of web data while improving downstream benchmark scores — indicating that quality beats quantity at training data scale.

### 6. Personal Information Removal

Best practices and some regulations require removing personally identifiable information (PII) from training data:

- Email addresses, phone numbers, SSNs — identified via regex.
- Names associated with sensitive content — harder to detect automatically.
- IP addresses, URLs containing usernames.

### 7. Domain and Format Mixing

The final training corpus is assembled by mixing sources according to a defined ratio. Key design decisions:

- **Upsampling** high-quality sources: Books and code may be upsampled 2–10× relative to their natural web frequency.
- **Multi-epoch considerations**: Some high-quality sources are repeated multiple times (4–10 epochs) while lower-quality web data is used once.

## Evaluating Data Quality

How do you know if your curation pipeline is working? Key evaluation approaches:

- **Benchmark-driven ablations**: Train small (1B parameter) models on different data variants; measure downstream benchmark scores.
- **Perplexity on held-out sets**: Track perplexity on curated held-out data (Wikipedia, books) to detect quality regressions.
- **Memorization audits**: Probe models for verbatim memorization of training data — high memorization rates indicate deduplication failures.
- **Bias and toxicity analysis**: Measure model outputs for demographic biases and toxic content correlated with training data composition.

## Landmark Open Datasets

| Dataset | Size | Sources | Key Features |
| --- | --- | --- | --- |
| The Pile | 825 GB | 22 diverse sources | First high-quality open multi-source dataset |
| RedPajama-v2 | 30 T tokens | Common Crawl + extras | Fully open, LLaMA recipe reproduction |
| Dolma | 3 T tokens | Web, code, science, books | Best-in-class documentation and data cards |
| FineWeb | 15 T tokens | Common Crawl | Quality-classifier filtering, state-of-the-art performance |
| ROOTS | 1.6 TB | Multilingual web + curated | Powers BLOOM, strong multilingual coverage |

## Synthetic Data at Scale

As the web approaches saturation for training high-quality LLMs, **synthetic data** — text generated by existing LLMs — is filling the gap:

- **Phi models** (Microsoft) demonstrated that small models trained on LLM-generated "textbook quality" data can punch far above their parameter count on reasoning benchmarks.
- **Self-instruct** and **Alpaca**: LLMs generating their own instruction-tuning data.
- **FineWeb-Edu**: A subset of FineWeb filtered for educational quality by a LLM classifier.

The risk: if future models are trained heavily on LLM-generated data, **model collapse** can occur — each generation's errors and biases get amplified, degrading overall capability. Active research is investigating how to mix synthetic and organic data sustainably.

## Data Governance and Legal Considerations

Training data curation increasingly intersects with legal and ethical obligations:

- **Copyright**: Does web-scraped text constitute fair use for training? Active litigation in multiple jurisdictions.
- **GDPR / Privacy**: Personal data in training corpora may violate data protection regulations in the EU and elsewhere.
- **Data provenance**: Increasingly, model cards and data cards document training data composition and filtering decisions for transparency and reproducibility.

Training data curation has evolved from an engineering afterthought into a core scientific discipline — one where careful decisions about quality, deduplication, and mixing have larger impacts on final model capability than architectural choices alone.
