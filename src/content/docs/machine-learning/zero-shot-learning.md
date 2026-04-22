---
title: Zero-Shot Learning
description: Understand zero-shot learning — how AI models generalize to unseen classes and tasks without any labeled examples, using semantic embeddings, natural language descriptions, and foundation model representations.
---

**Zero-shot learning (ZSL)** is a machine learning paradigm in which a model successfully classifies, retrieves, or reasons about categories it has **never seen any labeled examples of during training**. Instead of requiring examples of every possible class, zero-shot models leverage auxiliary information — semantic descriptions, attribute vectors, or natural language — to generalize to novel classes at inference time.

This capability is increasingly critical as AI systems are deployed in open-world settings where new categories constantly emerge and it is infeasible to collect labeled examples for every possible case.

## The Zero-Shot Problem Setup

In a conventional supervised learning setup, the training classes and test classes are identical. Zero-shot learning explicitly separates them:

- **Seen classes** $\mathcal{S}$: Classes with labeled training examples.
- **Unseen classes** $\mathcal{U}$: Classes with **no** labeled examples — only a semantic description.
- **Semantic space**: A shared embedding space (attributes, word vectors, or language model representations) where seen and unseen classes can both be represented.

The model learns a mapping from the visual or textual input space to the semantic space using seen classes. At test time, it uses the same mapping to classify inputs from unseen classes by finding the nearest class representation in semantic space.

## Why Zero-Shot Learning Works

The key insight is that **classes can be described without examples**. A model that has learned from thousands of animal categories develops rich visual representations that capture concepts like "stripes," "four legs," "wings," and "beak." A new animal — say, an Okapi — can be described as having "zebra-like stripes on its legs and a giraffe-like body." The model can locate Okapi in its learned representation space without ever seeing a photograph of one.

The mechanism relies on two conditions:

1. **Transferable features**: The features learned on seen classes must be general enough to apply to unseen classes.
2. **Informative semantics**: The semantic descriptions must capture properties that correspond to learnable visual or textual features.

## Semantic Representations

The quality of zero-shot learning depends critically on how unseen classes are described.

### Attribute Vectors

The earliest ZSL approaches used **manually defined attribute vectors** — binary or continuous-valued feature vectors where each dimension corresponds to a human-defined property.

For example, in the Animals with Attributes (AwA) benchmark:

| Animal | Stripes | Four-legged | Domestic | Ocean |
|--------|---------|-------------|----------|-------|
| Zebra | 1 | 1 | 0 | 0 |
| Tiger | 1 | 1 | 0 | 0 |
| Dolphin | 0 | 0 | 0 | 1 |

**Limitation**: Requires expensive manual annotation of attributes for every class, and attributes must be decided in advance — you cannot retroactively add a new attribute without re-annotating everything.

### Word Embeddings

**Word2Vec, GloVe, and FastText** embeddings represent class names as dense vectors in a semantic space where related concepts are geometrically close. A model can use the word vector for "okapi" as its class representation without any manual attribute work.

Limitation: Word embeddings encode only the surface semantics of a class name, not rich descriptive attributes. They work well for classes with distinctive names but poorly for classes with ambiguous names.

### Natural Language Descriptions

**Large language models** provide rich, flexible semantic representations via textual descriptions. Instead of a fixed attribute vector, a class is described in free-form text:

> "The okapi is a mammal native to the Democratic Republic of Congo. It has a dark brown coat with horizontal white stripes on the legs reminiscent of a zebra, despite being more closely related to the giraffe."

A text encoder (e.g., the text tower of CLIP) converts this description to a vector that can serve as the class embedding.

## CLIP: The Defining Zero-Shot Architecture

**CLIP (Contrastive Language-Image Pre-Training)**, released by OpenAI in 2021, demonstrated surprisingly strong zero-shot image classification without any task-specific training.

### How CLIP Works

CLIP trains two encoders jointly:

- An **image encoder** (ViT or ResNet) that maps images to a 512-dimensional embedding.
- A **text encoder** (Transformer) that maps text descriptions to the same embedding space.

Training objective: maximize cosine similarity between the embeddings of matched image-text pairs and minimize similarity for mismatched pairs — a contrastive objective over 400 million web-scraped (image, text) pairs.

### Zero-Shot Classification with CLIP

At inference time, no fine-tuning is needed:

1. For each candidate class, create a text prompt: `"a photo of a {class_name}"`.
2. Encode all class prompts with the text encoder → class embeddings.
3. Encode the query image with the image encoder → image embedding.
4. Classify the image as the class whose embedding has the highest cosine similarity.

This achieves competitive accuracy on many standard image classification benchmarks **without any labeled examples from those benchmarks** — a remarkable demonstration of zero-shot generalization.

### Prompt Engineering for CLIP

The exact phrasing of class descriptions significantly affects CLIP's zero-shot performance. **Prompt engineering** strategies include:

- Using domain-appropriate templates: `"a satellite image of {class_name}"` for remote sensing tasks.
- **Prompt ensembling**: Averaging embeddings across multiple prompts (e.g., `"a photo of a {class}"`, `"a picture of the {class}"`, `"an image of a {class}"`) reduces variance.
- **CuPL (Concept Unification through Prompt Learning)**: Using GPT-4 to generate diverse, descriptive prompts for each class and averaging their embeddings.

## Generalized Zero-Shot Learning

Standard ZSL evaluates only on unseen classes. **Generalized zero-shot learning (GZSL)** is the more realistic setting where the test set contains examples from **both seen and unseen classes**, and the model must classify without knowing which type of class a test example belongs to.

GZSL is significantly harder because models strongly bias toward seen classes (which dominated training). Mitigations include:

- **Calibrated stacking**: Adding a learned bias term to unseen class scores to compensate for the systematic seen-class preference.
- **Generative ZSL**: Training a conditional generator (VAE or GAN) to synthesize fake features for unseen classes using their semantic embeddings. The model is then trained on a mix of real seen-class features and synthetic unseen-class features — converting ZSL into a conventional supervised problem.

## Zero-Shot Learning in NLP

Zero-shot learning has a natural expression in **natural language processing**:

### Zero-Shot Text Classification

Using LLMs, text can be classified into categories without any labeled examples by framing classification as a **natural language inference (NLI)** task:

- **Premise**: The article text.
- **Hypothesis**: "This article is about sports."
- **Label**: Does the text entail the hypothesis?

Models like `facebook/bart-large-mnli` achieve strong zero-shot text classification this way.

Alternatively, instruction-following LLMs can be prompted directly:

> "Classify the following review as Positive, Negative, or Neutral. Review: [text]"

### Zero-Shot Named Entity Recognition

Framing NER as a question-answering task enables zero-shot extraction of entity types never seen during training: "What is the *product name* mentioned in the following text?" enables extraction of product entities even if the model was only trained on person and location entity types.

### Cross-Lingual Zero-Shot Transfer

Multilingual models trained on high-resource languages generalize to low-resource languages with no labeled data in the target language. A model trained for sentiment analysis in English can achieve reasonable performance in Swahili or Bengali purely through the shared multilingual representation space.

## Zero-Shot Object Detection

Zero-shot learning extends naturally to **object detection** — localizing objects of classes not seen during training.

**OWL-ViT** (Open-Vocabulary Object Detection with Vision Transformers) uses CLIP-style vision-language alignment to detect arbitrary objects from text queries at inference time. Given a query like `"a cracked pipe"`, the model can localize this object in an image even if "cracked pipe" was never a training category.

**Grounding DINO** combines CLIP with a detection backbone to enable open-set detection guided by natural language descriptions.

## Few-Shot vs. Zero-Shot vs. One-Shot

It is useful to situate zero-shot learning in the broader taxonomy:

| Setting | Examples per novel class | Key mechanism |
|---------|--------------------------|---------------|
| **Zero-shot** | 0 | Semantic descriptions or embeddings |
| **One-shot** | 1 | Similarity to the single example |
| **Few-shot** | 2–20 | Prototype, matching, or in-context learning |
| **Full supervised** | Hundreds–thousands | Direct optimization on examples |

**In-context learning** in LLMs is also sometimes called zero-shot (with no examples in the prompt) or few-shot (with example demonstrations), but operates through a different mechanism than classical ZSL.

## Evaluation Benchmarks

| Benchmark | Domain | Seen/Unseen split |
|-----------|--------|-------------------|
| **Animals with Attributes 2 (AwA2)** | Animals | 40 seen / 10 unseen |
| **CUB-200-2011** | Bird species | 150 seen / 50 unseen |
| **SUN database** | Scene recognition | 645 seen / 72 unseen |
| **ImageNet** | General image classification | 1000-class; zero-shot = no task training |

## Limitations and Open Problems

**Hubness problem**: In high-dimensional semantic spaces, certain class embeddings become "hubs" — they are the nearest neighbor of many test instances regardless of true class, degrading ZSL accuracy. Specialized distance metrics and normalization techniques partially address this.

**Semantic gap**: The learned mapping from visual to semantic space may not align perfectly with the manually defined attributes, especially when visual features capture subtleties not represented in the semantic description.

**Domain shift**: When test images come from a different distribution than training images (e.g., different image styles or capture conditions), zero-shot performance degrades.

**Evaluation contamination**: Foundation models like CLIP are trained on internet-scale data that may include benchmark test images. True zero-shot generalization requires careful curation of evaluation sets that are genuinely novel with respect to the pretraining data.

## Practical Applications

Zero-shot learning has practical value across many real-world scenarios:

- **Medical imaging**: Classifying rare diseases with few or no historical images by using textual descriptions from medical literature.
- **Industrial defect detection**: Identifying new types of manufacturing defects from a natural language description without collecting defect examples.
- **E-commerce product classification**: Adding new product categories to a taxonomy without re-labeling the entire catalog.
- **Content moderation**: Detecting new types of harmful content as they emerge, using policy descriptions rather than example images.
- **Scientific discovery**: Identifying novel chemical compounds, astronomical objects, or biological structures described in literature.

Zero-shot learning has moved from an academic research topic to a foundational capability of modern AI systems — largely through the success of large-scale vision-language pretraining. As foundation models grow more capable, the gap between zero-shot and fine-tuned performance continues to narrow.
