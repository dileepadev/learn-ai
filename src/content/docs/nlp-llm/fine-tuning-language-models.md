---
title: Fine-tuning Language Models - Adapting Pre-trained Models to Your Domain
description: Techniques for fine-tuning LLMs for specific tasks, domains, and use cases.
---

While pre-trained language models are incredibly capable, they often benefit from fine-tuning on task-specific or domain-specific data. This post explores how to adapt models to your needs.

## Why Fine-Tune?

Pre-trained models are general purpose, but specific tasks may need customization.

### Reasons to Fine-Tune

**Domain Specificity:**
- Medical LLM: Needs medical terminology and knowledge
- Legal LLM: Needs legal concepts and phrasing
- Scientific LLM: Needs research concepts

**Task-Specific Behavior:**
- Classification: Optimize for categorizing text
- Summarization: Learn to produce concise summaries
- Code Generation: Generate code for your domain

**Performance:**
- Custom data: Train on exact domain
- Rare distributions: Handle edge cases
- Performance: Better on your specific metrics

**Cost:**
- Smaller fine-tuned model: Better than querying large API
- Faster inference: Fewer parameters
- Privacy: Data stays with you

## Fine-Tuning vs Prompt Engineering

### When to Use Prompt Engineering

- Limited data (< 100 examples)
- Don't want to manage model
- Task is straightforward
- Want flexibility (change task easily)

### When to Use Fine-Tuning

- Lots of data (1000+ examples)
- Consistent task/domain
- Performance is critical
- Cost-sensitive (inference at scale)
- Need privacy

```
Prompt Engineering: Quick, flexible, limited
Fine-tuning: More work, specialized, better performance
```

## Fine-Tuning Process

### Step 1: Data Preparation

**Gather Examples:**
- Collect examples of inputs and desired outputs
- At least 100-1000 high-quality examples
- More data usually means better results
- Quality matters more than quantity

**Format Data:**
```
For next-token prediction:
{
  "text": "Question: What is ML? Answer: Machine learning is..."
}

For classification:
{
  "text": "This product is great!",
  "label": "positive"
}

For instruction tuning:
{
  "prompt": "Summarize this text: [text]",
  "completion": "[summary]"
}
```

### Step 2: Data Preprocessing

**Tokenization:**
- Convert text to tokens
- Use same tokenizer as pre-trained model

**Splitting:**
- Training set: 80% of data
- Validation set: 10% of data
- Test set: 10% of data

**Cleaning:**
- Remove duplicates
- Fix obvious errors
- Check for biases

### Step 3: Training

**Setup:**
1. Load pre-trained model
2. Configure training parameters
3. Set up optimizer and loss function
4. Create data loaders

**Hyperparameters:**

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| **Learning Rate** | 1e-5 to 5e-5 | Too high: Lose pre-training. Too low: Slow |
| **Batch Size** | 8-32 | Larger: Faster, needs more memory |
| **Epochs** | 2-5 | More: Potential overfitting |
| **Warmup Steps** | 10% of total | Gradual learning rate increase |
| **Weight Decay** | 0.01 | Regularization |

**Monitoring:**
- Track training loss (should decrease)
- Track validation loss (watch for overfitting)
- Track task-specific metrics

### Step 4: Evaluation

**On Validation Set:**
- Monitor loss
- Calculate task metrics
- Check for overfitting

**Validation Metrics by Task:**

**Classification:**
- Accuracy
- Precision, Recall, F1
- ROC-AUC

**Generation:**
- BLEU (translation)
- ROUGE (summarization)
- Perplexity

**Generation Quality:**
- Human evaluation
- Semantic similarity

### Step 5: Testing

**Hold-out Test Set:**
- Final performance estimate
- Don't use for any decisions during training
- Represents real-world performance

**Error Analysis:**
- What does model get wrong?
- Any systematic failures?
- Insights for improvement

### Step 6: Deployment

**Model Optimization:**
- Quantization: Reduce precision (int8 instead of float32)
- Pruning: Remove less important weights
- Distillation: Train smaller model from larger

**Serving:**
- Local deployment
- API endpoint
- Edge device

## Fine-Tuning Strategies

### Full Fine-Tuning

Update all model weights.

**Pros:**
- Maximum customization
- Best performance

**Cons:**
- Requires lots of computation
- Needs lots of memory
- Risk of catastrophic forgetting

### Parameter-Efficient Fine-Tuning

Update only small fraction of parameters.

#### LoRA (Low-Rank Adaptation)

Add small trainable matrices alongside frozen weights.

**How It Works:**
```
Output = (Base_Weight + LoRA_A @ LoRA_B) @ Input
         └─────────────────────────────┘
         Only ~0.1% additional parameters
```

**Advantages:**
- 10x less parameters to train
- 3-4x faster training
- Easily switch between tasks
- Fits in smaller memory

**Disadvantages:**
- Slightly lower performance than full fine-tuning
- Hyperparameter tuning needed

#### Prefix Tuning

Train prefix prepended to prompts.

**How It Works:**
```
Prefix (trainable) + Your Task = Full Prompt
```

**Use Case:** Multiple tasks with single model

#### Adapter Modules

Add small modules between layers.

**How It Works:**
```
Layer Output → Adapter (trainable) → Next Layer
              └─ Only adapter trains
```

**Benefits:**
- Modular
- Task-specific modules
- Share base model

### Instruction Tuning

Fine-tune on task examples with instructions.

```
Input: "Classify sentiment: I love this product"
Output: "positive"

Input: "Translate to French: Hello"
Output: "Bonjour"
```

**Benefit:** Model learns to follow instructions

### Reinforcement Learning from Human Feedback (RLHF)

Fine-tune using human preferences.

**Process:**
1. Generate multiple outputs from model
2. Humans rank them
3. Train reward model on rankings
4. Use reward model to fine-tune model
5. Iterate

**Result:** Model learns what humans prefer

## Practical Fine-Tuning Example

### Task: Sentiment Classification for Product Reviews

**Step 1: Data**
```
Dataset: 1000 labeled reviews
{
  "text": "Best product ever! Works perfectly.",
  "label": "positive"
},
{
  "text": "Broke after one day. Waste of money.",
  "label": "negative"
}
```

**Step 2: Preprocessing**
```
- Remove duplicates (5 duplicates removed)
- Split: 800 train, 100 valid, 100 test
- Tokenize with model's tokenizer
```

**Step 3: Training**
```
- Model: DistilBERT
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Training time: ~2 minutes (on GPU)
```

**Step 4: Results**
```
Training Loss:
- Epoch 1: 0.45
- Epoch 2: 0.25
- Epoch 3: 0.15

Validation Accuracy:
- Epoch 1: 92%
- Epoch 2: 94%
- Epoch 3: 95%

Test Accuracy: 94%
Precision: 0.95, Recall: 0.93
```

## Common Challenges and Solutions

### Overfitting

**Problem:** Model memorizes training data, poor on new data

**Signs:**
- Training loss: 0.05, Validation loss: 0.30

**Solutions:**
- Reduce training time (fewer epochs)
- Increase batch size
- Add regularization (weight decay)
- More training data
- Use smaller model

### Catastrophic Forgetting

**Problem:** Fine-tuning destroys pre-trained knowledge

**Signs:**
- Performance on general tasks drops

**Solutions:**
- Use low learning rate (1e-5 to 5e-5)
- Use small number of epochs (2-5)
- Use LoRA instead of full fine-tuning
- Mix general and task data

### Insufficient Data

**Problem:** Not enough examples

**Solutions:**
- Data augmentation (paraphrase, back-translate)
- Few-shot learning (prompt engineering)
- Transfer learning (start from related model)
- Use LoRA (needs less data)

### Slow Training

**Problem:** Takes too long

**Solutions:**
- Use smaller model (DistilBERT instead of BERT)
- Use LoRA (fewer parameters)
- Reduce batch size (faster but less stable)
- Use quantization
- Use multiple GPUs

## Tools and Frameworks

### Hugging Face Transformers

```python
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    callbacks=[EarlyStoppingCallback()]
)

trainer.train()
```

### Unsloth

Optimized fine-tuning library:
- 2-5x faster training
- 60% less memory usage
- Simple API

### LLaMA-LoRA

Fine-tuning large models with LoRA:
- Efficient training
- Web interface
- Community models

## Cost Considerations

### Computational Cost

**Full Fine-tuning:**
- GPU: Hours to days
- Cost: $10-1000+ per training

**LoRA Fine-tuning:**
- GPU: Minutes to hours
- Cost: $1-100 per training

### Data Cost

- Manual labeling: Expensive
- Crowdsourcing: Moderate
- Existing data: Free
- Synthetic data: Low cost but lower quality

### Inference Cost

- Local: Compute cost only
- API: Per-token cost
- Fine-tuned smaller model: Lower cost than large model

## When Not to Fine-Tune

- Task is simple, prompt engineering works
- Data is limited (< 50 examples)
- No computational resources
- Need frequent model updates
- Data is constantly changing

Use prompt engineering instead.

## Conclusion

Fine-tuning adapts pre-trained models to specific tasks and domains. The process involves preparing data, training with careful hyperparameter selection, and evaluating performance. Modern techniques like LoRA make fine-tuning accessible and efficient. Understanding when to fine-tune vs prompt engineer, and how to address common challenges, enables building effective custom AI systems. Fine-tuning bridges the gap between general pre-trained models and specialized domain-specific performance.
