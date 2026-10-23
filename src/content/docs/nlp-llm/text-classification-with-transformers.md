---
title: Text Classification with Transformers
description: Learn the practical workflow for fine-tuning a pretrained transformer for text classification, including data preparation, training, evaluation, and when to use LLM prompting instead.
---

Text classification assigns labels to text: spam detection, topic labeling, intent recognition, toxicity moderation. Fine-tuning a pretrained encoder remains one of the most reliable and cost-efficient approaches.

## Why Transformers

Pretrained encoders such as BERT, RoBERTa, and DeBERTa already encode rich language knowledge. A small classification head on top of the [CLS] (or pooled) representation, fine-tuned on labeled examples, often beats classical TF-IDF models, especially with limited data.

## Workflow

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np, evaluate

ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained("distilroberta-base")

def prep(batch):
    return tok(batch["text"], truncation=True, max_length=256)

ds = ds.map(prep, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)

f1 = evaluate.load("f1")
def metrics(p):
    return f1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

args = TrainingArguments("out", learning_rate=2e-5, num_train_epochs=3,
                         per_device_train_batch_size=16, eval_strategy="epoch")
Trainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["test"], tokenizer=tok, compute_metrics=metrics).train()
```

## Key Decisions

- **Model size**: smaller distilled models are cheaper to serve with little accuracy loss on many tasks
- **Max length**: truncation can drop critical text; for long documents chunk and aggregate
- **Learning rate**: 1e-5 to 5e-5 is typical for full fine-tuning
- **Class imbalance**: use class weights, resampling, and metrics like macro-F1 or PR-AUC instead of accuracy
- **Multi-label tasks**: use sigmoid outputs with binary cross-entropy instead of softmax

## LLM Prompting Alternative

For tiny datasets or fast-changing label sets, zero-shot or few-shot LLM classification needs no training. For high volume and stable labels, a fine-tuned small model is usually cheaper, faster, and more consistent. A good pattern is to use an LLM to label data, then distill into a small classifier.

## Evaluation Checklist

- Hold out data that mirrors production
- Inspect the confusion matrix and error examples
- Calibrate confidence if you will threshold or route low-confidence cases to humans
- Monitor drift after deployment
