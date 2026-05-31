---
title: "Knowledge Distillation for Large Language Models"
description: "Learn how to transfer knowledge from large teacher models to smaller students — from vanilla distillation to gradient matching and先进的 distillation techniques that enable efficient deployment."
---

Knowledge distillation compresses the capabilities of a large model into a smaller one that can be deployed more efficiently. For LLMs, it's the key technique for getting frontier model performance on edge devices.

## Why Distill LLMs

Large language models like GPT-4 or Claude require massive compute for inference. Distillation trains smaller models to mimic their behavior, enabling:

- **Deployment on edge devices**: Run 7B models on phones and laptops.
- **Lower inference costs**: 10× cheaper inference with 70B → 7B compression.
- **Latency reduction**: Smaller models are faster to run.
- **On-device privacy**: User data never leaves the device.

## Vanilla Knowledge Distillation

The classic approach: train the student to match the teacher's output distribution.

```python
# Teacher model (frozen)
teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat")

# Student model
student = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Distillation loss: KL divergence between output distributions
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, soft_targets, reduction='batchmean')
```

The **temperature** parameter softens both distributions, revealing more information about the teacher's reasoning.

### Limitations of Vanilla Distillation

- **Output-only**: The student sees only the final output distribution, not the internal reasoning.
- **Capacity gap**: Small students struggle to fit large teacher distributions.
- **Token-level**: Doesn't capture document-level or conversational coherence.

## Advanced Distillation Techniques

### MiniLLM: Gradient Matching

MiniLLM (Gu et al., 2023) matches the direction of gradients instead of output distributions:

```python
def compute_gradient_match_loss(student, teacher, input_ids, target_ids):
    # Get student and teacher logits
    student_logits = student(input_ids, labels=target_ids).logits
    teacher_logits = teacher(input_ids, labels=target_ids).logits
    
    # Compute forward KL divergence (for regularization)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )
    
    # Backward pass to get gradients
    student_loss = kl_loss
    student_loss.backward()
    student_gradients = [p.grad.clone() for p in student.parameters()]
    
    # Compute target gradients from teacher (one backward per layer, no update)
    teacher_loss = compute_kl_divergence(teacher_logits)
    teacher_gradients = compute_target_gradients(teacher, teacher_loss)
    
    # Match gradients: minimize cosine distance between student and teacher gradients
    gradient_loss = sum(
        cosine_distance(sg, tg)
        for sg, tg in zip(student_gradients, teacher_gradients)
    ) / len(student_gradients)
    
    return kl_loss + 0.5 * gradient_loss
```

Gradient matching significantly outperforms vanilla distillation, especially for larger teacher-student gaps.

### GKD (Generalized Knowledge Distillation)

GKD formalizes distillation as a minimax game between student and teacher:

- **Forward KL**: Student learns from teacher's outputs.
- **Reverse KL**: Teacher distills into student's subspace.
- **Mode coverage**: Focuses on high-probability outputs rather than averaging over all.

### Self-Distillation

The student distills from itself at previous checkpoints. This is a form of regularization that prevents catastrophic forgetting:

```python
# At checkpoint N, distill from checkpoint N-1
student_current = load_checkpoint("step_1000")
student_previous = load_checkpoint("step_500")

# Use previous checkpoint as teacher for current
loss = distillation_loss(
    student_current(outputs),
    student_previous(outputs).detach()
)
```

## Representation Distillation

Distill intermediate representations, not just outputs:

```python
class RepresentationDistillationLoss(nn.Module):
    def forward(self, student_reprs, teacher_reprs):
        # Project to common space
        student_proj = self.projector(student_reprs)
        teacher_proj = self.projector(teacher_reprs)
        
        # Align representations
        return cosine_similarity_loss(student_proj, teacher_proj)
```

Matching representations helps the student learn internal information flow from the teacher.

## Data Selection for Distillation

Not all data is equally useful for distillation:

### Importance Weighting
Select data where the teacher is most "certain" or most "capable":

```python
def score_distillation_data(teacher, data):
    scores = []
    for batch in data:
        logits = teacher(batch).logits
        entropy = -F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        confidence = 1 - entropy.mean()  # Higher = more confident
        scores.append((batch, confidence))
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

### Curriculum Distillation
Start distilling on easier examples, progressively increase difficulty:

```python
# Phase 1: Short, simple responses
# Phase 2: Medium complexity
# Phase 3: Long-form, complex reasoning
```

## Metrics for Distillation Quality

| Metric | Description |
|--------|-------------|
| **Output KL divergence** | How close is student output to teacher? |
| **Task accuracy** | Does student match teacher on tasks? |
| **Length alignment** | Does student produce similar length outputs? |
| **Perplexity gap** | Student's perplexity on held-out text |
| **Human evaluation** | Can humans distinguish outputs? |

## When Distillation Works Best

Distillation is most effective when:

1. **Teacher and student share architecture**: Easier when using the same base model family.
2. **Task is well-defined**: Distilling for a specific task is easier than general knowledge.
3. **Sufficient data**: More high-quality distillation data improves results.
4. **Teacher is much stronger**: There's actually something to learn from the teacher.

## Practical Distillation Pipeline

```python
def distill(teacher_name, student_name, dataset, output_dir):
    teacher = load_model(teacher_name)
    student = load_model(student_name)
    
    for epoch in range(num_epochs):
        for batch in dataloader(dataset):
            # Generate teacher responses
            with torch.no_grad():
                teacher_outputs = teacher.generate(
                    batch.prompts,
                    max_new_tokens=512,
                    temperature=0.7
                )
            
            # Compute distillation loss
            student_logits = student(batch.prompts, labels=teacher_outputs)
            loss = distillation_loss(student_logits, teacher_outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Evaluate on held-out tasks
        evaluate(student, tasks)
        
    # Save distilled student
    student.save_pretrained(output_dir)
```

Knowledge distillation has enabled the deployment of frontier AI capabilities on consumer hardware. As models continue to grow, distillation becomes increasingly important for making these capabilities accessible.