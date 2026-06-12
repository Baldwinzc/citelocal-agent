# Fine-Tuning and Transfer Learning

## Pretrain then adapt

Modern NLP follows a two-stage recipe: **pre-train** a large model on a broad,
self-supervised objective, then **adapt** it to a downstream task. Pre-training
learns general language structure from unlabelled text; adaptation specialises
that knowledge with far less labelled data than training from scratch would need.

## Fine-tuning

**Fine-tuning** continues training the pre-trained weights on a task's labelled
data, usually adding a small task-specific head. It typically reaches the best
task accuracy, but it **updates all the weights**, so each task needs its own full
copy of the model, and the knowledge is baked into the weights at training time.

## Parameter-efficient fine-tuning (PEFT)

Updating every weight is expensive at scale. **PEFT** methods freeze the base
model and train a small number of extra parameters:

- **Adapters** insert small bottleneck layers between existing layers.
- **LoRA** (low-rank adaptation) learns low-rank update matrices for the weights,
  adding <1% trainable parameters while matching full fine-tuning on many tasks.

PEFT lets one frozen base model serve many tasks by swapping small adapter weights.

## Catastrophic forgetting

Fine-tuning on a new task can overwrite knowledge the model had before —
**catastrophic forgetting**. PEFT mitigates it by leaving the base weights frozen.

## Fine-tuning vs retrieval for keeping knowledge current

Fine-tuning bakes facts into the weights, so updating knowledge means retraining,
and the model can still hallucinate facts it never saw. **Retrieval-augmented
generation** instead keeps knowledge in an external, swappable corpus the model
reads at inference time — better when facts change often or must be cited.
Fine-tuning is better for adapting *style, format, or task behaviour*; retrieval
is better for *injecting up-to-date, attributable facts*.
