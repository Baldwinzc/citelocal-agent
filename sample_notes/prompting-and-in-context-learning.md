# Prompting and In-Context Learning

## In-context learning

Large language models can perform a task from instructions and examples placed in
the **prompt**, without any weight updates — **in-context learning**. The model
infers the pattern from the context and applies it to the new input.

- **Zero-shot**: only an instruction, no examples.
- **Few-shot**: a handful of input→output examples precede the real input.

Because nothing is trained, in-context learning adapts behaviour instantly and
per-request, but it consumes context-window tokens and is sensitive to wording.

## In-context learning vs fine-tuning

They are two ways to specialise a model. **Fine-tuning** changes the weights —
durable, no per-request token cost, but needs labelled data and a training run.
**In-context learning** changes only the prompt — instant and data-free, but
re-paid in tokens every call and capped by the context window. Prompting is the
quick path; fine-tuning is the durable path for stable, high-volume behaviour.

## Chain-of-thought

Prompting the model to **reason step by step** ("chain-of-thought") before
answering improves performance on multi-step reasoning, at the cost of longer
outputs. It is a prompting technique, not a weight change.

## Prompt sensitivity

Model outputs can change noticeably with small prompt edits — wording, example
order, formatting. This brittleness is why prompts are versioned and evaluated
like code, and why a system prompt is kept explicit and stable.

## Prompting in a RAG system

In retrieval-augmented generation the retrieved passages are injected into the
prompt as context, and the instruction tells the model to answer **only** from
that context and to cite it. So RAG is, in part, a disciplined prompting pattern:
ground the generation in retrieved evidence rather than the model's parametric
memory.
