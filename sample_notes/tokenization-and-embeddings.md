# Tokenization and Embeddings

## Tokenization: text to tokens

Models don't read characters or whole words directly — text is first split into
**tokens**. **Subword tokenization** (BPE, WordPiece, Unigram) strikes a balance
between character- and word-level: frequent words become single tokens, while
rare or unseen words split into sub-pieces. This keeps the vocabulary bounded
(typically tens of thousands of tokens) while still representing any input, and it
handles out-of-vocabulary words gracefully.

- **BPE** (byte-pair encoding) greedily merges the most frequent adjacent pairs.
- **WordPiece** (used by BERT) merges pairs that most increase corpus likelihood.

Token count is not word count: one word can be several tokens, which matters for
context-window limits and API cost (both counted in tokens).

## Embeddings: tokens to vectors

An **embedding** maps a token (or a whole passage) to a dense vector in a
continuous space where geometric closeness reflects meaning. Two kinds:

- **Static embeddings** (word2vec, GloVe) give each word one fixed vector
  regardless of context — "bank" has the same vector in *river bank* and *bank
  loan*.
- **Contextual embeddings** (from a Transformer encoder) give a token a different
  vector depending on its sentence, because self-attention mixes in surrounding
  context. This is why Transformer encoders are the backbone of modern retrieval
  embeddings.

## Embedding dimensionality

A passage embedding is a fixed-length vector (e.g. 384 or 768 dimensions). Higher
dimensions can capture more nuance but cost more memory and compute per
similarity comparison; the dimension is fixed by the embedding model.

## Why this matters for retrieval

Dense retrieval embeds passages and the query into the same space and ranks by
cosine similarity, so the embedding model's quality and training domain directly
bound retrieval quality. Sparse retrieval (BM25) instead matches on the tokens
themselves, which is why exact rare terms (identifiers, error codes) favour it.
