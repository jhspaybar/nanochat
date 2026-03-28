# L3: Large Lookup Layers (arXiv:2601.21461v2)

**Authors**: Albert Tseng, Christopher De Sa (Cornell)

## Core Idea

L3 generalizes the tokenizer embedding table into decoder layers. Each token ID gets a set of learned K/V embeddings (a "lookup table"). During forward pass, the hidden state attends to these embeddings to produce a context-dependent aggregation. Unlike MoE, routing is **static** (determined by token ID, not hidden state), eliminating router training and load-balancing losses.

## Architecture

For a single token with hidden state `x` and token ID `t`:

```
L3(x, t) = W_mix [ LayerNorm(W_up(V_t^T Softmax(K_t x))) ; x ]
```

Where:
- `K_t` is shape `(d_t, d_in)` — per-token key embeddings
- `V_t` is shape `(d_t, d_emb)` — per-token value embeddings
- `d_t` varies per token (determined by LZW allocation)
- `W_up` is shape `(d_up, d_emb)` — up-projection
- `W_mix` is shape `(d_out, d_in + d_up)` — mix-projection
- LayerNorm is RMSNorm

**L3 sits BETWEEN decoder blocks**, not inside them. It modifies the residual stream, then the next decoder block processes the enriched hidden states normally.

Key parameters from the paper:
- `v = 710,000` total embeddings across all tokens
- `k = 512` max embeddings per token
- `d_emb = 512` (800M), `1024` (2.6B) — typically half the hidden size
- `d_up = 4096` constant across sizes
- Weight tying K=V has no quality impact but halves sparse params

## LZW Allocation Algorithm

The paper's LZW allocation is critical for quality (uniform allocation performs much worse). The algorithm:

1. Scan corpus LZW-style: find longest known prefix, increment its count, add the one-longer prefix with count 1
2. Sort all n-gram codewords by frequency (descending)
3. Iterate through codewords: for each, add an embedding to the **last token** of that codeword (if that token hasn't hit k_max)
4. Stop when total embeddings reach target v

Reference Python implementation from the paper:
```python
def train_lzw(files, tok, k):
    lzw_counter = {}
    for s in range(tok.n_vocab):
        lzw_counter[(s,)] = 0
    for fn in files:
        f = open(fn).readlines()
        for line in f:
            toks = tok.encode(line)
            last = 0
            cur = 1
            while cur < len(toks):
                while cur < len(toks) and tuple(toks[last:cur]) in lzw_counter:
                    cur += 1
                if cur > last+1:
                    lzw_counter[tuple(toks[last:cur-1])] += 1
                    lzw_counter[tuple(toks[last:cur])] = 1
                last = cur
                cur += 1
    lzw_counter = sorted(list(lzw_counter.items()), key=lambda x: x[1], reverse=True)
    alloc = [1 for _ in range(tok.n_vocab)]
    n_alloc = tok.n_vocab
    i = 0
    while n_alloc < target:
        if alloc[lzw_counter[i][0][-1]] < k:
            alloc[lzw_counter[i][0][-1]] += 1
            n_alloc += 1
        i += 1
    return alloc
```

## Key Experimental Results

- L3 layers placed between decoder blocks (e.g., after layers 4 and 16 in a 20-layer model)
- Best placement: middle of the model (too early = not enough context; too late = not enough impact)
- 2-4x sparsity ratios achieved with 1-2 L3 layers
- L3 outperforms iso-FLOP dense models AND iso-sparse MoEs at all tested sizes
- Training speed: ~87% of dense throughput (800M model, 8xA100)
- Inference: CPU-offloaded L3 adds minimal overhead (params prefetched during pre-L3 compute)
- Tuned lens shows sharp KL drops at L3 layer positions, indicating cached information

## Differences from Our nanochat Implementation

### 1. LZW Algorithm (MAJOR)
Our `compute_lzw_allocation` only counts unigram + bigram frequencies and does simple frequency-proportional allocation. The paper uses a proper LZW scan that discovers arbitrary-length n-grams and allocates embeddings to the **last token** of frequent n-grams. This is the key insight — the paper's uniform allocation ablation shows it performs much worse.

### 2. d_emb Dimension (MINOR for tied case)
The paper uses a separate `d_emb` dimension for V embeddings (typically half hidden size). Our implementation uses `d_emb = n_embd` when tied. Since tying K=V forces `d_emb = d_in`, this only matters for untied mode.

### 3. Sorted Batch Training (OPTIMIZATION)
The paper sorts tokens in each batch to form block-diagonal attention masks for better memory access. Our implementation uses a simple gather+pad approach that works but is less efficient.

### 4. Forward Pass Structure (CORRECT)
Our implementation correctly places L3 between decoder blocks as a residual addition: `x = x + L3(x, token_ids)`. The L3 layer does NOT feed into the standard attention K/V of the next layer — it modifies the residual stream.

## Relevance to nanochat

- L3 is orthogonal to MoE and can be combined with it
- The paper uses Llama-style architecture, similar to nanochat's GPT
- Priority fix: replace our simplified LZW with the paper's actual algorithm
- Consider adding `d_emb` as a separate parameter (default to n_embd for tied)
- Consider sorted-batch optimization for training throughput
