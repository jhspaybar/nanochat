"""
L3: Large Lookup Layers
Ref: arXiv:2601.21461v2

L3 generalizes token embeddings by placing per-token lookup tables inside
the decoder stack. Unlike MoE, routing is static (determined by token ID),
eliminating router training and load-balancing losses.
"""

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lzw_allocation(token_sequences, vocab_size, n_emb, k_max):
    """
    Compute per-token embedding allocation using LZW-style frequency analysis.

    Follows Algorithm 1 from the L3 paper (arXiv:2601.21461v2):
    1. Scan corpus LZW-style: at each position find the longest prefix already
       in the dictionary, increment its count, add the one-longer prefix as new.
    2. Sort all discovered n-gram codewords by frequency (descending).
    3. Iterate through codewords: add an embedding to the LAST token of each
       frequent n-gram (if that token hasn't hit k_max).
    4. Stop when total embeddings reach n_emb.

    Args:
        token_sequences: list of token ID lists (training data sample)
        vocab_size: size of vocabulary
        n_emb: target total embeddings
        k_max: max embeddings per token
    Returns:
        alloc: list[int] of length vocab_size (embeddings per token)
    """
    assert n_emb >= vocab_size, f"n_emb ({n_emb}) must be >= vocab_size ({vocab_size})"

    # Phase 1: Build LZW dictionary with frequency counts
    # Initialize with single-token entries (count 0)
    lzw_counter = {}
    for s in range(vocab_size):
        lzw_counter[(s,)] = 0

    # Scan each sequence LZW-style
    for seq in token_sequences:
        toks = seq
        last = 0
        cur = 1
        while cur < len(toks):
            # Find longest known prefix starting at `last`
            while cur < len(toks) and tuple(toks[last:cur]) in lzw_counter:
                cur += 1
            if cur > last + 1:
                # Increment count of the known prefix
                lzw_counter[tuple(toks[last:cur - 1])] += 1
                # Add the one-longer string as a new entry
                lzw_counter[tuple(toks[last:cur])] = 1
            last = cur
            cur += 1

    # Phase 2: Sort codewords by frequency (descending)
    sorted_codewords = sorted(lzw_counter.items(), key=lambda x: x[1], reverse=True)

    # Phase 3: Allocate embeddings
    # Every token starts with 1 embedding
    alloc = [1] * vocab_size
    n_alloc = vocab_size

    if n_alloc >= n_emb:
        return alloc

    # Iterate through codewords, adding embeddings to the last token of each
    i = 0
    while n_alloc < n_emb and i < len(sorted_codewords):
        last_token = sorted_codewords[i][0][-1]  # last token of the n-gram
        if alloc[last_token] < k_max:
            alloc[last_token] += 1
            n_alloc += 1
        i += 1

    # If we still haven't reached n_emb (ran out of codewords), distribute remaining
    while n_alloc < n_emb:
        added_any = False
        for tok in range(vocab_size):
            if n_alloc >= n_emb:
                break
            if alloc[tok] < k_max:
                alloc[tok] += 1
                n_alloc += 1
                added_any = True
        if not added_any:
            break

    return alloc


def allocation_to_bounds(alloc):
    """
    Convert allocation array to cumulative bounds tensor.

    bounds[0] = 0, bounds[i] = bounds[i-1] + alloc[i-1]
    bounds[-1] = sum(alloc) = n_emb

    Args:
        alloc: list[int] of per-token allocation counts
    Returns:
        bounds: torch.LongTensor of shape [len(alloc) + 1]
    """
    bounds = [0]
    for a in alloc:
        bounds.append(bounds[-1] + a)
    return torch.tensor(bounds, dtype=torch.long)


class L3Layer(nn.Module):
    """
    L3 layer: per-token lookup table with attention-like aggregation.

    Forward pass (gather+pad approach):
    1. Look up bounds for each token, gather KV embeddings, pad to k_max
    2. Compute scores = K @ x_norm, mask invalid positions, softmax
    3. Aggregate: weighted sum of V embeddings
    4. Up-project, RMSNorm, concat with x, mix-project
    Returns the delta (added residually by caller).
    """

    def __init__(self, n_embd, n_emb, d_up, tie_kv=True, vocab_size=0, k_max=32):
        super().__init__()
        self.n_embd = n_embd
        self.n_emb = n_emb
        self.d_up = d_up
        self.tie_kv = tie_kv
        self.k_max = k_max

        if tie_kv:
            # Single shared weight for both keys and values
            self.kv_weight = nn.Parameter(torch.empty(n_emb, n_embd))
        else:
            # Separate key and value weights
            self.k_weight = nn.Parameter(torch.empty(n_emb, n_embd))
            self.v_weight = nn.Parameter(torch.empty(n_emb, n_embd))

        # Up-project from d_emb (= n_embd when tied) to d_up
        self.w_up = nn.Linear(n_embd, d_up, bias=False)
        # Mix-project: concat(up_projected, x) -> n_embd
        self.w_mix = nn.Linear(d_up + n_embd, n_embd, bias=False)

        # Bounds buffer: sized for vocab so checkpoint loading works without shape mismatch
        bounds_size = (vocab_size + 1) if vocab_size > 0 else 1
        self.register_buffer("bounds", torch.zeros(bounds_size, dtype=torch.long), persistent=True)

    def set_bounds(self, bounds):
        """Register the precomputed bounds tensor as a buffer."""
        self.bounds = bounds
        # Precompute k_max from bounds
        alloc = bounds[1:] - bounds[:-1]
        self.k_max = int(alloc.max().item())

    def _attend_chunk(self, x_norm_chunk, starts_chunk, lengths_chunk, k_max):
        """Process a chunk of tokens through the L3 attention mechanism.

        Args:
            x_norm_chunk: [N, C] normalized hidden states for this chunk
            starts_chunk: [N] start indices into kv_weight
            lengths_chunk: [N] number of valid embeddings per token
            k_max: max embeddings (for padding)
        Returns:
            agg: [N, C] aggregated embeddings
        """
        N, C = x_norm_chunk.shape
        device = x_norm_chunk.device

        # Build index tensor [N, k_max] with valid indices and padding
        offsets = torch.arange(k_max, device=device).unsqueeze(0)  # [1, k_max]
        indices = starts_chunk.unsqueeze(1) + offsets               # [N, k_max]
        mask = offsets < lengths_chunk.unsqueeze(1)                  # [N, k_max]

        # Clamp indices to valid range (masked positions will be zeroed out)
        indices = indices.clamp(0, self.n_emb - 1)

        # Gather weights
        if self.tie_kv:
            kv = self.kv_weight[indices]  # [N, k_max, C]
            k_emb = kv
            v_emb = kv
        else:
            k_emb = self.k_weight[indices]  # [N, k_max, C]
            v_emb = self.v_weight[indices]  # [N, k_max, C]

        # Attention scores: K @ x_norm
        scores = torch.bmm(k_emb, x_norm_chunk.unsqueeze(2)).squeeze(2)  # [N, k_max]
        scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax + aggregate
        weights = F.softmax(scores, dim=-1)  # [N, k_max]
        weights = weights.masked_fill(~mask, 0.0)
        agg = torch.bmm(weights.unsqueeze(1), v_emb).squeeze(1)  # [N, C]
        return agg

    def forward(self, x, token_ids):
        """
        Args:
            x: [B, T, n_embd] hidden states
            token_ids: [B, T] token IDs
        Returns:
            delta: [B, T, n_embd] to be added residually by caller
        """
        B, T, C = x.shape

        # 1. RMSNorm the input
        x_norm = F.rms_norm(x, (C,))

        # 2. Look up bounds for each token
        flat_ids = token_ids.reshape(-1)  # [B*T]
        starts = self.bounds[flat_ids]     # [B*T]
        ends = self.bounds[flat_ids + 1]   # [B*T]
        lengths = ends - starts            # [B*T]
        k_max = self.k_max
        x_norm_flat = x_norm.reshape(B * T, C)

        # 3. Process in chunks to keep tensor sizes under INT_MAX for MPS compatibility
        # Each chunk produces [chunk, k_max, n_embd] tensors; limit to ~1B elements
        max_chunk = max(1, (2**30) // max(k_max * C, 1))
        N = B * T

        if N <= max_chunk:
            agg = self._attend_chunk(x_norm_flat, starts, lengths, k_max)
        else:
            chunks = []
            for i in range(0, N, max_chunk):
                j = min(i + max_chunk, N)
                chunks.append(self._attend_chunk(
                    x_norm_flat[i:j], starts[i:j], lengths[i:j], k_max))
            agg = torch.cat(chunks, dim=0)

        agg = agg.view(B, T, C)

        # 4. Up-project, RMSNorm, concat with x, mix-project
        up = self.w_up(agg)                    # [B, T, d_up]
        up = F.rms_norm(up, (self.d_up,))      # normalize
        cat = torch.cat([up, x], dim=-1)       # [B, T, d_up + n_embd]
        delta = self.w_mix(cat)                # [B, T, n_embd]

        return delta
