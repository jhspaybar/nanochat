"""
L3: Large Lookup Layers
Ref: arXiv:2601.21461v2

L3 generalizes token embeddings by placing per-token lookup tables inside
the decoder stack. Unlike MoE, routing is static (determined by token ID),
eliminating router training and load-balancing losses.

Forward pass uses the block-diagonal approach from Section A.3.4 of the paper:
1. Sort all tokens by ID (groups identical tokens together)
2. Build de-duplicated embedding pool for the sorted sequence
3. Process blocks of bb sorted tokens with masked attention
   (each token attends only to its own embeddings via masking)
4. Unsort results back to original order
5. Up-project, norm, concat with input, mix-project

This avoids padding every token to k_max — tokens share embedding pools
within blocks, and masking handles per-token selection. Much more memory-
efficient than per-token padding for large k_max.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import norm


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
    L3 layer: per-token lookup table with block-diagonal attention.

    Following arXiv:2601.21461v2 Section A.3.4:
    1. Sort tokens by ID -> groups identical tokens together
    2. Build de-duplicated embedding pool for sorted sequence
    3. Process blocks of bb sorted tokens with masked attention
       (each token attends only to its own embeddings)
    4. Unsort results back to original order
    5. Up-project, norm, concat with input, mix-project

    K/V are always tied (single kv_weight table used as both keys and values).
    Returns the delta (added residually by caller).
    """

    def __init__(self, n_embd, n_emb, d_up, vocab_size=0, k_max=32, bb=512):
        super().__init__()
        self.n_embd = n_embd
        self.n_emb = n_emb
        self.d_up = d_up
        self.k_max = k_max
        self.bb = bb

        # Single weight table for both keys and values (tied KV)
        self.kv_weight = nn.Parameter(torch.empty(n_emb, n_embd))
        self.attn_scale = 1.0 / math.sqrt(n_embd)

        # Up-project from n_embd to d_up
        self.w_up = nn.Linear(n_embd, d_up, bias=False)
        # Mix-project: concat(up_projected, x) -> n_embd
        self.w_mix = nn.Linear(d_up + n_embd, n_embd, bias=False)

        # Bounds buffer: sized for vocab so checkpoint loading works without shape mismatch
        bounds_size = (vocab_size + 1) if vocab_size > 0 else 1
        self.register_buffer("bounds", torch.zeros(bounds_size, dtype=torch.long), persistent=True)
        # emb_alloc[j] = token ID that embedding j belongs to (rebuilt by set_bounds, not saved)
        self.register_buffer("emb_alloc", torch.zeros(max(n_emb, 1), dtype=torch.long), persistent=False)

    def set_bounds(self, bounds):
        """Register bounds tensor and build derived emb_alloc mapping."""
        self.bounds = bounds
        alloc = bounds[1:] - bounds[:-1]
        self.k_max = int(alloc.max().item())
        # emb_alloc[j] = token ID for embedding j
        self.emb_alloc = torch.repeat_interleave(
            torch.arange(len(alloc), device=bounds.device, dtype=torch.long), alloc
        )

    @torch._dynamo.disable
    def forward(self, x, token_ids):
        """
        Args:
            x: [B, T, n_embd] hidden states
            token_ids: [B, T] token IDs
        Returns:
            delta: [B, T, n_embd] to be added residually by caller
        """
        B, T, C = x.shape
        N = B * T
        bb = self.bb

        # Pre-norm (same as backbone)
        q = norm(x).reshape(N, C)
        flat_ids = token_ids.reshape(-1)  # [N]

        # Sort tokens by ID — groups identical tokens together for efficient blocking
        seq_sort, fw = torch.sort(flat_ids, stable=True)
        bw = torch.empty_like(fw)
        bw[fw] = torch.arange(N, device=x.device)
        q_sorted = q[fw]  # [N, C]

        # Build de-duplicated embedding pool for sorted sequence.
        # Since tokens are sorted, unique_consecutive gives runs of identical IDs.
        unique, inverse, counts = torch.unique_consecutive(
            seq_sort, return_inverse=True, return_counts=True
        )
        unique_emb_starts = self.bounds[unique]            # [n_unique]
        unique_emb_ends = self.bounds[unique + 1]          # [n_unique]
        unique_emb_lengths = unique_emb_ends - unique_emb_starts  # [n_unique]

        # Cumulative offsets in the gathered embedding pool
        unique_offsets = torch.zeros(len(unique) + 1, dtype=torch.long, device=x.device)
        unique_offsets[1:] = torch.cumsum(unique_emb_lengths, dim=0)
        total_embs = int(unique_offsets[-1].item())

        # Build keep_cols: indices into the full embedding table.
        # Uses "batched arange" to avoid Python loops:
        # keep_cols[unique_offsets[j]:unique_offsets[j+1]] = range(bounds[unique[j]], bounds[unique[j]+1])
        flat_pos = torch.arange(total_embs, device=x.device)
        base_off = torch.repeat_interleave(unique_offsets[:-1], unique_emb_lengths)
        local_pos = flat_pos - base_off
        base_starts = torch.repeat_interleave(unique_emb_starts, unique_emb_lengths)
        keep_cols = base_starts + local_pos

        # Per sorted position: start/end range in the gathered pool
        starts = unique_offsets[inverse]       # [N]
        ends = unique_offsets[inverse + 1]     # [N]

        # Gather KV embeddings (RMSNorm'd like backbone QK-norm: q,k = norm(q), norm(k))
        # Q is already norm(x); here we norm K to match. Values use un-normed embeddings.
        emb_ids = self.emb_alloc[keep_cols]    # [total_embs] — token ID per embedding
        raw_KV = self.kv_weight[keep_cols]     # [total_embs, C]
        K = norm(raw_KV)                        # [total_embs, C] — normed keys for scoring
        V = raw_KV                              # [total_embs, C] — un-normed values for aggregation

        # Block-diagonal attention: iterate over blocks of bb sorted tokens.
        # Within each block, each token attends only to its own embeddings (via masking).
        # Because tokens are sorted, same-ID tokens are grouped → tight embedding pools.
        out_parts = []
        for block_start in range(0, N, bb):
            block_end = min(block_start + bb, N)
            block_q = q_sorted[block_start:block_end]         # [bs, C]
            block_ids = seq_sort[block_start:block_end]        # [bs]

            # Embedding pool for this block: contiguous range due to sorting
            emb_s = int(starts[block_start].item())
            emb_e = int(ends[block_end - 1].item())
            b_emb_ids = emb_ids[emb_s:emb_e]                  # [n_embs]
            bK = K[emb_s:emb_e]                                # [n_embs, C]
            bV = V[emb_s:emb_e]                                # [n_embs, C]

            # Masked attention: score with normed K, aggregate with un-normed V
            score = (block_q @ bK.T) * self.attn_scale         # [bs, n_embs]
            mask = block_ids.unsqueeze(1) == b_emb_ids.unsqueeze(0)  # [bs, n_embs]
            score = score.masked_fill(~mask, float('-inf'))
            w = F.softmax(score, dim=-1).masked_fill(~mask, 0.0)
            out_parts.append(w @ bV)                           # [bs, C]

        out = torch.cat(out_parts, dim=0)  # [N, C]

        # Unsort back to original order
        agg = out[bw].view(B, T, C)

        # Up-project, norm, concat with input, mix-project
        up = norm(self.w_up(agg))
        delta = self.w_mix(torch.cat([up, x], dim=-1))
        return delta

    @torch._dynamo.disable
    @torch.no_grad()
    def diagnostics(self, x, token_ids):
        """Like forward(), but also returns mean attention entropy."""
        B, T, C = x.shape
        N = B * T
        bb = self.bb

        q = norm(x).reshape(N, C)
        flat_ids = token_ids.reshape(-1)

        seq_sort, fw = torch.sort(flat_ids, stable=True)
        bw = torch.empty_like(fw)
        bw[fw] = torch.arange(N, device=x.device)
        q_sorted = q[fw]

        unique, inverse, counts = torch.unique_consecutive(
            seq_sort, return_inverse=True, return_counts=True
        )
        unique_emb_starts = self.bounds[unique]
        unique_emb_ends = self.bounds[unique + 1]
        unique_emb_lengths = unique_emb_ends - unique_emb_starts

        unique_offsets = torch.zeros(len(unique) + 1, dtype=torch.long, device=x.device)
        unique_offsets[1:] = torch.cumsum(unique_emb_lengths, dim=0)
        total_embs = int(unique_offsets[-1].item())

        flat_pos = torch.arange(total_embs, device=x.device)
        base_off = torch.repeat_interleave(unique_offsets[:-1], unique_emb_lengths)
        local_pos = flat_pos - base_off
        base_starts = torch.repeat_interleave(unique_emb_starts, unique_emb_lengths)
        keep_cols = base_starts + local_pos

        starts = unique_offsets[inverse]
        ends = unique_offsets[inverse + 1]

        emb_ids = self.emb_alloc[keep_cols]
        raw_KV = self.kv_weight[keep_cols]
        K = norm(raw_KV)
        V = raw_KV

        # Per-token embedding counts (in sorted order) for splitting entropy
        sorted_emb_counts = self.bounds[seq_sort + 1] - self.bounds[seq_sort]  # [N]

        out_parts = []
        all_entropies = []
        for block_start in range(0, N, bb):
            block_end = min(block_start + bb, N)
            block_q = q_sorted[block_start:block_end]
            block_ids = seq_sort[block_start:block_end]

            emb_s = int(starts[block_start].item())
            emb_e = int(ends[block_end - 1].item())
            b_emb_ids = emb_ids[emb_s:emb_e]
            bK = K[emb_s:emb_e]
            bV = V[emb_s:emb_e]

            score = (block_q @ bK.T) * self.attn_scale
            mask = block_ids.unsqueeze(1) == b_emb_ids.unsqueeze(0)
            score = score.masked_fill(~mask, float('-inf'))
            w = F.softmax(score, dim=-1).masked_fill(~mask, 0.0)

            # Entropy: -sum(w * log(w)) over valid positions per token
            log_w = w.clamp(min=1e-10).log()
            entropy = -(w * log_w).sum(dim=-1)  # [bs]
            all_entropies.append(entropy)
            out_parts.append(w @ bV)

        out = torch.cat(out_parts, dim=0)
        agg = out[bw].view(B, T, C)
        up = norm(self.w_up(agg))
        delta = self.w_mix(torch.cat([up, x], dim=-1))

        all_entropies = torch.cat(all_entropies)  # [N], sorted order
        mean_entropy = all_entropies.mean().item()

        # High-k entropy: only tokens with 5+ embeddings
        high_k_mask = sorted_emb_counts >= 5
        if high_k_mask.any():
            high_k_entropy = all_entropies[high_k_mask].mean().item()
        else:
            high_k_entropy = 0.0

        return delta, {"attn_entropy": mean_entropy, "attn_entropy_high_k": high_k_entropy}
