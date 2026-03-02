"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, norm
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn
from nanochat.l3 import L3Layer

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # L3 (Large Lookup Layers) config
    l3_after_layers: str = ""   # comma-separated layer indices (empty = disabled)
    l3_n_emb: int = 0           # total embeddings (0 = disabled)
    l3_d_up: int = 0            # up-projection dim (0 = auto: 4 * n_embd)
    l3_k_max: int = 32          # max embeddings per token
    l3_lambda: bool = False     # learnable per-L3-layer scaling (True = learnable, False = unscaled residual)
    # Looped transformer config
    n_loops: int = 1            # times to loop through blocks (1 = standard)
    l3_every_loops: int = 0     # insert L3 every N loop boundaries (0 = disabled, only when n_loops > 1)
    tbptl: int = 0              # truncated backprop: forward-only for first N loops (0 = disabled)
    mlp_type: str = "relu2"     # "relu2" (current) or "convswiglu"


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last effective layer processes
            if self.layer_idx + kv_cache.layer_offset == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 2/3 of 4x expansion for gate+up (roughly same param budget as relu2 MLP)
        inter = (4 * config.n_embd * 2 // 3 + 255) // 256 * 256  # round to 256 for tensor cores
        self.inter = inter
        self.c_gate_up = nn.Linear(config.n_embd, inter * 2, bias=False)  # fused gate + up
        self.c_proj = nn.Linear(inter, config.n_embd, bias=False)

    def forward(self, x):
        gate, up = self.c_gate_up(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.c_proj(x)
        return x


class ConvSwiGLU(nn.Module):
    def __init__(self, config, conv_kernel=2):
        super().__init__()
        # SwiGLU uses 2/3 of 4x expansion for gate+up (roughly same param budget as relu2 MLP)
        inter = (4 * config.n_embd * 2 // 3 + 255) // 256 * 256  # round to 256 for tensor cores
        self.inter = inter
        self.c_gate_up = nn.Linear(config.n_embd, inter * 2, bias=False)  # fused gate + up
        self.dwconv = nn.Conv1d(inter, inter, kernel_size=conv_kernel,
                                padding=conv_kernel // 2, groups=inter, bias=True)
        self.c_proj = nn.Linear(inter, config.n_embd, bias=False)

    def forward(self, x):
        gate, up = self.c_gate_up(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.dwconv(x.transpose(1, 2).to(self.dwconv.weight.dtype))
        x = x[..., :gate.size(1)]  # trim conv padding
        x = F.silu(x)
        x = x.transpose(1, 2).contiguous()
        x = self.c_proj(x)
        return x


_MLP_TYPES = {"relu2": MLP, "swiglu": SwiGLU, "convswiglu": ConvSwiGLU}


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = _MLP_TYPES[config.mlp_type](config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # L3 layers (placed between decoder blocks, or at loop boundaries when looping)
        l3_d_up = config.l3_d_up if config.l3_d_up > 0 else 4 * config.n_embd
        if config.n_loops > 1:
            # Looped mode: L3 at loop boundaries (mutually exclusive with l3_after_layers)
            assert not config.l3_after_layers, "l3_after_layers and n_loops>1 are mutually exclusive"
            self.l3_layer_indices = set()  # not used in looped mode
            self._l3_loop_indices = [loop for loop in range(1, config.n_loops)
                                     if config.l3_every_loops > 0 and (loop - 1) % config.l3_every_loops == 0]
            if config.l3_d_up == 0 and self._l3_loop_indices and config.l3_n_emb > 0:
                config.l3_d_up = l3_d_up
            self.l3_layers = nn.ModuleDict({
                str(loop): L3Layer(config.n_embd, config.l3_n_emb, l3_d_up,
                                   vocab_size=config.vocab_size, k_max=config.l3_k_max)
                for loop in self._l3_loop_indices
            }) if self._l3_loop_indices and config.l3_n_emb > 0 else nn.ModuleDict()
            sorted_l3 = sorted(self._l3_loop_indices) if self._l3_loop_indices else []
        else:
            # Standard mode: L3 after specific layer indices
            self.l3_layer_indices = set(int(x) for x in config.l3_after_layers.split(",") if x.strip()) if config.l3_after_layers else set()
            self._l3_loop_indices = []
            if config.l3_d_up == 0 and self.l3_layer_indices:
                config.l3_d_up = l3_d_up
            self.l3_layers = nn.ModuleDict({
                str(i): L3Layer(config.n_embd, config.l3_n_emb, l3_d_up,
                                vocab_size=config.vocab_size, k_max=config.l3_k_max)
                for i in self.l3_layer_indices
            }) if self.l3_layer_indices and config.l3_n_emb > 0 else nn.ModuleDict()
            sorted_l3 = sorted(self.l3_layer_indices) if self.l3_layer_indices else []
        # L3 per-layer learnable scaling (like resid_lambdas for the residual stream)
        self._l3_lambda_map = {i: pos for pos, i in enumerate(sorted_l3)}
        self.l3_lambdas = nn.Parameter(torch.ones(len(sorted_l3))) if (sorted_l3 and config.l3_lambda) else None
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if isinstance(block.mlp, ConvSwiGLU):
                torch.nn.init.uniform_(block.mlp.c_gate_up.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                torch.nn.init.zeros_(block.mlp.dwconv.bias)
                # dwconv.weight: leave at default init (Kaiming uniform, fine for small depthwise kernel)
            elif isinstance(block.mlp, SwiGLU):
                torch.nn.init.uniform_(block.mlp.c_gate_up.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # L3 layers: kv_weight uses Llama-style init (std=1/sqrt(n_embd)) so that
        # attention logits have O(1) variance at init, giving smooth softmax.
        # std=1.0 would give logits with std≈sqrt(n_embd)≈28, making softmax one-hot.
        l3_std = n_embd ** -0.5
        for l3_layer in self.l3_layers.values():
            torch.nn.init.normal_(l3_layer.kv_weight, mean=0.0, std=l3_std)
            torch.nn.init.uniform_(l3_layer.w_up.weight, -s, s)
            torch.nn.init.zeros_(l3_layer.w_mix.weight)  # Zero init (consistent with c_proj pattern)
        if self.l3_lambdas is not None:
            self.l3_lambdas.fill_(1.0)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)
            for l3_layer in self.l3_layers.values():
                l3_layer.kv_weight.data = l3_layer.kv_weight.data.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _l3_scale(self, key):
        """Return the learnable L3 scale for the given layer/loop key, or 1 if disabled."""
        if self.l3_lambdas is not None:
            return self.l3_lambdas[self._l3_lambda_map[key]]
        return 1

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        # L3 kv/k/v weights are embeddings (lookup tables), not matmul weights
        l3_embed_numel = 0
        l3_matrix_numel = 0
        for l3_layer in self.l3_layers.values():
            l3_embed_numel += l3_layer.kv_weight.numel()
            l3_matrix_numel += l3_layer.w_up.weight.numel() + l3_layer.w_mix.weight.numel()
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel + l3_embed_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        n_loops = self.config.n_loops
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops_per_pass = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops_per_pass += 12 * h * q * effective_seq
        # L3 FLOPs: only the attention-like compute (K_t@x and V_t@score).
        # W_up and W_mix are already counted in matmul params below.
        # Forward: 2*avg_k*n_embd (K@x) + 2*avg_k*n_embd (V@score) = 4*avg_k*n_embd
        # Fwd+bwd: 3x forward = 12*avg_k*n_embd
        l3_flops = 0
        if self.l3_layers:
            n_embd = self.config.n_embd
            avg_k = self.config.l3_n_emb / self.config.vocab_size if self.config.vocab_size > 0 else 1
            per_l3 = 12 * avg_k * n_embd
            l3_flops = int(per_l3 * len(self.l3_layers))
        # With looping, shared block params are used n_loops times per forward pass
        block_params = sum(p.numel() for p in self.transformer.h.parameters())
        other_matmul_params = (nparams - nparams_exclude) - block_params  # lm_head + l3 matrices
        num_flops_per_token = 6 * (n_loops * block_params + other_matmul_params) + n_loops * attn_flops_per_pass + l3_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + (self.l3_lambdas.numel() if self.l3_lambdas is not None else 0)
        # L3: separate embedding-like params from matrix params
        l3_embeds = 0
        l3_matrices = 0
        for l3_layer in self.l3_layers.values():
            l3_embeds += l3_layer.kv_weight.numel()
            l3_matrices += l3_layer.w_up.weight.numel() + l3_layer.w_mix.weight.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + l3_embeds + l3_matrices
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        result = {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'l3_embeds': l3_embeds,
            'l3_matrices': l3_matrices,
            'scalars': scalars,
            'total': total,
        }
        if self.config.n_loops > 1:
            result['n_loops'] = self.config.n_loops
        return result

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # Split block params by dimensionality: 2D matrices go to Muon, everything else (1D biases,
        # 3D conv weights) goes to AdamW. Muon is designed for 2D weight matrices only.
        matrix_params = [p for p in self.transformer.h.parameters() if p.dim() == 2]
        block_scalar_params = [p for p in self.transformer.h.parameters() if p.dim() != 2]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # L3 params: embedding-like (kv weights), matrix (w_up, w_mix), and scalars (l3_lambdas)
        l3_embed_params = []
        l3_matrix_params = []
        l3_scalar_params = [self.l3_lambdas] if self.l3_lambdas is not None else []
        for l3_layer in self.l3_layers.values():
            l3_embed_params.append(l3_layer.kv_weight)
            l3_matrix_params.append(l3_layer.w_up.weight)
            l3_matrix_params.append(l3_layer.w_mix.weight)
        assert len(list(self.parameters())) == len(matrix_params) + len(block_scalar_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(l3_embed_params) + len(l3_matrix_params) + len(l3_scalar_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # Block bias params (e.g. ConvSwiGLU depthwise conv bias) - empty when mlp_type="relu2"
        if block_scalar_params:
            param_groups.append(dict(kind='adamw', params=block_scalar_params, lr=scalar_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # L3 scalar params (like x0_lambdas: learned gating)
        if l3_scalar_params:
            param_groups.append(dict(kind='adamw', params=l3_scalar_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))
        # L3 param groups: kv_weight uses AdamW (like token embeddings, no weight decay)
        if l3_embed_params:
            param_groups.append(dict(kind='adamw', params=l3_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        all_matrix_params = matrix_params + l3_matrix_params
        for shape in sorted({p.shape for p in all_matrix_params}):
            group_params = [p for p in all_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        if self.config.n_loops > 1:
            # Looped forward pass: shared blocks executed n_loops times
            # TBPTL: first tbptl loops run under no_grad (no autograd graph, saves memory)
            n_layer = self.config.n_layer
            tbptl = self.config.tbptl if self.training else 0
            if tbptl > 0:
                with torch.no_grad():
                    for loop in range(tbptl):
                        l3_key = str(loop)
                        if loop > 0 and l3_key in self.l3_layers:
                            x = x + self._l3_scale(loop) * self.l3_layers[l3_key](x, idx)
                        if kv_cache is not None:
                            kv_cache.layer_offset = loop * n_layer
                        for i, block in enumerate(self.transformer.h):
                            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            # Trainable loops (tbptl onward): full gradient tracking
            for loop in range(tbptl, self.config.n_loops):
                l3_key = str(loop)
                if loop > 0 and l3_key in self.l3_layers:
                    x = x + self._l3_scale(loop) * self.l3_layers[l3_key](x, idx)
                if kv_cache is not None:
                    kv_cache.layer_offset = loop * n_layer
                for i, block in enumerate(self.transformer.h):
                    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                    ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        else:
            # Standard (non-looped) forward pass
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                # L3 layer after this block (if configured)
                if str(i) in self.l3_layers:
                    x = x + self._l3_scale(i) * self.l3_layers[str(i)](x, idx)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.no_grad()
    def l3_diagnostics(self, idx):
        """Compute L3 diagnostic metrics outside of torch.compile. Returns flat dict with l3/ prefix keys."""
        if not self.l3_layers:
            return {}
        B, T = idx.size()
        T0 = 0
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        metrics = {}
        if self.config.n_loops > 1:
            # Looped mode: L3 at loop boundaries
            for loop in range(self.config.n_loops):
                l3_key = str(loop)
                if loop > 0 and l3_key in self.l3_layers:
                    delta, diag = self.l3_layers[l3_key].diagnostics(x, idx)
                    scaled_delta = self._l3_scale(loop) * delta
                    key = f"loop{loop}"
                    metrics[f"l3/delta_ratio_{key}"] = (scaled_delta.norm() / (x.norm() + 1e-8)).item()
                    for dname, dval in diag.items():
                        metrics[f"l3/{dname}_{key}"] = dval
                    x = x + scaled_delta
                for i, block in enumerate(self.transformer.h):
                    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                    ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                    x = block(x, ve, cos_sin, self.window_sizes[i], None)
        else:
            # Standard mode: L3 after specific layers
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], None)
                if str(i) in self.l3_layers:
                    delta, diag = self.l3_layers[str(i)].diagnostics(x, idx)
                    scaled_delta = self._l3_scale(i) * delta
                    key = f"layer{i}"
                    metrics[f"l3/delta_ratio_{key}"] = (scaled_delta.norm() / (x.norm() + 1e-8)).item()
                    for dname, dval in diag.items():
                        metrics[f"l3/{dname}_{key}"] = dval
                    x = x + scaled_delta
        # Weight norms (w_up, w_mix only — kv_weight is F.normalized at forward time so raw norm is meaningless)
        for key, l3_layer in self.l3_layers.items():
            metrics[f"l3/w_up_norm_{key}"] = l3_layer.w_up.weight.norm().item()
            metrics[f"l3/w_mix_norm_{key}"] = l3_layer.w_mix.weight.norm().item()
        # Lambda values
        if self.l3_lambdas is not None:
            for key, pos in self._l3_lambda_map.items():
                metrics[f"l3/lambda_{key}"] = self.l3_lambdas[pos].item()
        return metrics

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
