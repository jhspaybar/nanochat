"""
Performance benchmarks for L3 layer forward pass.

Run with: python -m pytest tests/test_l3_perf.py -v -s
"""

import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.l3 import compute_lzw_allocation, allocation_to_bounds, L3Layer


def _make_realistic_l3(n_embd=384, vocab_size=32768, n_emb=None, d_up=None, k_max=512, device="cpu"):
    """Create an L3Layer with realistic LZW allocation."""
    if n_emb is None:
        n_emb = max(vocab_size, vocab_size * 3)
    if d_up is None:
        d_up = 4 * n_embd

    torch.manual_seed(42)
    layer = L3Layer(n_embd=n_embd, n_emb=n_emb, d_up=d_up)

    for name, p in layer.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)

    sequences = []
    rng = torch.Generator().manual_seed(42)
    for _ in range(10):
        probs = torch.zeros(vocab_size)
        for i in range(vocab_size):
            probs[i] = 1.0 / (i + 1)
        probs = probs / probs.sum()
        seq = torch.multinomial(probs, 2048, replacement=True, generator=rng).tolist()
        sequences.append(seq)

    alloc = compute_lzw_allocation(sequences, vocab_size, n_emb, k_max)
    bounds = allocation_to_bounds(alloc)
    layer.set_bounds(bounds)
    layer = layer.to(device)
    return layer, alloc


def _make_batch(B, T, vocab_size, alloc, device="cpu"):
    probs = torch.tensor([float(a) for a in alloc])
    probs = probs / probs.sum()
    token_ids = torch.multinomial(probs, B * T, replacement=True).reshape(B, T)
    x = torch.randn(B, T, 384, device=device)
    token_ids = token_ids.to(device)
    return x, token_ids


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _time_fn(fn, warmup=5, repeat=20, device="cpu"):
    for _ in range(warmup):
        fn()
        _sync(device)
    times = []
    for _ in range(repeat):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def test_l3_perf_kmax_sweep():
    """Test how k_max affects performance. Lower k_max = less padding = faster."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"k_max Sweep (device={device})")
    print(f"{'='*60}")

    n_embd = 384
    d_up = 4 * n_embd
    B, T = 32, 512
    N = B * T

    for k_max in [16, 32, 64, 128, 512]:
        layer, alloc = _make_realistic_l3(n_embd=n_embd, d_up=d_up, k_max=k_max, device=device)
        x, token_ids = _make_batch(B, T, 32768, alloc, device=device)

        # Stats
        flat_ids = token_ids.reshape(-1)
        lengths = layer.bounds[flat_ids + 1] - layer.bounds[flat_ids]
        max_chunk = max(1, (2**30) // max(k_max * n_embd, 1))
        num_chunks = (N + max_chunk - 1) // max_chunk
        tensor_elems = min(N, max_chunk) * k_max * n_embd

        fwd_times = _time_fn(lambda: layer(x, token_ids), device=device)
        fwd_ms = sum(fwd_times) / len(fwd_times) * 1000
        fwd_min = min(fwd_times) * 1000

        print(f"  k_max={k_max:>3}: fwd={fwd_ms:.1f}ms (min {fwd_min:.1f}ms), "
              f"chunks={num_chunks}, chunk_elems={tensor_elems/1e6:.0f}M, "
              f"avg_d_t={lengths.float().mean():.1f}")


def test_l3_perf_in_compiled_model():
    """Test L3 inside a compiled wrapper to simulate real training."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"L3 Inside Compiled Model (device={device})")
    print(f"{'='*60}")

    n_embd = 384
    d_up = 4 * n_embd

    # Simple wrapper that mimics a transformer block + L3
    class SimpleModel(nn.Module):
        def __init__(self, l3_layer):
            super().__init__()
            self.linear1 = nn.Linear(n_embd, n_embd, bias=False)
            self.linear2 = nn.Linear(n_embd, n_embd, bias=False)
            self.l3 = l3_layer

        def forward(self, x, token_ids):
            x = self.linear1(x)
            x = F.rms_norm(x, (n_embd,))
            x = self.linear2(x)
            x = x + self.l3(x, token_ids)
            return x

    B, T = 32, 512
    N = B * T

    for k_max in [32, 64, 512]:
        layer, alloc = _make_realistic_l3(n_embd=n_embd, d_up=d_up, k_max=k_max, device=device)
        x, token_ids = _make_batch(B, T, 32768, alloc, device=device)

        model = SimpleModel(layer).to(device)
        torch.nn.init.normal_(model.linear1.weight, std=0.02)
        torch.nn.init.normal_(model.linear2.weight, std=0.02)

        # Uncompiled baseline
        def fwd_bwd_eager():
            x_in = x.clone().requires_grad_(True)
            out = model(x_in, token_ids)
            out.sum().backward()

        eager_times = _time_fn(fwd_bwd_eager, device=device)
        eager_ms = sum(eager_times) / len(eager_times) * 1000

        # Compiled
        compiled_model = torch.compile(model)

        def fwd_bwd_compiled():
            x_in = x.clone().requires_grad_(True)
            out = compiled_model(x_in, token_ids)
            out.sum().backward()

        compiled_times = _time_fn(fwd_bwd_compiled, warmup=8, device=device)
        compiled_ms = sum(compiled_times) / len(compiled_times) * 1000

        speedup = (eager_ms / compiled_ms - 1) * 100
        print(f"  k_max={k_max:>3}: eager={eager_ms:.1f}ms, compiled={compiled_ms:.1f}ms "
              f"({speedup:+.0f}% from compile)")

        del model, compiled_model


def test_l3_perf_component_breakdown():
    """Profile individual components of the L3 forward pass."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Component Breakdown k_max=512 (device={device})")
    print(f"{'='*60}")

    n_embd = 384
    d_up = 4 * n_embd
    layer, alloc = _make_realistic_l3(n_embd=n_embd, d_up=d_up, k_max=512, device=device)

    B, T = 32, 512
    N = B * T
    x, token_ids = _make_batch(B, T, 32768, alloc, device=device)

    flat_ids = token_ids.reshape(-1)
    k_max = layer.k_max

    x_norm = F.rms_norm(x, (n_embd,))
    x_flat = x_norm.reshape(-1, n_embd)

    # Attention via per-k tiered approach (calls _attend which groups by k value)
    t = _time_fn(lambda: layer._attend(x_flat, flat_ids, device), device=device)
    print(f"\n  Attend tiered ({N} tokens, {len(layer._unique_k)} k-groups): {sum(t)/len(t)*1000:.1f}ms")

    # Dense
    agg = torch.randn(B, T, n_embd, device=device)
    t = _time_fn(lambda: (layer.w_up(agg), layer.w_mix(torch.cat([F.rms_norm(layer.w_up(agg), (d_up,)), x], dim=-1))), device=device)
    print(f"  Dense (w_up+mix): {sum(t)/len(t)*1000:.1f}ms")

    # Full forward
    t = _time_fn(lambda: layer(x, token_ids), device=device)
    print(f"  TOTAL forward:    {sum(t)/len(t)*1000:.1f}ms ({N/(sum(t)/len(t)):.0f} tok/s)")
