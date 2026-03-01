"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import math
import torch
import torch.nn.functional as F


# =============================================================================
# Metal Flash Attention on MPS (registered as torch custom ops for torch.compile)
# =============================================================================
_mfa_forward = None
_mfa_backward = None


def enable_mps_flash():
    """Enable Metal Flash Attention for MPS. Called from training script with --mps-flash."""
    global _mfa_forward, _mfa_backward
    try:
        from metal_flash_sdpa._C import mfa_attention_forward, mfa_attention_backward
        _mfa_forward = mfa_attention_forward
        _mfa_backward = mfa_attention_backward
    except ImportError:
        raise ImportError("metal-flash-sdpa not installed. Install from: https://github.com/alliprice/metal-flash-sdpa")


# Custom ops: torch.compile sees these as opaque nodes (no graph break).
# The function bodies reference _mfa_forward/_mfa_backward globals which are
# set by enable_mps_flash() before compilation. During tracing, torch.compile
# uses the register_fake implementations for shape inference.

@torch.library.custom_op("nanochat_mfa::forward", mutates_args=())
def _mfa_fwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """MFA forward. q/k/v in (B, H, T, D), returns (output, lse)."""
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    o, lse = _mfa_forward(qt, kt, vt, scale, is_causal)
    return o.transpose(1, 2).contiguous(), lse


@_mfa_fwd_op.register_fake
def _mfa_fwd_fake(q, k, v, scale, is_causal):
    B, H, T, D = q.shape
    return q.new_empty(q.shape), q.new_empty(B * H * T, dtype=torch.float32)


@torch.library.custom_op("nanochat_mfa::backward", mutates_args=())
def _mfa_bwd_op(grad_out: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor, out: torch.Tensor, lse: torch.Tensor,
                scale: float, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MFA backward. All spatial tensors in (B, H, T, D), lse is flat (B*H*T,)."""
    gt = grad_out.transpose(1, 2).contiguous()
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    ot = out.transpose(1, 2).contiguous()
    dq, dk, dv = _mfa_backward(qt, kt, vt, ot, lse, gt, scale, is_causal)
    return dq.transpose(1, 2).contiguous(), dk.transpose(1, 2).contiguous(), dv.transpose(1, 2).contiguous()


@_mfa_bwd_op.register_fake
def _mfa_bwd_fake(grad_out, q, k, v, out, lse, scale, is_causal):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def _mfa_setup_ctx(ctx, inputs, output):
    q, k, v, scale, is_causal = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.scale = scale
    ctx.is_causal = is_causal


def _mfa_backward_fn(ctx, grad_out, _grad_lse):
    q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = _mfa_bwd_op(grad_out, q, k, v, out, lse, ctx.scale, ctx.is_causal)
    return dq, dk, dv, None, None


_mfa_fwd_op.register_autograd(_mfa_backward_fn, setup_context=_mfa_setup_ctx)


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _use_fa3():
    """Determine whether to use FA3 based on availability and override."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    return HAS_FA3  # auto


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        # Use Metal Flash Attention on MPS when available (custom op — no graph break)
        if _mfa_forward is not None and q.device.type == 'mps' and not enable_gqa and Tq >= 256:
            scale = q.size(-1) ** -0.5
            out, _lse = _mfa_fwd_op(q, k, v, scale, True)
            return out
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if _use_fa3():
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
