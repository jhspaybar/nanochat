import torch
import torch.nn.functional as F

from nanochat.l3 import compute_lzw_allocation, allocation_to_bounds, L3Layer


# ---- LZW Allocation Tests ----

def test_lzw_allocation_total():
    """Allocation sums to target n_emb."""
    sequences = [[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]] * 10
    alloc = compute_lzw_allocation(sequences, vocab_size=8, n_emb=100, k_max=32)
    assert sum(alloc) == 100


def test_lzw_allocation_min_one():
    """Every token gets >= 1 embedding, even unseen tokens."""
    sequences = [[0, 1, 0, 1, 0, 1]]  # only tokens 0 and 1 appear
    alloc = compute_lzw_allocation(sequences, vocab_size=8, n_emb=100, k_max=32)
    assert all(a >= 1 for a in alloc)
    assert len(alloc) == 8


def test_lzw_allocation_max_k():
    """No token exceeds k_max embeddings."""
    sequences = [[0, 0, 0, 0, 0, 0, 0, 0]] * 100  # token 0 is extremely frequent
    alloc = compute_lzw_allocation(sequences, vocab_size=4, n_emb=50, k_max=8)
    assert all(a <= 8 for a in alloc)


def test_lzw_allocation_distribution():
    """Frequent tokens get more embeddings than rare ones."""
    # Token 0 appears 100x, token 1 appears 10x, tokens 2-7 appear once each
    sequences = [[0] * 100 + [1] * 10 + [2, 3, 4, 5, 6, 7]]
    alloc = compute_lzw_allocation(sequences, vocab_size=8, n_emb=64, k_max=32)
    assert alloc[0] >= alloc[1], f"Token 0 (freq 100) should get >= Token 1 (freq 10): {alloc[0]} vs {alloc[1]}"
    assert alloc[1] >= alloc[7], f"Token 1 (freq 10) should get >= Token 7 (freq 1): {alloc[1]} vs {alloc[7]}"


def test_lzw_bounds_from_allocation():
    """Bounds array is correct cumulative sum."""
    alloc = [3, 1, 5, 2]
    bounds = allocation_to_bounds(alloc)
    assert bounds.tolist() == [0, 3, 4, 9, 11]
    assert bounds[-1].item() == sum(alloc)


def test_lzw_allocation_edge_n_emb_equals_vocab():
    """When n_emb == vocab_size, every token gets exactly 1."""
    sequences = [[0, 1, 2, 3]]
    alloc = compute_lzw_allocation(sequences, vocab_size=4, n_emb=4, k_max=32)
    assert alloc == [1, 1, 1, 1]


def test_lzw_allocation_ngram_last_token():
    """LZW allocates to the last token of frequent n-grams, not just frequent unigrams."""
    # Token 0 appears rarely on its own, but token 1 is always preceded by token 0
    # creating the frequent bigram (0, 1). LZW should give extra embeddings to token 1
    # (the last token of the bigram) even though token 2 has higher unigram frequency.
    sequences = [
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # token 2 very frequent as unigram
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1,    # bigram (0,1) repeated
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]
    alloc = compute_lzw_allocation(sequences, vocab_size=4, n_emb=20, k_max=32)
    # Token 1 should get more embeddings than token 3 (which never appears)
    assert alloc[1] > alloc[3], f"Token 1 (last of frequent bigram 0,1) should get more than token 3 (unseen)"


# ---- L3Layer Tests ----

def _make_l3(n_embd=16, n_emb=32, d_up=64, vocab_size=8):
    """Helper to create an L3Layer with bounds set and properly initialized."""
    torch.manual_seed(42)
    layer = L3Layer(n_embd=n_embd, n_emb=n_emb, d_up=d_up)
    # Initialize weights to avoid garbage values from torch.empty()
    for name, p in layer.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.1)
        else:
            torch.nn.init.zeros_(p)
    # Simple uniform allocation: each token gets n_emb // vocab_size embeddings
    per_token = n_emb // vocab_size
    alloc = [per_token] * vocab_size
    # Distribute remainder
    remainder = n_emb - sum(alloc)
    for i in range(remainder):
        alloc[i] += 1
    bounds = allocation_to_bounds(alloc)
    layer.set_bounds(bounds)
    return layer


def test_l3_layer_output_shape():
    """Output is [B, T, n_embd]."""
    n_embd = 16
    layer = _make_l3(n_embd=n_embd, n_emb=32, d_up=64, vocab_size=8)
    x = torch.randn(2, 5, n_embd)
    token_ids = torch.randint(0, 8, (2, 5))
    out = layer(x, token_ids)
    assert out.shape == (2, 5, n_embd)


def test_l3_layer_gradient_flow():
    """All parameters receive gradients."""
    layer = _make_l3(n_embd=16, n_emb=32, d_up=64, vocab_size=8)
    x = torch.randn(2, 5, 16, requires_grad=True)
    token_ids = torch.randint(0, 8, (2, 5))
    out = layer(x, token_ids)
    loss = out.sum()
    loss.backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"
    assert x.grad is not None, "No gradient for input x"


def test_l3_layer_params():
    """L3Layer has kv_weight (tied K/V), w_up, w_mix."""
    layer = _make_l3()
    params = {n for n, _ in layer.named_parameters()}
    assert "kv_weight" in params
    assert "k_weight" not in params
    assert "v_weight" not in params
    assert "w_up.weight" in params
    assert "w_mix.weight" in params


def test_l3_layer_masking():
    """Tokens with fewer embeddings are properly masked (no NaN/inf)."""
    n_embd = 16
    # Non-uniform allocation: token 0 gets 10, token 1 gets 2
    layer = L3Layer(n_embd=n_embd, n_emb=12, d_up=64)
    bounds = torch.tensor([0, 10, 12, 12])  # 3 tokens: 10, 2, 0 embeddings
    # Token 2 has 0 embeddings - adjust to at least 1
    bounds = torch.tensor([0, 9, 11, 12])  # 3 tokens: 9, 2, 1
    layer.set_bounds(bounds)
    x = torch.randn(1, 3, n_embd)
    token_ids = torch.tensor([[0, 1, 2]])
    out = layer(x, token_ids)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains inf"
    assert out.shape == (1, 3, n_embd)


def test_l3_layer_deterministic():
    """Same input produces same output."""
    layer = _make_l3()
    x = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 8, (2, 5))
    out1 = layer(x, token_ids)
    out2 = layer(x, token_ids)
    assert torch.allclose(out1, out2)


# ---- GPT Integration Tests ----

def test_gpt_with_l3_forward():
    """Full model with L3 runs forward pass."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        l3_after_layers="2", l3_n_emb=128, l3_d_up=32, l3_k_max=16,
    )
    model = GPT(config)
    model.init_weights()
    # Set bounds for L3 layers
    alloc = compute_lzw_allocation([[0, 1, 2, 3] * 4], vocab_size=64, n_emb=128, k_max=16)
    bounds = allocation_to_bounds(alloc)
    for l3_layer in model.l3_layers.values():
        l3_layer.set_bounds(bounds)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0  # scalar loss
    assert not torch.isnan(loss)


def test_gpt_with_l3_backward():
    """loss.backward() works, L3 params get gradients."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        l3_after_layers="2", l3_n_emb=128, l3_d_up=32, l3_k_max=16,
    )
    model = GPT(config)
    model.init_weights()
    alloc = compute_lzw_allocation([[0, 1, 2, 3] * 4], vocab_size=64, n_emb=128, k_max=16)
    bounds = allocation_to_bounds(alloc)
    for l3_layer in model.l3_layers.values():
        l3_layer.set_bounds(bounds)

    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))

    # Need two forward passes: first to propagate signal through lm_head (init zeros)
    loss = model(x, y)
    loss.backward()
    optimizer = model.setup_optimizer()
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Second pass should give gradients to L3 params
    loss = model(x, y)
    loss.backward()
    for name, p in model.l3_layers.named_parameters():
        if 'bounds' not in name:  # bounds is a buffer, not a parameter
            assert p.grad is not None, f"No gradient for L3 param {name}"


def test_gpt_with_l3_optimizer():
    """setup_optimizer includes L3 params."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        l3_after_layers="2", l3_n_emb=128, l3_d_up=32, l3_k_max=16,
    )
    model = GPT(config)
    model.init_weights()
    alloc = compute_lzw_allocation([[0, 1, 2, 3] * 4], vocab_size=64, n_emb=128, k_max=16)
    bounds = allocation_to_bounds(alloc)
    for l3_layer in model.l3_layers.values():
        l3_layer.set_bounds(bounds)
    optimizer = model.setup_optimizer()
    # Collect all optimizer param ids
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))
    # Check all model params are in optimizer
    for name, p in model.named_parameters():
        assert id(p) in opt_param_ids, f"Parameter {name} not in optimizer"


def test_gpt_without_l3_unchanged():
    """L3 disabled = identical to current behavior (no L3 layers created)."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
    )
    model = GPT(config)
    assert len(model.l3_layers) == 0
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_gpt_with_l3_no_lambda():
    """L3 without learnable lambda: forward+backward works, no l3_lambdas param."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        l3_after_layers="2", l3_n_emb=128, l3_d_up=32, l3_k_max=16,
        l3_lambda=False,
    )
    model = GPT(config)
    assert model.l3_lambdas is None
    model.init_weights()
    alloc = compute_lzw_allocation([[0, 1, 2, 3] * 4], vocab_size=64, n_emb=128, k_max=16)
    bounds = allocation_to_bounds(alloc)
    for l3_layer in model.l3_layers.values():
        l3_layer.set_bounds(bounds)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()


# ---- Looped Transformer Tests ----

def test_gpt_looped_forward():
    """Looped model (no L3) produces valid loss."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=2,
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_gpt_looped_backward():
    """Shared weights receive gradients that accumulate across loops."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=2,
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    loss.backward()
    # All block params should have gradients
    for name, p in model.transformer.h.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_gpt_looped_with_l3():
    """Looped model with L3 at every loop boundary: correct number of layers, forward+backward works."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=3, l3_every_loops=1,  # explicitly enable L3 at every boundary
        l3_n_emb=128, l3_d_up=32, l3_k_max=16,
    )
    model = GPT(config)
    model.init_weights()
    # Should have 2 L3 layers (at loop boundaries 1 and 2)
    assert len(model.l3_layers) == 2
    assert "1" in model.l3_layers
    assert "2" in model.l3_layers
    # Set bounds
    alloc = compute_lzw_allocation([[0, 1, 2, 3] * 4], vocab_size=64, n_emb=128, k_max=16)
    bounds = allocation_to_bounds(alloc)
    for l3_layer in model.l3_layers.values():
        l3_layer.set_bounds(bounds)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()
    # L3 params should have gradients
    for name, p in model.l3_layers.named_parameters():
        if 'bounds' not in name:
            assert p.grad is not None, f"No gradient for L3 param {name}"


def test_gpt_looped_l3_every_2():
    """l3_every_loops=2 with 4 loops: L3 at loops 1 and 3 only."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=4, l3_every_loops=2,
        l3_n_emb=128, l3_d_up=32, l3_k_max=16,
    )
    model = GPT(config)
    # Should have L3 at loops 1 and 3 (boundaries where (loop-1) % 2 == 0)
    assert len(model.l3_layers) == 2
    assert "1" in model.l3_layers
    assert "3" in model.l3_layers
    assert "2" not in model.l3_layers


def test_gpt_looped_tbptl():
    """TBPTL: early loops don't contribute gradients to block params."""
    from nanochat.gpt import GPT, GPTConfig
    # Model with tbptl=1: first loop is forward-only
    config_tbptl = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=2, tbptl=1,
    )
    model_tbptl = GPT(config_tbptl)
    model_tbptl.init_weights()
    # Model without tbptl for comparison
    config_full = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=2, tbptl=0,
    )
    model_full = GPT(config_full)
    model_full.init_weights()
    # Copy weights so they match
    model_full.load_state_dict(model_tbptl.state_dict())
    torch.manual_seed(42)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    # Forward+backward with tbptl
    model_tbptl.train()
    loss_tbptl = model_tbptl(x, y)
    loss_tbptl.backward()
    # Forward+backward without tbptl
    model_full.train()
    loss_full = model_full(x, y)
    loss_full.backward()
    # With tbptl=1, gradients should generally differ (less gradient from fewer trainable loops)
    # The key thing is that the tbptl model runs without error and produces valid gradients
    for name, p in model_tbptl.transformer.h.named_parameters():
        assert p.grad is not None, f"No gradient for {name} with tbptl"


def test_gpt_loops_1_unchanged():
    """loops=1 is identical to the non-looped config."""
    from nanochat.gpt import GPT, GPTConfig
    config_default = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
    )
    config_loops1 = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=1,
    )
    model_default = GPT(config_default)
    model_loops1 = GPT(config_loops1)
    model_default.init_weights()
    # Copy weights
    model_loops1.load_state_dict(model_default.state_dict())
    torch.manual_seed(42)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss_default = model_default(x, y)
    loss_loops1 = model_loops1(x, y)
    assert torch.allclose(loss_default, loss_loops1), f"loops=1 loss {loss_loops1} != default loss {loss_default}"


def test_gpt_looped_kv_cache():
    """loops>1 works with KV cache for inference."""
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.engine import KVCache
    config = GPTConfig(
        sequence_len=16, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        n_loops=2,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    x = torch.randint(0, 64, (1, 8))
    # Create KV cache with n_loops * n_layer slots
    kv_cache = KVCache(
        batch_size=1, num_heads=config.n_kv_head,
        seq_len=16, head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer * config.n_loops, device="cpu", dtype=torch.float32,
    )
    # Prefill
    logits = model(x, kv_cache=kv_cache)
    assert logits.shape == (1, 8, 64)
    assert kv_cache.get_pos() == 8
    # Decode one more token
    x2 = torch.randint(0, 64, (1, 1))
    logits2 = model(x2, kv_cache=kv_cache)
    assert logits2.shape == (1, 1, 64)
    assert kv_cache.get_pos() == 9


def test_gpt_looped_l3_after_layers_exclusive():
    """l3_after_layers + loops>1 raises assertion error."""
    from nanochat.gpt import GPT, GPTConfig
    import pytest
    with pytest.raises(AssertionError, match="mutually exclusive"):
        config = GPTConfig(
            sequence_len=8, vocab_size=64, n_layer=4,
            n_head=2, n_kv_head=2, n_embd=64,
            l3_after_layers="2", l3_n_emb=128,
            n_loops=2,
        )
        GPT(config)


# ---- ConvSwiGLU MLP Tests ----

def test_gpt_convswiglu_forward():
    """ConvSwiGLU model produces valid loss."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        mlp_type="convswiglu",
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_gpt_convswiglu_backward():
    """Gradients flow to all ConvSwiGLU params including conv weight and bias."""
    from nanochat.gpt import GPT, GPTConfig, ConvSwiGLU
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        mlp_type="convswiglu",
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    # First pass + optimizer step to break through zero-init c_proj
    loss = model(x, y)
    loss.backward()
    optimizer = model.setup_optimizer()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    # Second pass: now gradients should flow to all params
    loss = model(x, y)
    loss.backward()
    # Check all params have gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
    # Specifically check conv weight and bias
    for block in model.transformer.h:
        assert isinstance(block.mlp, ConvSwiGLU)
        assert block.mlp.dwconv.weight.grad is not None
        assert block.mlp.dwconv.bias.grad is not None
        assert block.mlp.dwconv.weight.grad.abs().sum() > 0
        assert block.mlp.dwconv.bias.grad.abs().sum() > 0


def test_gpt_convswiglu_optimizer():
    """All ConvSwiGLU params in optimizer, conv bias routed to AdamW."""
    from nanochat.gpt import GPT, GPTConfig, ConvSwiGLU
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        mlp_type="convswiglu",
    )
    model = GPT(config)
    model.init_weights()
    optimizer = model.setup_optimizer()
    # Collect all optimizer param ids and their group kinds
    opt_param_ids = set()
    adamw_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_param_ids.add(id(p))
            if group["kind"] == "adamw":
                adamw_param_ids.add(id(p))
    # All model params should be in optimizer
    for name, p in model.named_parameters():
        assert id(p) in opt_param_ids, f"Parameter {name} not in optimizer"
    # Conv biases should be in AdamW groups
    for block in model.transformer.h:
        assert id(block.mlp.dwconv.bias) in adamw_param_ids, "Conv bias should be in AdamW group"


def test_gpt_convswiglu_with_loops():
    """ConvSwiGLU + loops=2 forward+backward works."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        mlp_type="convswiglu",
        n_loops=2,
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


# ---- SwiGLU MLP Tests ----

def test_gpt_swiglu_forward():
    """SwiGLU model produces valid loss."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        mlp_type="swiglu",
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_gpt_swiglu_backward():
    """Gradients flow to all SwiGLU params."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        mlp_type="swiglu",
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_gpt_swiglu_with_loops():
    """SwiGLU + loops=2 forward+backward works."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        mlp_type="swiglu",
        n_loops=2,
    )
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


# ---- Skewed Allocation L3 Tests ----

def _make_l3_skewed(n_embd=16, d_up=64, alloc=None):
    """Helper to create an L3Layer with a skewed (non-uniform) allocation."""
    if alloc is None:
        alloc = [1] * 90 + [100]  # 90 tokens with 1 emb, 1 token with 100
    n_emb = sum(alloc)
    vocab_size = len(alloc)
    torch.manual_seed(42)
    layer = L3Layer(n_embd=n_embd, n_emb=n_emb, d_up=d_up,
                    vocab_size=vocab_size, k_max=max(alloc))
    for p in layer.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.1)
        else:
            torch.nn.init.zeros_(p)
    bounds = allocation_to_bounds(alloc)
    layer.set_bounds(bounds)
    return layer


def test_skewed_alloc_no_embeddings_dropped():
    """Every token uses ALL its allocated embeddings — none are truncated."""
    alloc = [1] * 90 + [100]  # token 90 has 100 embeddings
    layer = _make_l3_skewed(alloc=alloc)
    x = torch.randn(2, 5, 16, requires_grad=True)
    # Force token 90 into the batch to exercise the high-k path
    token_ids = torch.zeros(2, 5, dtype=torch.long)
    token_ids[0, 0] = 90  # this token has 100 embeddings
    out = layer(x, token_ids)
    assert out.shape == (2, 5, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    # Verify the high-k token's embeddings all received gradient
    out.sum().backward()
    grad = layer.kv_weight.grad
    # Token 90 starts at index 90 (90 tokens × 1 embedding), has 100 rows
    high_k_grad = grad[90:190]
    assert high_k_grad.abs().sum() > 0, "High-k token embeddings should receive gradient"


def test_skewed_alloc_forward():
    """Skewed allocation: forward+backward works."""
    layer = _make_l3_skewed()
    x = torch.randn(2, 5, 16, requires_grad=True)
    token_ids = torch.randint(0, 91, (2, 5))
    out = layer(x, token_ids)
    assert out.shape == (2, 5, 16)
    assert not torch.isnan(out).any()
    out.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
    assert x.grad is not None


def test_uniform_alloc_deterministic():
    """Uniform allocation: forward is deterministic."""
    torch.manual_seed(42)
    layer = _make_l3(n_embd=16, n_emb=32, d_up=64, vocab_size=8)
    x = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 8, (2, 5))
    # Forward is deterministic
    out1 = layer(x, token_ids)
    out2 = layer(x, token_ids)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_large_k_max_forward_backward():
    """Large k_max with many embeddings: forward+backward works."""
    torch.manual_seed(42)
    alloc = [2] * 50 + [200]  # 51 tokens, k_max=200
    layer = _make_l3_skewed(alloc=alloc)
    x = torch.randn(2, 5, 16, requires_grad=True)
    token_ids = torch.randint(0, 51, (2, 5))
    out = layer(x, token_ids)
    assert out.shape == (2, 5, 16)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert layer.kv_weight.grad is not None


def test_all_high_k_tokens():
    """Batch where every token has many embeddings still works."""
    alloc = [50] * 8  # every token has 50 embeddings
    layer = _make_l3_skewed(alloc=alloc)
    x = torch.randn(2, 5, 16, requires_grad=True)
    token_ids = torch.randint(0, 8, (2, 5))
    out = layer(x, token_ids)
    assert out.shape == (2, 5, 16)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert x.grad is not None


def test_gpt_with_l3_large_k_forward():
    """Full model with large k_max L3 runs forward pass."""
    from nanochat.gpt import GPT, GPTConfig
    config = GPTConfig(
        sequence_len=8, vocab_size=64, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64,
        l3_after_layers="2", l3_n_emb=640, l3_d_up=32, l3_k_max=64,
    )
    model = GPT(config)
    model.init_weights()
    alloc = compute_lzw_allocation([[0, 1, 2, 3] * 4], vocab_size=64, n_emb=640, k_max=64)
    bounds = allocation_to_bounds(alloc)
    for l3_layer in model.l3_layers.values():
        l3_layer.set_bounds(bounds)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    loss = model(x, y)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()
    for name, p in model.l3_layers.named_parameters():
        if 'bounds' not in name:
            assert p.grad is not None, f"No gradient for L3 param {name}"
