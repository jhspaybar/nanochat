# Recursive/Looped Transformers Research

## Key Thesis: Recursion for Reasoning + Sparse Memory for Memorization

Two ICLR 2025 papers establish the theoretical foundation:

1. **"Mixture of Parrots"** (arXiv:2410.19034) — MoE/sparse experts disproportionately improve **memorization** over reasoning. Knowledge tasks correlate with total params; reasoning correlates with model width.

2. **"Reasoning with Latent Thoughts"** (arXiv:2502.17416) — Looped/recursive transformers disproportionately improve **reasoning**. A k-layer model looped L times nearly matches a kL-layer model on reasoning tasks.

3. **"Scaling up Test-Time Compute with Latent Reasoning"** (arXiv:2502.05171) — Explicitly states: "the recurrent-depth setup excels at learning reasoning patterns, while the MoE excels at effectively storing and retrieving complex information, and their complementarity supports the hypothesis that a future architecture would contain both modifications."

**No published paper combines recursive layers with L3-style per-token memory.** This is an unexplored direction.

## Ouro: The State of the Art for Looped LLMs

**Paper:** "Scaling Latent Reasoning via Looped Language Models" (arXiv:2510.25741), ByteDance

- **Mechanism:** N unique layers applied T times sequentially. Ouro-1.4B = 24 layers x 4 loops = 96 effective passes.
- **Result:** 1.4B params matches 4B dense on reasoning (GSM8K +6, MATH500 +23 over Qwen3-4B).
- **Key insight:** Looping does NOT increase factual capacity (~2 bits/param regardless). The advantage is entirely in reasoning depth.
- **Practical limit:** 4x loop depth is stable. 8x was unstable.
- **Adaptive early exit:** Learned per-step exit probability. Threshold q is a deployment knob for latency-accuracy tradeoff.
- **KV cache:** Naive looping needs separate caches per loop (4x memory). Reusing only last-step cache during decoding gives negligible accuracy loss.

### Stability Tricks (Critical)
- **Sandwich RMSNorm:** Both before AND after attention/FFN in each block
- **Reduced recurrence:** 8 -> 4 loops for stability
- **Conservative learning rates**
- **Progressive batch size scaling** (4M -> 8M tokens)
- **Gradient clipping** at 1.0
- **Staged training pipeline** (7.7T tokens across multiple phases)

## Other Key Papers

### Relaxed Recursive Transformers (arXiv:2410.20672, ICLR 2025)
- Converts pretrained models (e.g., Gemma 2B, 18 layers) to recursive by keeping first K layers and looping
- Adds **per-depth LoRA modules** to differentiate loop passes
- Recursive Gemma 1B outperforms similarly-sized non-recursive models
- Proposes "Continuous Depth-wise Batching" for 2-3x inference throughput

### LoopFormer (arXiv:2602.11451, Feb 2026)
- Treats iterative refinement as a trajectory, conditioning on internal time t and step size
- Uses shortcut-consistency objective (self-distillation within loop)
- At inference, users choose any budget M <= L, scales smoothly without retraining

### Mixture of LoRAs for Recursive Transformers (arXiv:2512.12880)
- Lightweight LoRA experts inside shared FFN of recursive transformer
- "ModernALBERT" with rotary embeddings, GeGLU, FlashAttention
- Expert merging at inference for deployment efficiency
- Tested at 50M-120M scale (encoder models)

### From Growing to Looping (arXiv:2602.16490, Feb 2026)
- Unifies looping (reuse layers) and depth-growing (duplicate middle layers)
- Applying inference-time looping to depth-grown models improves accuracy up to 2x on reasoning
- Works despite model never being trained to loop

## Meta's Memory Layers (arXiv:2412.09764, ICLR 2025)

- Replaces FFN layers with sparse key-value lookup memory (up to 128B memory params)
- Adds massive params without increasing FLOPs
- Improves factual accuracy >100% in some benchmarks
- Models with memory layers match dense models requiring 4x compute
- **Not combined with recursive architectures** — uses standard Llama backbone

## Proposed Architecture for nanochat

```
[wte] -> [Block 0-3] -> [L3] -> [Block 0-3] -> [L3] -> [Block 0-3] -> [lm_head]
          (shared)     (memory)   (loop 2)     (memory)   (loop 3)
```

- 4 shared layers x 3 loops = 12 effective depth
- L3 between each loop pass provides fresh per-token knowledge
- L3 attention is context-dependent (query = hidden state), so same L3 layer retrieves different info each loop as hidden state evolves
- Value embeddings preserve token identity across loops (critical for rank-collapse prevention)

### Design Decisions to Make
1. Should L3 be the same layer each pass, or separate L3 layers?
2. Should we apply L3 before the first loop, between loops, or both?
3. Do existing x0 residual connections provide enough stabilization, or need Ouro-style sandwich norm?
4. LR tuning — Ouro used conservative LRs for stability
5. How many loops? Start with 3 (matching Ouro's 4x as safe limit)

### Stability Considerations
- nanochat already has: pre-norm RMSNorm, QK norm, x0 skip connections, zero-init output projections
- May still need: sandwich norm (post-norm after sublayers), gradient clipping, conservative LR
- L3 between loops is untested — could help (fresh knowledge) or hurt (instability)
- l3_lambdas (learned gating) may be critical to prevent L3 from overpowering during loops
