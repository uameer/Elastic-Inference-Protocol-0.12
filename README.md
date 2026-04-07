# Elastic Inference Protocol (EIP-0.12) — Empirical Audit

> Can we identify *when* a transformer "knows" the answer — and skip the rest?

## What This Is

An empirical investigation into **layer-wise token entropy** in transformer 
models. We probe hidden states at intermediate layers to find the point of 
"Semantic Settlement" — where a token's representation stabilizes and further 
computation adds no signal.

This repo contains the measurement tooling, real experimental results, and 
the hypothesis that this settlement point moves earlier in larger models.

## Key Finding (TinyLlama 1.1B)

![Entropy Trajectories](eip_entropy_audit.png)

- **Layers 1–16:** All tokens plateau at 10–12 bits entropy. No early settlement.
- **Layers 16–22:** Sharp phase transition. Most tokens collapse rapidly.
- **`x` token:** Approaches near-zero entropy by final layers.
- **The 0.12 gate is not crossed** at 1.1B scale.

### What this means

Early exit is not viable at 1.1B — semantic settlement is compressed into 
the final 6 layers rather than distributed across depth. This is the 
**baseline finding** this repo exists to test against larger models.

## The Hypothesis

| Model Scale | Predicted Settlement Layer | Status |
|-------------|---------------------------|--------|
| 1.1B (TinyLlama) | L16-22 (terminal) | ✅ Confirmed |
| 7B (Mistral) | L12-18 (mid-depth?) | 🔬 To be tested |
| 70B (Llama-3) | L8-16 (early?) | 🔬 To be tested |

If settlement moves earlier as parameters scale, the EIP gate becomes 
viable — and this repo will have the empirical evidence to prove it.

## Run It Yourself
```bash
pip install torch transformers matplotlib
python eip_audit.py
```

Runs on CPU. No GPU required for 1.1B.

## What's Next

- [ ] Mistral-7B layer entropy audit (needs A100)
- [ ] Cross-model settlement layer comparison chart  
- [ ] Threshold calibration across token categories
- [ ] Formal write-up

## Honest Caveats

- Results are on one model and one prompt so far
- The 0.12 threshold is a hypothesis, not yet an empirically derived constant
- Hidden state entropy ≠ output certainty — methodology notes in `NOTES.md`

## Contributing

If you have A100 access and want to run the 7B audit, open an issue.
