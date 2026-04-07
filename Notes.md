# Methodology Notes

## What We Measure
Hidden state entropy at intermediate transformer layers, projected through 
the LM head at each probe point. This is NOT the same as output token 
probability — it is a proxy for representational uncertainty at each depth.

## Known Limitations
- Single model tested (TinyLlama 1.1B) 
- Single prompt used for calibration
- 0.12-bit threshold is a hypothesis, not empirically derived at this scale
- CPU inference only — no quantization artifacts

## What Would Validate the Hypothesis
1. Run same audit on Mistral-7B and Llama-3-8B
2. Show settlement layer moves earlier as parameters scale
3. Test across 500+ diverse prompts, not one

## What the Data Does NOT Show
- That EIP early-exit would preserve reasoning quality
- That 82% of tokens would exit early in production workloads
- Any benchmark performance numbers
