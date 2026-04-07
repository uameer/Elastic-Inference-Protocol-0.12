# Empirical Audit: TinyLlama 1.1B Layer-wise Entropy

**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0  
**Date:** April 2026  
**Hardware:** CPU (Google Colab)  
**Status:** Real experimental results

## Finding
Phase transition observed at layers 16-22. No token crosses 
the 0.12-bit threshold at this scale. Full results in /results/.
