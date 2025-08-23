# 2025 UG Test - ASR Model

**Name:** <Your Name>  
**Date:** <YYYY-MM-DD>

## 1. Data & Preprocessing (≤ 1/2 page)
- Dataset summary (hours, speakers, sampling rate)
- Segmentation from timestamps (how produced; any QC)
- Preprocessing choices & rationale:
  - resampling to 16 kHz
  - text normalization (what rules?)
  - silence trimming (how much?)
  - filtering or concatenating short utterances (thresholds, why?)
  - augmentations (spec augment, speed perturbation)

## 2. Training Strategy (≤ 1/2 page)
- Models compared: baseline whisper-base, LoRA, decoder-only, last-2-layers, etc.
- Hyperparameters (epochs, LR, batch, grad-accum, warmup, sched, freeze encoder?)
- Learning curves (loss/wer vs. steps) – include a small figure or table
- Any stability tricks (grad checkpoint, clipping, label smoothing)

## 3. Results (≤ 1/2 page)
- Test WER for each model (table). Note domain adaptation gains.
- Qualitative error examples (child pronunciations, disfluencies)
- Brief discussion of what helped most and why.

## 4. Reproducibility
- Exact commit/seed, hardware, runtime.
- How to run: commands used for prep/train/eval.
