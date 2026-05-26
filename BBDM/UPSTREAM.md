# BBDM — Upstream Library

This directory contains a vendored copy of the BBDM library used as the backbone for Diff-IrisPAD.

## Source

- **Paper**: "BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models" (Li et al., CVPR 2023)
- **Original repo**: https://github.com/xuekt98/BBDM
- **Vendored commit**: see `.git/` inside this directory

## Why Vendored

The Diff-IrisPAD training launcher (`iris_bbdm_pad/training/train_bbdm_bonafide.py`) invokes `BBDM/main.py` as a subprocess with `PYTHONPATH=BBDM`. Vendoring avoids version drift and ensures the exact config format (YAML keys, `n_steps` semantics) matches what the training scripts expect.

## Modifications

None. This is the upstream BBDM library unmodified. All Diff-IrisPAD customisations live in `iris_bbdm_pad/`.

## VQGAN Weights

`resources/vq-f4/model.ckpt` (722 MB) — the VQ-f4 encoder checkpoint from CompVis/latent-diffusion. Also available upstream at:
https://github.com/CompVis/latent-diffusion
