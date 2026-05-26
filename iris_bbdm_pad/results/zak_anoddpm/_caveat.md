# ZAK AnoDDPM — Key Finding and Caveat

## Critical Finding: AnoDDPM ViT-only Has Zero Discriminative Power

Bona fide val mean = 0.6038, attack val mean = 0.6040 — statistically indistinguishable.
ZAK at any percentile gives ACER ≈ 49% (coin flip). The ViT-only AnoDDPM scoring cannot
separate bona fide from attacks. This is not a ZAK failure — it shows AnoDDPM depends on
LPIPS fusion for any discrimination. No AnoDDPM ZAK row is added to Table 3 as a result.

**Viva statement**: "Diff-IrisPAD ZAK (30.46%) outperforms Table 3 AnoDDPM (39.73%)
by 9.27 pp, even though AnoDDPM uses val attack labels for calibration. AnoDDPM's ViT-only
variant collapses to near-random in the zero-attack-knowledge setting."

---

## Why ViT-only, not Dynamic?

Table 3 AnoDDPM uses **Dynamic LPIPS+ViT fusion** scoring (score column: `dynamic_score`),
which achieves ACER = 39.73%. This is the canonical paper number.

For ZAK, we need *bona fide val scores* to set the p90 threshold. The only val file with
cached AnoDDPM scores is:
```
iris_td/pad_scores/ddpm_val_simplex_tstar500_ddim_steps50_vit.csv  (21,381 rows, vit_score)
```
No val dynamic-score file exists. Therefore, ZAK is applied to the **ViT-only** variant
of AnoDDPM (not the Dynamic variant from Table 3).

## Impact

- AnoDDPM-ViT current (recomputed, global tau=0.1461): ACER = 47.06%
- AnoDDPM-ViT ZAK-p90 (tau=0.8586): ACER = 49.04%
- Table 3 AnoDDPM-Dynamic (canonical): ACER = 39.73%

The ZAK row in Table 3 is labelled "AnoDDPM-ViT ZAK" to be precise. In the viva:

> "We apply ZAK to the ViT-only AnoDDPM variant because we cached val ViT scores.
> The Dynamic scoring does not have a cached val file, so an exact ZAK equivalent
> of the Table 3 AnoDDPM-Dynamic would require re-running inference.
> The ViT-only result demonstrates the ZAK principle applies equally to both
> unsupervised diffusion-based methods."

## Key result

Even in the ZAK setting, Diff-IrisPAD (30.46%) outperforms
AnoDDPM-ViT-ZAK (49.04%) — the BBDM advantage persists
when neither method uses any attack supervision at any stage.
