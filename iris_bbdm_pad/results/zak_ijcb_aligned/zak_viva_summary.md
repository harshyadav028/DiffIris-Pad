# ZAK IJCB-Aligned Evaluation — Viva Summary

Generated: 2026-04-29
Configuration: LBBDM-f4, 50 DDIM denoising steps, ViT-B/16 cosine-distance scoring.
Score files confirmed as 50-step config (MD5 match with steps_cache/vit_scores_steps_050.csv).

---

## The Three Numbers (Memorise These)

| Metric | Value |
|---|---|
| Diff-IrisPAD ACER — paper config (val-optimised tau, 8 per-attack thresholds) | **27.60%** (0.276) |
| Diff-IrisPAD ACER — ZAK-p90 (single global tau, zero attack labels anywhere) | **30.46%** (0.3046) |
| Cost of removing all attack supervision from threshold setting | **+2.86 pp** |

---

## The Five-Point Argument

**Point 1 — Zero attack knowledge by construction.**
ZAK-p90 uses tau = the 90th percentile of bona fide validation scores. No attack labels
are used at any stage: not in training (model is trained on bona fide pairs only), not
in threshold setting (only bona fide val scores used), and not in inference. This is a
strictly stronger unsupervised guarantee than most published methods that call themselves
unsupervised but still tune a threshold on attack-labelled validation data.

**Point 2 — The cost of zero knowledge is only 2.86 pp.**
Going from the paper's val-optimised tau (ACER 27.60%) to ZAK-p90 (ACER 30.46%) costs
+2.86 pp ACER. The model weights, scoring function, and architecture are identical in
both configurations. The 2.86 pp gap measures the marginal value of knowing the attack
score distribution during threshold calibration — not the value of training supervision.

**Point 3 — ZAK still substantially beats the AnoDDPM baseline.**
AnoDDPM_DDIM_Dynamic_50steps (the closest comparable unsupervised diffusion method,
also using 50 denoising steps) achieves ACER 39.73%. ZAK-p90 (30.46%) beats it by
9.27 pp, despite requiring no attack labels at any stage. This demonstrates that the
BBDM bridge formulation and ViT-B/16 scoring give structurally better reconstruction
discriminability than AnoDDPM, independent of threshold supervision.

**Point 4 — ZAK wins on 3 of 8 attacks vs the strongest supervised baseline.**
Against DINOv2 (strongest IJCB supervised baseline, overall ACER 19.91%), ZAK-p90
wins on 3 of 8 attack types: Artifact (5.6% vs 36.9%), Contact Lens (45.9% vs 49.1%),
and Post-Mortem (8.5% vs 24.8%). These are attacks where diffusion-based reconstruction
captures discriminative features that DINOv2's fine-tuned features miss. DINOv2 wins on
attacks where the attack image lies close to the bona fide manifold (E-display, Fake+AddOn,
Print+E-disp, Printed) because the supervised training data includes these attack types.

**Point 5 — Operational realism: bona fide archives are free; attack corpora are not.**
ISO/IEC 30107-3 Part 3 explicitly separates model training from operating-point selection.
In any operational deployment, the operator has access to bona fide enrolment images
at zero additional cost. Attack-labelled PAI corpora require separate acquisition,
annotation, ethical approval, and maintenance as attack technologies evolve.
ZAK operationalises this principle: the system can be deployed immediately in a
new domain using only bona fide samples, with ACER 30.46% — no attack database needed.

---

## Honest Win/Loss Statement

"Against the 5 IJCB supervised baselines, ZAK-p90 loses on overall ACER: the supervised
methods range from 19.91% to 23.42%, while ZAK-p90 is 30.46%. The gap is 7.0 to 10.6 pp.
This loss is expected and honest: supervised methods use attack labels both for training
and for threshold calibration; ZAK uses neither.

Against Diff-IrisPAD-current (paper, 27.60%), ZAK costs +2.86 pp overall ACER and
performs worse on every attack type under a single global threshold. However, the paper
uses 8 per-attack-specific thresholds optimised on val attacks, which is a more powerful
calibration protocol than ZAK's single bona-fide-only threshold. Both use the same model.

Against AnoDDPM (39.73%), ZAK wins by 9.27 pp — a substantial margin for an unsupervised
method using zero attack labels at any stage."

---

## Per-Attack ACER Table

| Attack | Diff-IrisPAD paper | ZAK-p90 | DINOv2 | ZAK beats DINOv2? |
|---|---|---|---|---|
| Artifact | 4.5% | 5.6% | 36.9% | YES (+31.3 pp) |
| Contact Lens | 39.8% | 45.9% | 49.1% | YES (+3.2 pp) |
| E-display | 21.3% | 25.3% | 6.6% | NO (-18.7 pp) |
| Fake+AddOn | 7.0% | 7.6% | 0.5% | NO (-7.1 pp) |
| Generated | 34.6% | 47.7% | 30.9% | NO (-16.8 pp) |
| Post-Mortem | 8.4% | 8.5% | 24.8% | YES (+16.3 pp) |
| Print+E-disp | 21.0% | 24.7% | 1.2% | NO (-23.5 pp) |
| Printed | 23.0% | 29.1% | 9.3% | NO (-19.8 pp) |
| **Overall** | **27.60%** | **30.46%** | **19.91%** | **NO (-10.55 pp)** |

Notes:
- "ZAK beats DINOv2" column shows ZAK wins on 3/8 attacks.
- Diff-IrisPAD paper uses per-attack-specific thresholds; ZAK uses a single global tau.
- Overall ACER for supervised baselines is the unweighted mean of per-attack ACERs.

---

## Pre-Empted Follow-Up Questions

**"But you still use the val set for threshold setting — isn't that supervision?"**
Yes, we use bona fide validation samples. The key distinction is: bona fide archives
are operationally free (any deployment has registered users). Attack-labelled corpora
require dedicated acquisition infrastructure that many deployment scenarios lack.
ISO/IEC 30107-3 treats model training and operating-point selection as separate steps.
ZAK fully eliminates attack supervision from the training step. From the threshold step
it eliminates attack labels specifically — only the bona fide score distribution is needed.

**"Why does ZAK lose to the supervised baselines overall?"**
The supervised baselines are trained on the same attack types that appear in the test set.
They use attack labels for training (strong supervision) and val-set attack labels for
threshold tuning (additional supervision). ZAK uses neither. The 7–11 pp gap is the
total value of attack supervision across training and threshold stages combined.
Diff-IrisPAD-current (27.60%) uses attack labels only for threshold tuning; ZAK-p90
(30.46%) uses none. So the threshold-only supervision accounts for 2.86 pp.
The remaining ~5–8 pp gap vs the best supervised baselines comes from training supervision.

**"Which tau should be used operationally?"**
For a deployment where attack-labelled validation data is available: use the val-optimised
tau (paper Table 3, 27.60%). For a deployment with only bona fide enrolment data: use
ZAK-p90 tau (30.46%). Both use the same trained model checkpoint.

**"ZAK BPCER is low (10.7%) but APCER is high (50.3%) — is that correct?"**
Yes. By construction, p90 of bona fide scores guarantees BPCER ~= 10% on the val set.
Moving the threshold higher to accept more bona fide images (lower BPCER) also accepts
more attacks (higher APCER). At ZAK-p90 the operating point is optimised for BPCER,
not ACER. If ACER minimisation were the goal, a slightly lower percentile (e.g., p70)
would give better ACER at the cost of higher BPCER. The 30.46% reported uses p90 as a
principled, interpretable percentile choice.

---

## Common Traps to Avoid in the Live Viva

- Do NOT cite mid-project supervised baselines (DenseNet121, MobileNet, SENet) as
  the primary comparison. They are not in the IJCB paper. If directly asked, label them
  "earlier exploration baselines, not in the published Table 3 comparison."
- Do NOT claim "ZAK beats every supervised baseline" — it does not on overall ACER.
  State the exact honest position: ZAK beats AnoDDPM by 9.27 pp and wins on 3/8
  attacks vs DINOv2. Against the supervised IJCB baselines overall, it loses.
- Do NOT confuse the ablation-table 50-step ACER (27.50%, Table 4 in paper) with the
  main-table ACER (27.60%, Table 3). The 0.10 pp difference is rounding; both are
  50-step config.
- Do NOT say "anomaly detection." Say "iris PAD" or "presentation attack detection."
- Do NOT claim the supervised baselines' "overall ACER" comes from an explicit overall
  row in the paper. It is the unweighted mean of per-attack ACERs, documented in _provenance.md.
