# Table 1: LivDet-Iris 2025 Task 1 Comparison

Our method marked with **bold**. ACER = (APCER + BPCER) / 2.

| Team | Method | Supervision | AUROC | APCER% | BPCER% | Notes |
| ---- | ------ | ----------- | ----- | ------ | ------ | ----- |
| Dermalog-Iris | Patch-based 3-network fusion | Supervised | 0.9057 | 10.69 | 28.26 |  |
| MSU D-NetPAD | DenseNet121 | Supervised | 0.9014 | 6.44 | 40.33 |  |
| BUCEA | Attention multi-level fusion | Supervised | 0.8984 | 11.29 | 32.94 |  |
| Baseline ViT+Synth | ViT + GAN synthetic data | Supervised | 0.8648 | 29.47 | 2.62 |  |
| Baseline ViT | ViT vanilla | Supervised | 0.8450 | 28.94 | 2.45 |  |
| Baseline ResNet | ResNet101 | Supervised | 0.8242 | 30.03 | 2.24 |  |
| MSU SPAD | DenseNet121 | Supervised | 0.8043 | 25.38 | 27.62 |  |
| EF 004 | Dynamic ensemble | Supervised | 0.6827 | 16.62 | 87.06 |  |
| HDA | CLIP + LoRA | Supervised | 0.5692 | 88.29 | 4.74 |  |
| **IIT Mandi (Ours)** | BBDM Anomaly Detection | Unsupervised | N/A | 33.04 | 19.93 | Zero attack supervision |
