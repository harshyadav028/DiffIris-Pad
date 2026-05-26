# Open-Set Supervised Baselines Comparison

Supervised models trained on 7/8 attack types. BBDM trained on bona fide only (0 attack types). Our method in **bold**.

| Model | Type | ACER% | APCER% | BPCER% |
| ----- | ---- | ----- | ------ | ------ |
| DenseNet121_LastBlock | Supervised Open-Set | 38.39 | 71.82 | 4.97 |
| MobileNetV3LargeModel | Supervised Open-Set | 40.83 | 80.22 | 1.44 |
| EfficientNetV2SModel_LastBlock | Supervised Open-Set | 41.33 | 75.59 | 7.07 |
| MobileNetV2Model | Supervised Open-Set | 43.02 | 83.99 | 2.05 |
| SENetModel | Supervised Open-Set | 44.06 | 78.51 | 9.60 |
| MobileNetV3LargeModel_LastBlock | Supervised Open-Set | 46.67 | 87.16 | 6.17 |
| EfficientNetV2SModel | Supervised Open-Set | 47.06 | 93.18 | 0.93 |
| MobileNetV2Model_LastBlock | Supervised Open-Set | 49.44 | 93.43 | 5.46 |
| SENetModel_LastBlock | Supervised Open-Set | 50.31 | 90.75 | 9.87 |
| **BBDM (Ours)** | **Unsupervised (BBDM)** | **26.49** | **33.04** | **19.93** |