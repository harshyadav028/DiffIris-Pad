## Per-Attack APCER (%) — Leave-One-Out Evaluation

> Supervised models: trained on 7/8 attack types, tested on held-out type.  
> **BBDM (ours)**: trained on **0 attack types** (bona fide only).  
> Lower APCER = better detection.

| Model | Art. | CL | E-disp. | Fake+ | Gen. | P-M. | P+E | Print | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNetV2SModel | — | — | — | 3.0 | — | — | — | — | 3.0 |
| MobileNetV3LargeModel | — | — | — | 6.0 | — | — | — | — | 6.0 |
| MobileNetV2Model | — | — | — | 10.4 | — | — | — | — | 10.4 |
| DenseNet121 | — | 97.1 | 32.0 | 14.9 | 37.8 | 20.4 | 1.2 | 24.9 | 32.6 |
| DenseNet121_LastBlock | — | 95.3 | 25.6 | 7.5 | 51.2 | 22.0 | 3.6 | 26.1 | 33.0 |
| MobileNetV2Model_LastBlock | — | 92.7 | 32.6 | 16.4 | 28.8 | 24.9 | 4.5 | 41.6 | 34.5 |
| MobileNetV3LargeModel_LastBlock | — | 97.6 | 30.0 | 4.5 | 35.2 | 32.2 | 2.0 | 40.6 | 34.6 |
| SENetModel_LastBlock | — | 91.3 | 29.0 | 3.0 | 36.9 | 43.5 | 11.4 | 37.2 | 36.0 |
| EfficientNetV2SModel_LastBlock | — | 98.3 | 35.6 | 7.5 | 44.8 | 27.2 | 5.3 | 39.7 | 36.9 |
| **BBDM_LBBDM-f4** | **4.4** | **13.1** | **18.2** | **28.4** | **6.4** | **3.9** | **10.5** | **16.8** | **12.7** |