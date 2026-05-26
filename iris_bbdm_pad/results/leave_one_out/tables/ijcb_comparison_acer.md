## Per-Attack ACER (%) — Leave-One-Out Evaluation

> Supervised models: trained on 7/8 attack types, tested on held-out type.  
> **BBDM (ours)**: trained on **0 attack types** (bona fide only).  
> Lower ACER = better detection.

| Model | Art. | CL | E-disp. | Fake+ | Gen. | P-M. | P+E | Print | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNetV2SModel | — | — | — | 3.2 | — | — | — | — | 3.2 |
| MobileNetV3LargeModel | — | — | — | 5.4 | — | — | — | — | 5.4 |
| MobileNetV2Model | — | — | — | 7.6 | — | — | — | — | 7.6 |
| DenseNet121 | — | 48.9 | 17.0 | 8.2 | 20.0 | 10.9 | 1.5 | 13.5 | 17.1 |
| DenseNet121_LastBlock | — | 48.2 | 15.4 | 6.2 | 27.3 | 13.4 | 3.6 | 14.5 | 18.4 |
| MobileNetV2Model_LastBlock | — | 47.3 | 19.6 | 11.1 | 17.2 | 15.8 | 5.2 | 23.7 | 20.0 |
| MobileNetV3LargeModel_LastBlock | — | 49.2 | 18.0 | 6.1 | 20.4 | 19.1 | 4.6 | 23.2 | 20.1 |
| EfficientNetV2SModel_LastBlock | — | 49.8 | 22.0 | 8.3 | 26.2 | 17.0 | 5.8 | 22.7 | 21.7 |
| SENetModel_LastBlock | — | 46.5 | 19.1 | 5.6 | 23.9 | 26.8 | 9.4 | 22.7 | 22.0 |
| **BBDM_LBBDM-f4** | **19.2** | **34.3** | **35.9** | **23.2** | **25.0** | **6.6** | **29.1** | **28.6** | **25.2** |