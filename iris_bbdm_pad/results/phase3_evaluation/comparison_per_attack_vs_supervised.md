# Table 5: Per-Attack APCER Comparison (BBDM vs Supervised)

Note: Artifact attack type is present in our test data but was not
included in the open-set supervised model evaluation.

| Attack_Type | BBDM_APCER% | DenseNet121_APCER% | DenseNet121_LastBlock_APCER% | MobileNetV3LargeModel_LastBlock_APCER% | Note |
| ----------- | ----------- | ------------------ | ---------------------------- | -------------------------------------- | ---- |
| Artifact | 0.73 | N/A | N/A | N/A | Not in open-set test set |
| CL | 32.73 | 97.07 | 95.30 | 97.59 |  |
| E-display | 35.07 | 31.99 | 25.60 | 29.99 |  |
| Fake with Add On | 13.43 | 14.93 | 7.46 | 4.48 |  |
| Generated | 8.52 | 37.77 | 51.17 | 35.17 |  |
| PostMortem | 0.00 | 20.38 | 22.02 | 32.19 |  |
| Print and E-display | 17.32 | 1.22 | 3.57 | 2.05 |  |
| Printed | 15.15 | 24.88 | 26.09 | 40.63 |  |
