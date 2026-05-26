# Scoring Method Comparison

* marks best method (lowest ACER).

| Method | ACER% | APCER% | BPCER% | EER% | AUC |
| ------ | ----- | ------ | ------ | ---- | --- |
| mse_score | 44.08 | 71.71 | 16.44 | 43.81 | 0.6161 |
| lpips_score | 30.51 | 18.80 | 42.22 | 33.77 | 0.7481 |
| recon_score * | 30.41 | 15.65 | 45.17 | 33.92 | 0.7408 |
| trajectory_score | 50.76 | 90.65 | 10.88 | 53.74 | 0.4457 |
| combined_score | 52.07 | 84.96 | 19.17 | 50.46 | 0.4970 |