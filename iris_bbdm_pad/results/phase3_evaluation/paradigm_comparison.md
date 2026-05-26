# Table 3: Paradigm Comparison

| Approach | Requires Attack Data | Open-Set Capable | ACER Range | Notes |
| -------- | -------------------- | ---------------- | ---------- | ----- |
| Closed-set Supervised CNN | Yes (all types) | No | 0.2% – 2.3% | Requires attack samples for all test types |
| Open-set Supervised CNN | Yes (partial) | Partial | 1.6% – 50.3% | Degrades on unseen attack types |
| **BBDM Anomaly Detection (Ours)** | None | Yes | 26.5% | First BBDM-based iris PAD; truly zero-shot on attacks |
