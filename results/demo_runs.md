| Demo run | Model | Records | Flagged as attack | Detection rate (of real attacks) | False-alarm rate (of normal) | Mean confidence |
|---|---|---|---|---|---|---|
| 1. Mixed traffic (1,000 random records) | XGBoost | 1,000 | 421 (42.1%) | 70.1% | 3.8% | 0.98 |
| 1. Mixed traffic (1,000 random records) | MLP (PyTorch) | 1,000 | 419 (41.9%) | 69.4% | 4.3% | 0.98 |
| 2. Known attack types only (500) | XGBoost | 500 | 391 (78.2%) | 78.2% | - | 0.99 |
| 2. Known attack types only (500) | MLP (PyTorch) | 500 | 407 (81.4%) | 81.4% | - | 0.97 |
| 3. Novel attack types only (500) | XGBoost | 500 | 209 (41.8%) | 41.8% | - | 0.92 |
| 3. Novel attack types only (500) | MLP (PyTorch) | 500 | 203 (40.6%) | 40.6% | - | 0.97 |
