### Binary (normal vs attack) - KDDTest+, mean ± std over seeds

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | ms / 1k rec | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.800 ± 0.007 | 0.970 ± 0.001 | 0.669 ± 0.012 | 0.792 ± 0.008 | 0.970 ± 0.001 | 0.028 ± 0.000 | 4.4 | 0.24 |
| Random Forest | 0.771 ± 0.007 | 0.968 ± 0.001 | 0.617 ± 0.013 | 0.754 ± 0.010 | 0.967 ± 0.002 | 0.027 ± 0.000 | 20.1 | 1.70 |
| MLP (PyTorch) | 0.808 ± 0.011 | 0.953 ± 0.020 | 0.697 ± 0.009 | 0.805 ± 0.011 | 0.922 ± 0.002 | 0.046 ± 0.020 | 2.5 | 0.11 |
| Logistic Regression | 0.751 | 0.915 | 0.621 | 0.740 | 0.785 | 0.077 | 0.3 | 0.00 |

### 5-class (normal / DoS / Probe / R2L / U2R) - KDDTest+, mean ± std over seeds

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| XGBoost | 0.782 ± 0.001 | 0.813 ± 0.002 | 0.550 ± 0.003 | 0.579 ± 0.004 | 0.748 ± 0.002 |
| Random Forest | 0.747 ± 0.001 | 0.787 ± 0.003 | 0.510 ± 0.001 | 0.533 ± 0.002 | 0.709 ± 0.000 |
| MLP (PyTorch) | 0.769 ± 0.011 | 0.742 ± 0.016 | 0.641 ± 0.003 | 0.641 ± 0.025 | 0.754 ± 0.018 |
| Logistic Regression | 0.766 | 0.772 | 0.603 | 0.623 | 0.736 |

### Generalisation (binary F1, seed 0 / mean) and novel-attack detection

| Model | Val F1 | Test F1 | Detection: known attacks | Detection: novel attacks |
|---|---|---|---|---|
| XGBoost | 0.999 | 0.792 | 78.7% | 42.2% |
| Random Forest | 0.998 | 0.754 | 77.2% | 30.6% |
| MLP (PyTorch) | 0.998 | 0.805 | 81.1% | 39.2% |
| Logistic Regression | 0.974 | 0.740 | 71.5% | 39.3% |
