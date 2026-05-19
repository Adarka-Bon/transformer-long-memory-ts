# Transformer Architectures for Long-Memory Time Series Forecasting

This repository contains implementations and experiments for transformer-based architectures applied to long-memory financial time series forecasting.

The project investigates the capability of transformer attention mechanisms to model long-range temporal dependencies in stock market data and evaluates different architectural approaches for sequential prediction tasks.

---

# Implemented Architectures

## Standard Transformer
Encoder-only transformer architecture with full self-attention mechanism and positional encoding for temporal sequence modeling.

## ProbSparse Transformer
Sparse attention architecture inspired by Informer, designed to reduce computational complexity while preserving long-range dependency modeling.

## MultiFeature Transformer
Hybrid architecture combining:
- temporal attention pathway;
- feature-wise attention pathway;
- adaptive fusion mechanism.

The model simultaneously captures:
- temporal dependencies;
- inter-feature relationships;
- long-memory dynamics.

---

# Research Objectives

- Analyze transformer architectures for long-memory time series forecasting;
- Investigate attention mechanisms for long-range dependency modeling;
- Compare dense and sparse attention approaches;
- Explore temporal and variate-wise feature interactions;
- Evaluate forecasting quality on financial sequential data.

---

# Methodology

The experimental pipeline includes:
- financial time series preprocessing;
- feature engineering;
- sequence generation;
- transformer-based forecasting;
- comparative evaluation;
- prediction visualization and analysis.

---

# Evaluation Metrics

The models are evaluated using:
- Mean Squared Error (MSE);
- Root Mean Squared Error (RMSE);
- Mean Absolute Error (MAE);
- Directional Accuracy.

---

# Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# Usage

Install dependencies, then open the notebook:

```bash
jupyter notebook transform.ipynb
```
annd enjoy:)

---

# Research Context

This project was developed as part of a diploma/master thesis focused on transformer architectures for long-memory time series forecasting and financial sequence modeling.

---

# License

This project is released under the MIT License.
