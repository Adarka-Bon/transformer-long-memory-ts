# Transformer Architectures for Long-Memory Time Series Forecasting

This repository contains the implementation and experimental study of several transformer-based architectures for financial time series forecasting with long-range temporal dependencies.

The project was developed as part of a diploma/master thesis focused on analyzing the ability of transformer models to capture long-memory behavior in sequential financial data and comparing different attention-based approaches for forecasting tasks.

Unlike classical statistical methods, transformer architectures allow the model to learn complex nonlinear dependencies directly from historical observations. At the same time, financial time series often exhibit long-memory properties, nonstationarity, and multi-scale temporal dynamics, which makes forecasting particularly challenging.

The main objective of this work is to investigate how different transformer architectures behave under such conditions and whether attention mechanisms are capable of effectively modeling long-range dependencies in financial sequences.

---

# Implemented Architectures

## Standard Transformer

The baseline encoder-only transformer architecture with:
- positional encoding;
- multi-head self-attention;
- feed-forward layers;
- residual connections and normalization.

This model serves as a reference architecture for evaluating forecasting quality and temporal dependency modeling.

---

## ProbSparse Transformer

An Informer-inspired sparse attention architecture designed to reduce the quadratic complexity of full self-attention.

Instead of computing attention scores for all token pairs, the model selects the most informative queries using a ProbSparse mechanism, allowing more efficient processing of long sequences while preserving important temporal dependencies.

The implementation includes:
- sparse attention masking;
- top-k query selection;
- transformer encoder blocks;
- forecasting projection head.

---

## MultiFeature Transformer

The main architecture proposed and investigated in this work.

The model combines two parallel representation pathways:

### Temporal Path
Processes dependencies along the time dimension using self-attention over sequential observations.

### Variate Path
Processes interactions between features by applying attention across variables rather than timesteps.

The extracted representations are fused using an adaptive gating mechanism:

```math
H = gH_{temp} + (1-g)H_{var}
```

This design allows the model to simultaneously capture:
- long-range temporal patterns;
- feature-wise dependencies;
- multi-dimensional interactions in financial sequences.

---

# Experimental Pipeline

The notebook contains the complete experimental workflow, including:
- data loading and preprocessing;
- feature engineering;
- sequence generation;
- transformer training;
- forecasting experiments;
- metric evaluation;
- visualization of predictions and training dynamics.

The experiments were conducted on financial market time series data using transformer-based forecasting approaches implemented in PyTorch.

---

# Evaluation

The models are evaluated using several regression metrics:
- Mean Squared Error (MSE);
- Root Mean Squared Error (RMSE);
- Mean Absolute Error (MAE);
- Directional Accuracy.

Comparative experiments demonstrate that architectures incorporating feature-wise interactions and sparse attention mechanisms provide more stable forecasting performance on long sequential windows compared to the baseline transformer model.

In particular, the MultiFeature Transformer showed improved ability to capture complex temporal dynamics and inter-feature relationships in financial data.

---

# Technologies

The project was implemented using:
- Python;
- PyTorch;
- NumPy;
- Pandas;
- Scikit-learn;
- Matplotlib;
- Jupyter Notebook.

---

# Usage

Open the notebook:

```bash
jupyter notebook transform.ipynb
```

---

# Research Context

This repository represents the practical and experimental part of a diploma/master thesis devoted to transformer architectures for long-memory time series forecasting.

The work combines theoretical analysis of attention mechanisms with empirical experiments on financial sequential data and focuses on the study of long-range dependency modeling in transformer-based forecasting systems.

---

# License

This project is released under the MIT License.
