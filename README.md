# DEEPFIN

**Machine-learning financial forecasting from SEC EDGAR/XBRL filings**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![C++20](https://img.shields.io/badge/C%2B%2B-20-purple)
![Keras 3](https://img.shields.io/badge/Keras-3-red)
![JAX](https://img.shields.io/badge/Backend-JAX-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

DEEPFIN is a course-work research project that builds an empirical financial forecasting pipeline from public corporate filings. It transforms audited SEC EDGAR/XBRL financial statements into standardized financial tensors and uses them to predict future company-level financial indicators.

The system combines:

- a Python data pipeline for SEC data collection and tensor construction;
- a custom C++20 financial calculation engine exposed to Python through Pybind11;
- a JAX-backed Keras 3 recurrent neural model;
- company-level train/validation splitting to reduce leakage;
- comparison against classical machine-learning baselines.

DEEPFIN should be understood as a forecasting and decision-support prototype, not as an autonomous financial planning system. Its goal is to test whether structured corporate financial trajectories can be modeled as sequences and whether a neural sequence model can generalize to unseen companies.

## Research Scope

The project addresses the gap between traditional spreadsheet-based financial analysis and modern machine-learning methods. Public companies disclose large volumes of structured financial data, but this data must be collected, harmonized, converted into temporal samples, and validated carefully before it can support forecasting.

The central research question is:

> Can a deep learning model trained on standardized SEC financial tensors produce competitive or stronger forecasts of future corporate financial indicators than classical machine-learning models and static financial-analysis logic?

The implementation focuses on:

- SEC EDGAR annual filing data;
- S&P 500 company metadata for the main dataset;
- fiscal years 2017-2025 by default;
- company-level validation rather than random window mixing;
- external stress validation on random SEC-listed companies.

## Pipeline

1. **SEC Data Collection**
   - Synchronizes ticker-CIK mappings.
   - Collects annual company filing data.
   - Stores reusable local filing data and metadata.

2. **Financial Feature Calculation**
   - Uses a C++20 calculation engine for deterministic financial formulas.
   - Computes raw operational indicators and derived risk/profitability features.
   - Includes Altman Z-Score and solvency ratio calculations.

3. **Tensor Construction**
   - Converts company-year observations into standardized financial tensors.
   - Builds sequential supervised samples from historical company trajectories.
   - Saves raw tensors and company-level train/validation splits.

4. **Model Training**
   - Uses Keras 3 with the JAX backend.
   - Trains an LSTM-based sequence model with residual dense features.
   - Uses Huber loss, AdamW, dropout, gradient clipping, early stopping, and checkpointing.

5. **Evaluation**
   - Compares DEEPFIN with Linear Regression, Ridge, Lasso, ElasticNet, SVR, KNN, Decision Tree, Random Forest, Gradient Boosting, and AdaBoost.
   - Reports R2, MSE, and MAE.
   - Tests both controlled validation and external random SEC-company samples.

## Financial Tensor Features

The tensor representation combines accounting facts, derived financial indicators, and contextual values, including:

- revenue;
- cost of goods sold;
- SG&A;
- research and development;
- capital expenditure;
- inventory;
- Altman Z-Score;
- solvency ratio;
- EBITDA margin;
- EBITDA-to-book-equity proxy;
- revenue velocity;
- net income margin;
- book-equity proxy;
- total assets;
- sector encoding when available.

The report notes that the field named `MarketCap` in the implementation is populated from statement data as assets minus liabilities. It should therefore be interpreted as a book-equity proxy, not observed stock-market capitalization.

## Model Architecture

DEEPFIN treats each company as a financial time series. Each sample uses a short sequence of historical annual financial tensors to predict a future financial state.

The neural architecture includes:

- input cleaning and clipping;
- per-sample, per-feature scaling;
- layer normalization;
- bidirectional LSTM sequence representation;
- dropout regularization;
- a second LSTM layer;
- a flattened residual branch from the original input;
- dense GELU layers;
- a final dense forecast output.

An LSTM architecture was selected instead of a Transformer because the current task uses short annual accounting sequences. The recurrent model provides a direct sequential inductive bias for ordered annual financial statements.

## Results Summary

In the report's fold-based evaluation, DEEPFIN achieved the strongest mean metrics among the tested models:

| Model | R2 | MSE | MAE |
| --- | ---: | ---: | ---: |
| DEEPFIN | 0.4257 | 0.3610 | 0.1528 |
| ElasticNet | 0.3826 | 0.3882 | 0.1659 |
| Lasso | 0.3822 | 0.3884 | 0.1661 |
| Ridge | 0.3754 | 0.3926 | 0.1722 |
| Linear Regression | 0.3747 | 0.3930 | 0.1727 |

On holdout validation, ElasticNet achieved the best R2, while DEEPFIN achieved the lowest MAE:

| Model | R2 | MSE | MAE |
| --- | ---: | ---: | ---: |
| ElasticNet | 0.3743 | 0.4094 | 0.1631 |
| DEEPFIN | 0.3675 | 0.4138 | 0.1621 |

External validation on 50 random SEC-listed companies for target year 2025 produced:

| Metric | Value |
| --- | ---: |
| Tensor R2 | 0.2836 |
| Tensor MSE | 1.1328 |
| Tensor MAE | 0.3335 |

These results show that DEEPFIN captures useful predictive signal across controlled validation, company-level holdout testing, and external SEC-company evaluation.

## Installation

Requirements:

- Python 3.10+
- C++20 compiler
- CMake
- Python dependencies from `requirements.txt`

Clone the repository:

```bash
git clone https://github.com/bikmarik/course_work.git
cd course_work
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build/install the local package and C++ extension:

```bash
pip install -e .
```

## Usage

Build the dataset:

```bash
python build_dataset.py
```

Train the model:

```bash
python train.py
```

Evaluate the model:

```bash
python evaluate_500.py
```

Run prediction:

```bash
python predict.py
```

## Repository Structure

```text
.
├── build_dataset.py
├── evaluate_500.py
├── predict.py
├── train.py
├── requirements.txt
├── setup.py
├── CMakeLists.txt
├── scripts/
└── src/
    ├── dataGenius/
    │   ├── calculate_data.cpp
    │   └── process_data.py
    └── dfmaker3000.py
```
