# Crude Oil Price Regime Forecasting & Automated Trading Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a quantitative framework designed to model, forecast, and computationally trade crude oil price regimes. The methodology bridges hidden state extraction with probabilistic graphical models, utilizing **Gaussian Hidden Markov Models (HMM)** to discretize continuous macro-financial time series and **Dynamic Bayesian Networks (DBN)** to infer acyclic structural causal relationships amongst macroeconomic drivers.

The final component of this pipeline translates predictive causal inference into an actionable, backtested quantitative trading strategy, evaluating the framework's alpha-generation capabilities against benchmark portfolios.

## Table of Contents

- [Methodology & Architecture](#methodology--architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results & Backtesting](#results--backtesting)
- [Disclaimer](#disclaimer)

## Methodology & Architecture

The analytical pipeline is sequentially structured into five primary modules:

1. **Data Engineering & Ingestion:** 
   Programmatic extraction and harmonization of disparate datasets via the U.S. Energy Information Administration (EIA) and Federal Reserve Economic Data (FRED) APIs. Features encompass OPEC capacity, OECD strategic reserves, global FOREX indices, and WTI benchmarks.
   
2. **Latent Regime Extraction (HMM):** 
   To circumvent the limitations of discrete Bayesian networks acting on continuous financial data, the pipeline trains stationary 3-state Gaussian HMMs via the Baum-Welch algorithm. Viterbi decoding is subsequently applied to map continuous non-stationary returns into discrete temporal market regimes (e.g., low, medium, and high volatility states).

3. **Causal Structure Learning (DBN):** 
   Deploys a Hill-Climbing heuristic optimized with the Bayesian Information Criterion (BIC) to autonomously discover the Directed Acyclic Graph (DAG). This maps the conditional dependencies between global macroeconomic factors and the target variable (WTI regime).

4. **Out-of-Sample Validation & Inference:** 
   Rigorous hold-out validation ensures the temporal integrity of the model. Historical HMM emission matrices are frozen to safely discretize out-of-sample test arrays prior to probabilistic Bayesian inference, averting data leakage.

5. **Strategy Backtesting:** 
   Translates maximum a posteriori (MAP) regime predictions into standard Long/Short directional market exposures. Features dynamic equity curve computation and performance benchmarking.

## Project Structure

The operational lifecycle is encapsulated sequentially within the Jupyter Notebook:
- `Data Extraction & Cleansing`
- `HMM Initialization & State Discretization`
- `Bayesian Network Topology Mapping`
- `Model Validation and Inference`
- `Strategy Backtest Engine`

## Installation & Setup

### Requirements

Ensure your environment is running Python 3.8 to 3.10. Due to specific dependency constraints (e.g., `causalnex`), Python 3.11+ is not recommended for this particular deployment.

### Environment Implementation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Initialize a dedicated Conda environment
conda create -n bayesian_trading_env python=3.10
conda activate bayesian_trading_env

# Install requisite libraries
pip install pandas numpy requests hmmlearn causalnex pgmpy matplotlib seaborn
```

*Note: You must provision your own API keys for both the EIA and FRED to execute the data ingestion blocks.*

## Usage

1. Open the primary computational notebook:
   ```bash
   jupyter notebook "GWP_1_Submission_Code copy.ipynb"
   ```
2. Inject your respective EIA and FRED API endpoints in the designated configuration blocks.
3. Execute the pipeline sequentially. Be advised that the internal structure learning components (Hill-Climbing) involve high computational complexity and may necessitate substantial processing time depending on data cardinality.

## Results & Backtesting

The terminal output of the pipeline generates standard quantitative performance metrics:
- **Classification Accuracy:** Directional hit rate of the Bayesian Network predictions.
- **Backtest Visualization:** A comparative equity curve mapping the cumulative returns of the algorithm's active trading strategy against a passive 'Buy and Hold' WTI benchmark. Note that temporal differences in historical data inputs (e.g., 2004-2020 vs. older dissertation constraints) naturally yield variance in gross strategy output.

## Disclaimer

This project is for academic and educational purposes only. The models and trading algorithms developed herein are not indicative of future market performance and do not constitute financial advice.

---
*Developed as part of the WorldQuant University Risk Management Curriculum.*

