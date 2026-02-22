# Advanced Portfolio Management: Machine Learning Enhancements

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Quant Finance](https://img.shields.io/badge/Quant-Finance-teal.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository houses a comprehensive quantitative portfolio management pipeline exploring the intersection of Modern Portfolio Theory (MPT) and advanced Machine Learning techniques. The project systematically enhances a baseline Mean-Variance Optimization (MVO) framework by integrating state-of-the-art methodologies designed to address the empirical shortcomings of traditional portfolio allocation.

The primary objective is to evaluate the isolated and combined impact of **Covariance Denoising (CREM)**, **Hierarchical Risk Parity (HRP) Clustering**, and **Combinatorial Purged Cross-Validation (CPCV) Backtesting** on out-of-sample portfolio performance and stability.

## Table of Contents

- [Methodology & Innovations](#methodology--innovations)
- [Project Architecture](#project-architecture)
- [Requirements](#requirements)
- [Experimental Results](#experimental-results)
- [Disclaimer](#disclaimer)

## Methodology & Innovations

The pipeline constructs and evaluates portfolios moving from a constrained baseline to highly advanced, machine-learning-augmented allocations.

### 1. Baseline Portfolio Optimization
* **Mean-Variance Optimization (MVO):** The foundational benchmark utilizing classical Sharpe ratio maximization.
* **Constraints:** Strictly long-only (no short selling) with a rigid maximum allocation constraint of 15% per asset to enforce baseline diversification.

### 2. Machine Learning Enhancements
The core of the project involves implementing and testing three sophisticated improvements:
* **Covariance Denoising (CREM):** Utilizing Random Matrix Theory (RMT) and Principal Component Analysis (PCA) to filter out market noise from the empirical covariance matrix. This stabilizes the inverse covariance matrix, mitigating the extreme weight instability typical of MVO.
* **Hierarchical Risk Parity (HRP) Clustering:** Employs graph theory and machine learning clustering (Tree Clustering, Quasi-Diagonalization, Recursive Bisection) to allocate capital based on the hierarchical structure of asset correlations, rather than relying on noisy expected return estimates.
* **Advanced Walk-Forward Backtesting (CPCV Principles):** Implements a rigorous validation framework incorporating **Purge** and **Embargo** periods. This mathematical "air-gap" actively prevents data leakage and look-ahead bias between training and testing folds, generating realistic out-of-sample estimates.

### 3. Synergistic Combinations
The experiment systematically evaluates permutations of these techniques:
* Denoising + Clustering (HRP utilizing a noise-filtered covariance matrix)
* Denoising + Advanced Backtesting
* Clustering + Advanced Backtesting
* The globally optimized combination: **Denoising + Clustering + Advanced Backtesting**

## Project Architecture

The operational workflow is structured progressively within the Jupyter Notebook (`Portfolio_Management with ML improvements.ipynb`):
- `Step 1: Baseline Portfolio Optimization` (Data fetching & constrained MVO)
- `Step 2: Implementation of Enhancements` (CREM Denoising, HRP Clustering, Walk-Forward Frameworks)
- `Step 3: Sub-system Integration & Method Combinations`
- `Step 4: In-Sample Performance Comparison & Analytics Mapping`
- `Step 5: Rigorous Out-of-Sample Testing & Final Evaluation`

## Requirements

Ensure your environment is running Python 3.8+. The pipeline requires standard quantitative finance and machine learning libraries.

```bash
# Core Quantitative & Data Processing Libraries
pip install pandas numpy scipy pathlib

# Data Acquisition
pip install yfinance
```

*(Note: While `scikit-learn` or specialized clustering libraries can be used, this notebook explicitly builds the core HRP and Denoising logic using foundational `scipy` and `numpy` functions for transparency and academic rigor.)*

## Experimental Results

The terminal analytics matrix compares the optimal weights, in-sample metrics, and out-of-sample performance across all architectural permutations. 

Key analytical focus areas include:
* **Weight Stability:** Observing the reduction in allocation concentration when moving from baseline MVO to Denoised HRP.
* **Out-of-Sample Sharpe & Drawdown:** Evaluating the true efficacy of the Purged/Embargoed walk-forward framework in predicting robust future performance relative to standard backtesting.

## Disclaimer

This repository is for academic, educational, and research purposes only. The models and allocation strategies developed herein do not constitute actionable financial, investment, or trading advice.

---
*Developed as part of the WQU MScFE 652 Portfolio Management Curriculum.*
