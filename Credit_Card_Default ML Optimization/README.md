# Quantitative Machine Learning Optimization: Bias-Variance & Hyperparameter Search

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains an advanced pipeline demonstrating fundamental and modern machine learning challenges: **The Bias-Variance Tradeoff** and **High-Dimensional Hyperparameter Optimization**. By leveraging empirical frameworks on structured datasets (Insurance pricing and Taiwan Credit Card defaults), this project bridges theoretical machine learning constraints with robust, computationally efficient optimization techniques.

The project systematically isolates mathematical complexities such as polynomial overfitting, and subsequently tackles hyperparameter search space navigation using Deep Neural Networks (DNNs). 

## Table of Contents

- [Methodology & Architecture](#methodology--architecture)
  - [Part I: The Bias-Variance Tradeoff](#part-i-the-bias-variance-tradeoff)
  - [Part II: Advanced Hyperparameter Optimization](#part-ii-advanced-hyperparameter-optimization)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Experimental Results](#experimental-results)
- [Disclaimer](#disclaimer)

## Methodology & Architecture

The architecture is divided into two mathematically distinct but practically interconnected quantitative phases:

### Part I: The Bias-Variance Tradeoff
* **Dataset:** US Health Insurance Costs.
* **Objective:** Empirically demonstrate model memorization vs. generalization by escalating model complexity.
* **Implementation:** 
  Utilizes **Polynomial Regression** across ascending dimensional degrees (1 through 30). By capturing both In-Sample (Train) and Out-of-Sample (Test) MSE, the pipeline visualizes the exact inflexion point where irreducible variance violently overcomes bias, causing the model to collapse on unseen data.

### Part II: Advanced Hyperparameter Optimization
* **Dataset:** Taiwan Credit Card Defaults (Imbalanced Binary Classification).
* **Objective:** Maximize Deep Neural Network (DNN) predictive accuracy through rigorous hyperparameter space exploration.
* **Optimization Frameworks:**
  1. **Grid Search (Exhaustive Search):** The brute-force combinatorial baseline. Highly parallelizable but computationally prohibitive in high-dimensional spaces.
  2. **Random Search:** A statistically superior technique for high-dimensional spaces, relying on the mathematical distribution of hyperparameter importance to find near-optimal configs using significantly fewer compute cycles.
  3. **Bayesian Optimization:** The state-of-the-art technique utilizing Gaussian Processes. It builds a probabilistic surrogate model of the objective function, mathematically balancing exploration (uncertain spaces) and exploitation (known optimal regimes) to converge on optimal parameters exceptionally fast.

### Neural Network Architecture:
Flexible Dense baseline architectures regularized with internal Dropout and `L2` weight decay. Built with TensorFlow/Keras and encapsulated via `scikeras` for deep integration with the Scikit-Learn optimization API.

## Project Structure

The operational lifecycle is encapsulated sequentially within the Jupyter Notebook (`CreditCard_Default_ML_Optimization.ipynb`):
- `Data Gathering & Preprocessing (Outlier Mitigation)`
- `Bias-Variance Target Isolation (Polynomial Regression)`
- `DNN Baseline Initialization & Scaling`
- `Combinatorial Grid Search Execution`
- `Stochastic Random Search Execution`
- `Gaussian Process Bayesian Optimization Execution`
- `Surrogate Model Extrapolation & Best Estimator Analytics`

## Requirements

Ensure your environment is running Python 3.8+. You will need standard quantitative libraries along with deep learning frameworks.

```bash
# Core Quantitative & ML Libraries
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Deep Learning Framework & Keras Wrapper
pip install tensorflow scikeras

# Specialized Optimization Libraries (if testing Bayesian extensions)
pip install scikit-optimize
```

## Experimental Results

The terminal analytics matrix compares the computational efficiency and Out-of-Sample validation metrics of the evaluated search algorithms. 

Key analytical focus areas include:
* **Overfitting Identification:** Observing the structural divergence of Train vs. Test error trajectories during polynomial degree escalation.
* **Optimizer Convergence Efficiency:** Measuring the temporal cost and metric output differential between Grid Search, Random Search, and the targeted Bayesian Optimization scheme. Bayesian Optimization is expected to locate the global maxima with a substantially lower iteration count.

## Disclaimer

This repository is for academic, educational, and research purposes only. The models and analytical frameworks developed herein do not constitute actionable financial or professional advice.

---
*Developed as part of the WorldQuant University Machine Learning Curriculum.*
