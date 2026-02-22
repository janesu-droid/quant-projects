# Cryptocurrency Directional Forecasting: Deep Learning & Data Leakage Mitigation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains an advanced quantitative deep learning pipeline designed to forecast the directional movement of Bitcoin (BTC-USD). The core focus of this project is not only on applying state-of-the-art neural network architectures to algorithmic trading but also on demonstrating the critical impact of **Look-Ahead Bias (Data Leakage)** in financial time-series forecasting. 

The framework systematically transitions from a naive methodology (intentionally incorporating data leakage to expose overfitting) to a robust **Embargoed Walk-Forward Validation** scheme, successfully neutralizing look-ahead bias and generating realistic, out-of-sample portfolio returns.

## Table of Contents

- [Methodology & Architecture](#methodology--architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Experimental Results](#results--final-analysis)
- [Disclaimer](#disclaimer)

## Methodology & Architecture

The pipeline evaluates three distinct Deep Learning architectures and structures the experiment across three progressive validation steps:

### Neural Network Architectures Evaluated:
1. **Multilayer Perceptron (MLP):** A densely connected baseline network for capturing non-linear relationships.
2. **Long Short-Term Memory (LSTM):** A recurrent neural network specifically suited for exploiting the temporal sequence properties of crypto returns.
3. **CNN with Gramian Angular Field (CNN-GAF):** An advanced computer vision approach utilizing the `pyts` library to encode 1D financial time-series arrays into 2D polar-coordinate images, subsequently processed by Convolutional Neural Networks (CNN) for spatial pattern recognition.

### Experimental Phases:
* **Step 1: The Leakage Trap (Standard Train/Test Split)**
  A naive train/test partition where the 30-day forward cumulative return target overlaps chronologically with past training features. This acts as a conceptual trap, yielding artificially inflated metrics (e.g., AUC near 1.0) and unrealistic compounding strategy returns.
  
* **Step 2: Standard Walk-Forward Validation**
  Introduces rolling window backtesting (e.g., 500 train / 500 test & 500 train / 100 test). While temporally logical, standard walk-forward mechanics fail to purge the serial correlation overlap in the target boundaries, meaning overfitting strictly persists.

* **Step 3: Embargoed Walk-Forward**
  Implements rigorous quantitative hygiene. An absolute **30-day Embargo Period** is strictly enforced between the chronologically rolling training and testing sets. This mathematical "air-gap" completely bleeds out serial correlation and eradicates look-ahead bias, surfacing the true underlying alpha capabilities of the deep learning models.

## Project Structure

The operational lifecycle is encapsulated sequentially within the Jupyter Notebook:
- `Data Gathering & Initial Exploratory Data Analysis (EDA)`
- `Feature Engineering & Leakage Label Design`
- `Deep Learning Model Initialization (MLP, LSTM, GAF-CNN)`
- `Walk-Forward Backtesting Mechanics`
- `Embargo Implementation & Overfitting Disappearance Analysis`

## Requirements

Ensure your environment is running Python 3.8+. To run the computer vision time-series transformation, you will explicitly need the `pyts` library.

```bash
# Core Machine Learning & Data Processing
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn yfinance

# Deep Learning Framework
pip install tensorflow

# Time Series Space-to-Image Encoding
pip install pyts
```

## Results & Final Analysis

The terminal analytics matrix compares the **AUC** and **Strategy Cumulative Returns** horizontally across Step 1 (Leakage), Step 2 (Rolling), and Step 3 (Embargoed). 

The primary takeaway is the visible collapse in performance metrics from Step 1 to Step 3. This confirms the successful structural eradication of overfitting. The survival of any positive alpha in the Embargoed phase serves as the sole true indicator of the model's out-of-sample predictive robustness.

## Disclaimer

This project is for academic and educational purposes only. The models and trading algorithms developed herein are designed as experiments in mitigating look-ahead bias and do not constitute actionable financial or investment advice.

---
*Developed as part of the WQU Deep Learning Curriculum.*
