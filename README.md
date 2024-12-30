# Logistic Regression Simulation

This repository contains a Python script that simulates a binary classification problem using logistic regression principles. The script generates synthetic data, applies logistic regression to model the relationship between predictor variables and a binary outcome, and visualizes the results.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Code Explanation](#code-explanation)
- [Visualization](#visualization)

## Overview

The script generates a dataset with two predictor variables (`x1` and `x2`) and a binary outcome variable (`y`). The relationship between the predictors and the outcome is modeled using logistic regression. The generated data is visualized using scatter plots to illustrate the relationship between the predictors and the binary outcome.

## Requirements

To run the code, you will need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib
```
## Code Explaination: 
### Importing libraries :
```
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
```
### Set Seed for Reproducibility:
```
np.random.seed(123)
```
This ensures that the random numbers generated can be replicated.
### Define Parameters:
```
N = 1000
theta_vec = np.array([-2, 0.5, 0.2])
```
N: Number of samples to generate (1000).
theta_vec: Coefficients for the logistic regression model.
### Generate Predictor Variables:
```
x1 = np.random.normal(loc=2, scale=1, size=N)
x2 = np.random.normal(loc=10, scale=2, size=N)
```
Two predictor variables are generated from normal distributions.
### Calculate Linear Combination:
```
h_theta = np.column_stack((np.ones(N), x1, x2)).dot(theta_vec)
```
This computes the linear combination of the predictors using the coefficients.
### Calculate Probabilities:
```
prob = 1 / (1 + np.exp(-h_theta))
```
The logistic function is applied to obtain probabilities.
### Generate Binary Outcomes:
```
y = np.random.binomial(p=prob, n=1, size=N)
```
Binary outcomes are generated based on the calculated probabilities.
### Create DataFrame:
```
data_mat = pd.DataFrame((y, x1, x2), index=["y", "x1", "x2"]).T
```
A pandas DataFrame is created to hold the generated data.
## Visualization
```
fig = plt.figure(figsize=(10, 8))
_ = fig.add_subplot(2, 1, 1).scatter(x1, y, edgecolor="blue", color="None")
_ = plt.xlabel("x1")
_ = fig.add_subplot(2, 1, 2).scatter(x2, y, edgecolor="blue", color="None")
_ = plt.xlabel("x2")
_ = plt.tight_layout()
plt.show()
```
The first subplot shows the relationship between x1 and y, while the second subplot shows the relationship between x2 and y.
<img width="562" alt="Screenshot 2024-12-30 234924" src="https://github.com/user-attachments/assets/3be1d59e-510f-4431-b8ff-2d3e9c9ac91a" />



