# Paper Template

## Data Analysis

### Load and Read the Data

### Identify the Outlier Using IQR

### Data Normalization

### Feature Engineering

### Feature Selection

## ML Algorithms

### MLP Regression

### ElasticNet

Balanced Regularization: ElasticNet strikes a balance between L1 (Lasso) and L2 (Ridge) regularization, making it suitable for AI-driven calibration tasks. It effectively handles multicollinearity and feature selection, ensuring that only relevant sensor inputs influence the calibration process.

### Support Vector Machine (SVM)

Non-linear Mapping: SVR is chosen for AI-driven calibration tasks due to its capability to map non-linear relationships between sensor inputs and calibration outputs effectively. This is crucial in scenarios where sensor drift introduces complex variations that need precise calibration adjustments.

### RandomForestRegressor

Ensemble Learning Benefits: Random Forest Regressor benefits AI-driven calibration because it leverages ensemble learning to aggregate predictions from multiple decision trees. This approach improves generalization and resilience against overfitting, which is crucial in handling diverse sensor data affected by drift. Each decision tree in Random Forest focuses on different subsets of data and features, making it robust against noisy sensor measurements. This capability ensures that AI-driven calibration remains accurate despite fluctuations caused by sensor drift.

## Results

### Visualization of RandomForestRegressor

![Best Model Visualization](best_model.png)

### Models Evaluations

| Model              | MSE    | R^2    | Training Time (s) | Prediction Time (s) | Predicted True Measurements |
|--------------------|--------|--------|--------------------|----------------------|-----------------------------|
| MLP Regression     | 0.0004 | 0.9886 | 0.671              | 0.00226              | 0.97314647                  |
| ElasticNet         | 0.0008 | 0.9748 | 0.0826             | 0.00054              | 0.95499576                  |
| SVR                | 0.0013 | 0.9611 | 0.258              | 0.00884              | 0.91607862                  |
| RandomForestRegre  | 0.0002 | 0.9943 | 3.0521             | 0.07438              | 0.96357482                  |

- **MLP Regression** achieved the lowest MSE (0.0004) and highest R^2 (0.9886), indicating very accurate predictions and a good model fit.
- **RandomForestRegressor** also performed well with very low MSE (0.0002) and high R^2 (0.9943), albeit with longer training and prediction times compared to other models.
- **ElasticNet** and **SVR** show slightly higher MSE and lower R^2 compared to MLP and RandomForest, suggesting they may be slightly less accurate for this particular calibration task.

Models like MLP Regression and RandomForestRegressor demonstrate high accuracy (low MSE) and precision (R^2 close to 1), making them suitable for precise calibration tasks where accurate measurement retrieval from sensor data is essential.

## Conclusions

In summary:
- **MLP Regression** achieved the lowest MSE and highest R^2.
- **RandomForestRegressor** showed robust performance with low MSE and high R^2.
- **ElasticNet** and **SVR** exhibited slightly lower accuracy metrics.

These findings highlight the effectiveness of MLP Regression and RandomForestRegressor in accurate calibration tasks.

## GitHub

[Github Repository Link](https://github.com/InasALKamachy/ML-for-calibration/tree/C)
