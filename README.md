# - AI-Pipeline for Industrial Sensor Data Assimilation

Author: Inas AL-Kamachy

## Data Analysis

### Load and Read the Data
First, a function `read_and_clean_svmlight` is defined to read data files related to SVMlight format and perform basic cleaning by removing semicolons from the index. This operation loops over the 10 batch files, combining all loaded features (sensor readings) into a single DataFrame `X` and target labels (gas types) into a Series `Y`, then concatenates the features and the target in one dataframe.

### Identify the Outlier Using IQR
Using the Interquartile Range (IQR) method, outliers are identified and replaced with the column median.

### Data Normalization
Data is scaled to the range (-1, 1), as recommended in the [Gas Sensor Array Drift Dataset](http://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset).

### Feature Engineering
Additional features named (month) are added, which contain the data volume for different sample gases in 10 batches.

### Feature Selection
The main `data.csv` shape is (13910, 129). To reduce dimensionality, an XGBoost Regression (`XGBRegressor`) is trained for regression tasks, evaluating performance on training and testing data (`eval_set`). The `early_stopping_rounds` parameter aids in the automatic termination of training based on validation set performance, guiding effective feature selection. The final `data.csv` shape was (13910, 11).

## ML Algorithms
We use four different ML algorithms:

- MLP Regression
- ElasticNet 
- SVR
- Random Forest Regressor

### MLP Regression
MLP (Multi-layer Perceptron) Regression is well-suited for calibration tasks involving AI because it can model complex non-linear relationships between sensor inputs and outputs. This capability is crucial in AI-driven calibration where the relationship between sensor readings and actual values may be intricate and non-linear.

### ElasticNet
ElasticNet strikes a balance between L1 (Lasso) and L2 (Ridge) regularization, making it suitable for AI-driven calibration tasks. It effectively handles multicollinearity and feature selection, ensuring that only relevant sensor inputs influence the calibration process.

### Support Vector Machine (SVM)
SVR is chosen for AI-driven calibration tasks due to its capability to map non-linear relationships between sensor inputs and calibration outputs effectively. This is crucial in scenarios where sensor drift introduces complex variations that need precise calibration adjustments.

### RandomForestRegressor
Random Forest Regressor benefits AI-driven calibration because it leverages ensemble learning to aggregate predictions from multiple decision trees. This approach improves generalization and resilience against overfitting, which is crucial in handling diverse sensor data affected by drift. Each decision tree in Random Forest focuses on different subsets of data and features, making it robust against noisy sensor measurements. This capability ensures that AI-driven calibration remains accurate despite fluctuations caused by sensor drift.

## Results

### Visualization of RandomForestRegressor
![Best Model Visualization](best_model.png)

### Models Evaluations

| Model              | MSE    | \( R^2 \)  | Training Time (s) | Prediction Time (s) | Predicted True Measurements |
|--------------------|--------|------------|--------------------|---------------------|-----------------------------|
| MLP Regression     | 0.0004 | 0.9886     | 0.671              | 0.00226             | 0.97314647                  |
| ElasticNet         | 0.0008 | 0.9748     | 0.0826             | 0.00054             | 0.95499576                  |
| SVR                | 0.0013 | 0.9611     | 0.258              | 0.00884             | 0.91607862                  |
| RandomForestRegre  | 0.0002 | 0.9943     | 3.0521             | 0.07438             | 0.96357482                  |

Performance Metrics of Various Regression Models

### Conclusions

For each model, GridSearchCV was employed to identify the optimal parameters, ensuring the models were fine-tuned for accurate predictions. Mean Squared Error (MSE) and \( R^2 \) were used as evaluation metrics, where MSE quantifies the average prediction error and \( R^2 \) measures the model's fit to the actual measurements. These metrics are crucial in assessing the accuracy and effectiveness of inverse modeling for sensor calibration. MLP Regression demonstrated exceptional performance with the lowest MSE recorded at 0.0004 and the highest \( R^2 \) of 0.9886, highlighting its ability to make highly accurate predictions and achieve a robust fit to the data. RandomForestRegressor also yielded impressive results, boasting a remarkably low MSE of 0.0002 and a high \( R^2 \) of 0.9943. Despite longer training and prediction times compared to other models, its ensemble learning approach contributed to its robust performance. On the other hand, ElasticNet and SVR exhibited slightly higher MSE values and lower \( R^2 \) scores compared to MLP and RandomForestRegressor. This suggests they may be slightly less precise for this specific calibration task, potentially due to the complexity and non-linearity of the sensor data. In conclusion, MLP Regression and RandomForestRegressor emerged as standout models, demonstrating not only high accuracy (low MSE) but also precision (\( R^2 \) close to 1). These attributes make them particularly well-suited for demanding calibration tasks where retrieving accurate measurements from sensor data is critical.

## GitHub Repository
[GitHub Repository Link](https://github.com/InasALKamachy/ML-for-calibration/tree/C)
