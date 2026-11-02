# Industrial Predictive Maintenance Using Machine Learning

An end-to-end machine learning project that predicts potential industrial machine failures using sensor data from the AI4I 2020 Predictive Maintenance Dataset. The project compares multiple ML algorithms, handles class imbalance, optimizes the decision threshold, and uses SHAP for model explainability.

## Project Overview

Unexpected machine failures can cause production downtime, financial losses, and maintenance costs.

This project aims to predict whether an industrial machine is likely to fail based on its operating conditions and sensor measurements.

The final model uses **XGBoost** and is deployed through a **Streamlit web application**.

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

- Total records: 10,000
- Original features: 14
- Target variable: `Machine failure`
- Failure cases: 339 (3.39%)
- Non-failure cases: 9,661 (96.61%)

The dataset is highly imbalanced, making failure detection more challenging than ordinary binary classification.

## Features Used

The following six features were selected:

| Feature | Description |
|---|---|
| Type | Product quality variant (L, M, H) |
| Air temperature [K] | Air temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Machine torque |
| Tool wear [min] | Tool usage duration |

`UDI` and `Product ID` were excluded as identifiers. The failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were excluded to avoid target leakage.

## Machine Learning Workflow

### 1. Exploratory Data Analysis

- Dataset inspection and statistical analysis
- Target class distribution analysis
- Numerical feature distributions
- Boxplots and outlier inspection
- Understanding class imbalance

### 2. Data Preprocessing

- Selected relevant input features
- Train-test split (70:30)
- Stratified sampling to preserve class distribution
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer and Pipeline for preprocessing

### 3. Model Comparison

The following algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine
6. XGBoost

### 4. Handling Class Imbalance

Because only 3.39% of records represent machine failures, accuracy alone is not a reliable evaluation metric.

Different approaches were explored:

- Class weighting
- SMOTE
- XGBoost `scale_pos_weight`

### 5. Hyperparameter Tuning

RandomizedSearchCV with 5-fold Stratified Cross-Validation was used to tune XGBoost.

The search optimized the F1-score.

### 6. Decision Threshold Optimization

Instead of relying only on the default classification threshold of 0.5, out-of-fold training predictions were used to select a threshold that maximized the F1-score.

Selected threshold: **0.64506**

### 7. Model Explainability Using SHAP

SHAP (SHapley Additive exPlanations) was used to understand model predictions.

- Global feature importance
- Local prediction explanations
- Feature contributions to model output

Tool wear, torque, and rotational speed were among the most influential features.

These explanations describe learned model associations, not causal relationships.

## Model Performance

The final tuned XGBoost model, using the optimized decision threshold, achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 98% |
| Precision | 0.80 |
| Recall | 0.67 |
| F1-Score | 0.73 |

### Confusion Matrix

Using the standard scikit-learn convention:

| | Predicted No Failure | Predicted Failure |
|---|---:|---:|
| Actual No Failure | 2881 | 17 |
| Actual Failure | 34 | 68 |

The model correctly identified 68 out of 102 machine failure cases in the test set.

**Important validation limitation:** The test set was consulted repeatedly during model experimentation and threshold evaluation. Therefore, it is not an untouched independent holdout, and these results should not be interpreted as an unbiased estimate of real-world generalization.

## Streamlit Application

The trained model is integrated into a Streamlit application.

Application features:

- Input machine operating parameters
- Predict machine failure
- Display estimated failure score
- Apply the optimized classification threshold
- Display prediction results

The saved preprocessing pipeline allows the application to accept raw input features directly.

## Project Structure

```text
industrial-predictive-maintenance/
│
├── app/
│   └── app.py
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│   ├── xgboost_model.pkl
│   └── threshold.pkl
│
├── notebooks/
│   └── predictive_maintenance.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/devansh29-12/industrial-predictive-maintenance.git
```

### 2. Navigate to the project directory

```bash
cd industrial-predictive-maintenance
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows:

```powershell
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit
- Jupyter Notebook

## Limitations

- The dataset is a synthetic benchmark dataset and may not represent all real industrial environments.
- The dataset has significant class imbalance.
- Test-set reuse during experimentation limits the reliability of the final performance estimate.
- Predicted scores