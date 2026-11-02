# Industrial Predictive Maintenance using Machine Learning

An end-to-end Machine Learning project that predicts machine failures using operating conditions and sensor measurements. The project uses the AI4I 2020 Predictive Maintenance dataset and a tuned XGBoost classifier, with a Streamlit application for interactive predictions.

## Project Overview

Unexpected machine failures can lead to production downtime, maintenance costs and operational inefficiencies.

This project explores how machine learning can identify potential machine failures based on operating parameters such as air temperature, process temperature, rotational speed, torque and tool wear.

The objective is to build an end-to-end ML pipeline covering data preprocessing, exploratory data analysis, model comparison, class imbalance handling, hyperparameter tuning, threshold optimization, model explainability and deployment.

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

- Total records: 10,000
- Original features: 14 columns
- Target variable: `Machine failure`
- Failure cases: 339 (3.39%)
- Non-failure cases: 9,661 (96.61%)

The dataset is highly imbalanced, making failure detection more challenging than ordinary binary classification.

### Selected Features

| Feature | Description |
|---|---|
| Type | Product quality variant (L, M, H) |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Machine process temperature |
| Rotational speed [rpm] | Rotational speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Accumulated tool wear |

**Target:** `Machine failure`

UDI and Product ID were removed because they are identifiers. The failure-mode indicators (TWF, HDF, PWF, OSF and RNF) were excluded to avoid target leakage.

## Machine Learning Workflow

### 1. Exploratory Data Analysis

- Dataset structure and statistical analysis
- Missing-value and duplicate checks
- Target class distribution analysis
- Numerical feature distributions
- Outlier investigation using boxplots
- Correlation and feature exploration

Genuine observations were retained rather than automatically removing outliers.

### 2. Data Preprocessing

- Train-test split with stratification
- Numerical feature scaling using StandardScaler
- Categorical encoding using OneHotEncoder
- ColumnTransformer for preprocessing
- Pipeline-based model training to prevent preprocessing leakage

### 3. Model Comparison

The following algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost

Since the dataset is highly imbalanced, model performance was evaluated using precision, recall, F1-score and confusion matrices rather than accuracy alone.

### 4. Handling Class Imbalance

The following approaches were explored:

- Class weighting
- SMOTE
- XGBoost `scale_pos_weight`

The experiments demonstrated the trade-off between precision and recall when detecting minority-class failures.

### 5. Hyperparameter Tuning

RandomizedSearchCV with StratifiedKFold cross-validation was used to tune the XGBoost model.

The search optimized parameters such as:

- Learning rate
- Maximum tree depth
- Number of estimators
- Subsample
- Column sampling
- Gamma
- Minimum child weight
- Positive class weight

### 6. Threshold Optimization

Instead of relying on the default classification threshold of 0.5, an optimized threshold was selected using out-of-fold training predictions and the precision-recall curve.

This allowed the classification decision to better balance precision and recall.

### 7. Model Explainability

SHAP (SHapley Additive exPlanations) was used to understand model predictions.

Both global and local explanations were explored.

The analysis highlighted features such as:

- Tool wear
- Torque
- Rotational speed
- Air temperature
- Process temperature

These represent learned model associations, not proven causal relationships.

## Model Performance

The tuned XGBoost model with the optimized classification threshold achieved the following results in the recorded test evaluation:

| Metric | Score |
|---|---:|
| Accuracy | 98% |
| Precision (Failure class) | 0.80 |
| Recall (Failure class) | 0.67 |
| F1-score (Failure class) | 0.73 |

**Confusion Matrix**

Using the standard scikit-learn convention:

| | Predicted No Failure | Predicted Failure |
|---|---:|---:|
| Actual No Failure | 2881 | 17 |
| Actual Failure | 34 | 68 |

The model detected 68 out of 102 failure cases in this evaluation.

**Important validation note:** The test set was consulted during multiple model comparison and experimentation stages. Therefore, these results should be treated as exploratory rather than an untouched, independent estimate of generalization performance.

## Streamlit Application

The trained XGBoost pipeline and optimized threshold were saved and integrated into a Streamlit application.

The application allows users to:

- Enter machine operating parameters
- Select machine type
- Generate failure predictions
- View estimated failure probability
- View the classification result
- Inspect model diagnostics and input parameters

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance handling | Imbalanced-learn |
| Explainability | SHAP |
| Deployment | Streamlit |
| Model serialization | Joblib |
| Development | Jupyter Notebook, VS Code |

## Project Structure

```text
Industrial-Predictive-Maintenance/
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
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Industrial-Predictive-Maintenance
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Limitations

- The AI4I 2020 dataset is a synthetic benchmark dataset, not live industrial sensor data.
- Model predictions are based on patterns learned from this dataset.
- Estimated probabilities have not been independently validated for calibration.
- The test set was reused during experimentation, so an additional untouched evaluation is needed for a stronger generalization claim.
- The application is an ML prototype, not a certified industrial safety or maintenance system.

## Future Improvements

- Evaluate the model on an independent dataset.
- Investigate probability calibration.
- Add real-time sensor data integration.
- Implement model monitoring and prediction logging.
- Explore drift detection.

## Author

Developed as an end-to-end Machine Learning portfolio project to explore predictive maintenance, imbalanced classification, model optimization, explainable AI and deployment.

---

**Disclaimer:** This project is intended for educational and demonstration purposes. Predictions should not be used as the sole basis for industrial maintenance or safety decisions.