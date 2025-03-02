# Default Prediction Model

## Overview
This project develops a machine learning model to predict whether an individual will default on a loan. Given the highly imbalanced nature of the dataset, special techniques were employed to improve prediction accuracy while emphasizing recall over precision.

## Data Preprocessing

### 1. Handling Missing Values
- Columns with more than **20% missing values** were dropped.
- For remaining missing values:
  - **Numerical features** were filled using the **median**.
  - **Categorical features** were filled using the **mode**.

### 2. Handling Outliers
- Outliers were detected and **capped** to prevent extreme values from negatively impacting the model.

### 3. Encoding Categorical Variables
- Categorical features were transformed using **One-Hot Encoding** to be suitable for machine learning algorithms.

## Model Development

### 1. Train-Test Split
- The dataset was split into **train and test sets** using **Stratified Sampling** to maintain the proportion of defaults across both sets.

### 2. Handling Imbalance
- The dataset exhibited **class imbalance** with very few cases of default. To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training set to balance the classes.

### 3. Feature Selection
- **SelectFromModel** was used to select the most important features before training models.

### 4. Model Training and Evaluation
Four machine learning models were tested:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **XGBoost Classifier**

Hyperparameter tuning was performed using **Grid Search** with **Stratified K-Fold Cross-Validation**.

### 5. Evaluation Metric
- The model was evaluated using **F2 Score**, which places more weight on Recall than Precision. Since the goal is to **identify as many defaulters as possible** (minimizing false negatives), recall is prioritized over precision.

## Results
- **XGBoost performed best**, achieving:
  - **Recall = 0.89**
  - **Precision = 0.65**
- This balance ensures that most defaulters are correctly identified while maintaining a reasonable precision.

## Model Interpretation
- **SHAP values** were used to interpret model predictions.
- Key findings:
  - **`credit_type_EQUI`** was the most important feature.
  - Higher **income** → **Lower default probability**.
  - Higher **loan amount** → **Higher default probability**.
  - Higher **LTV (Loan-to-Value ratio)** → **Higher default probability**.
  - **`business_or_commercial_b/c`** had **no predictive power** and could be dropped.

## Conclusion
This model effectively predicts loan default using a balanced approach. The emphasis on **recall through F2 Score** ensures that we minimize missed defaulters, making it valuable for financial institutions aiming to mitigate credit risk.

