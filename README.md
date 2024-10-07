# Credit Card Fraud Detection

## Overview
This repository contains a machine learning project focused on detecting fraudulent credit card transactions. Fraud detection is a critical challenge faced by financial institutions, as it helps in identifying and preventing potential frauds to minimize financial losses. The project uses various classification algorithms and techniques to handle the imbalanced nature of the dataset, where fraudulent transactions are much rarer than legitimate ones.

## Purpose
The purpose of this project is to build an efficient machine learning model capable of accurately classifying transactions as either fraudulent or non-fraudulent. The model is evaluated using appropriate metrics that consider the imbalanced nature of the dataset.

## Application
Fraud detection models like the one developed in this project are applicable in:
- Banking and financial institutions for monitoring transactions
- E-commerce platforms to prevent payment fraud
- Insurance companies to detect fraudulent claims

## Topics Covered

### 1. Data Preprocessing
- Handling missing values
- Scaling and normalizing features
- Encoding categorical variables
- Dealing with class imbalance through oversampling techniques like **SMOTE** or **Random Oversampling**

### 2. Feature Selection and Engineering
- Identifying important features using techniques like:
  -**Variance Threshold**    
  - **Mutual Information** for feature relevance
  - **Permutation Importance** (though the library `eli5` needs to be installed for full execution)

### 3. Model Building
- Pipelines were created to streamline the process of data preprocessing and model training.
- Multiple classifiers were trained and compared, including:
  - Random Forest
  - Gradient Boosting (e.g., **XGBoost**)

### 4. Model Evaluation
- **Confusion Matrix**: Used to assess model performance, specifically precision, recall, and accuracy.
- **Imbalanced Data Handling**: Special focus on correctly classifying the minority class (fraudulent transactions).
  
### 5. Handling Overfitting
- Cross-validation was employed to prevent overfitting.
- Feature importance was analyzed using **Permutation Importance**.

## Project Workflow
1. **Data Loading**: The dataset of credit card transactions is loaded, and basic exploratory data analysis is performed.
2. **Preprocessing**: Missing values are handled, and data is scaled and encoded. Special techniques are used to address the imbalanced nature of the dataset.
3. **Modeling**: Multiple machine learning algorithms are applied and compared using performance metrics like **accuracy**, **precision**, **recall**, and **AUC** (Area Under Curve).
4. **Evaluation**: Confusion matrices and other evaluation metrics are generated to assess the model's ability to detect fraud.
5. **Improving Model Performance**: Techniques such as cross-validation and permutation importance are utilized to further optimize model performance and reduce overfitting.

## Dependencies
- Python 3.x
- Scikit-learn
- Pandas
- Matplotlib
- NumPy
- Imbalanced-learn
- XGBoost
- eli5 (for permutation importance)
