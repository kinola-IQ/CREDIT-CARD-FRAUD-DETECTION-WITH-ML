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
- **Downcasting** for the purpose of memory optimization
- Dealing with class imbalance through oversampling techniques like **SMOTETomek** or **Random Oversampling**

### 2. Feature Selection and Engineering
- Identifying important features using techniques like: 
  - **Feature importance**
  - **Permutation Importance** 

### 3. Model Building
- Multiple classifiers were trained and compared, including:
  - Random Forest classifer
  - Gradient Boosting classifier

### 4. Model Evaluation
- **Confusion Matrix**: Used to assess model performance, specifically precision, recall, and accuracy.
- **Imbalanced Data Handling**: Special focus on correctly classifying the minority class (fraudulent transactions).
- **loss funcions** : Matthews correlation coeficient, average precison, etc

### 5.key Metric
  **Matthews Correlation Coefficient (MCC)**: The MCC achieved was 92.4%, indicating a strong model performance, especially 
 in handling the imbalanced nature of the dataset
  
### 5. Handling Overfitting
- initially **sequential feature selection (SFS)** was employed but proved to be too computationally expensive
- The Model's embedded **feature importance** was used and then compared to the **Permutation importance**
- Feature importance was analyzed using **Permutation Importance**.

## Project Workflow
1. **Data Loading**: The dataset of credit card transactions is loaded, and basic exploratory data analysis is performed.
2. **Preprocessing**: Data was Downcasted after inspection of the variance of all features. Special techniques are used to address the imbalanced nature of the dataset.
3. **Modeling**: Multiple machine learning algorithms are applied and compared using performance metrics like **accuracy**, **precision**, **recall**, and **AUC** (Area Under Curve).
4. **Evaluation**: Confusion matrices and other evaluation metrics are generated to assess the model's ability to detect fraud.
5. **Improving Model Performance**: Techniques such as cross-validation,hyperparameter tuning with optuna and permutation importance are utilized to further optimize model performance and reduce overfitting.

## Dependencies
- Python 3.x
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Imbalanced-learn
- Optuna
- Warnings
