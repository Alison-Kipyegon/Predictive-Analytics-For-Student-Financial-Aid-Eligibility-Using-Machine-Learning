# Predictive Analytics for Student Financial Aid Eligibility using Machine Learning

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Dataset](#dataset)  
6. [Machine Learning Models](#machine-learning-models)  
7. [Evaluation](#evaluation)  
8. [Project Structure](#project-structure)  
9. [Contributing](#contributing)  
10. [References](#references)  

---

## Project Overview
The **Financial Aid Prediction System** is a machine learning–based application designed to help higher education institutions predict which students are likely to need financial assistance. The system uses historical academic, demographic, and socioeconomic data to make predictions, improving the efficiency and fairness of financial aid allocation.  

Key objectives:  
- Predict students who need financial assistance using interpretable ML models.  
- Enhance decision-making for financial aid officers.  
- Ensure fairness, transparency, and actionable insights in financial aid allocation.  

---

## Features
- Predictive modeling using **Decision Tree** and **Logistic Regression** classifiers.  
- Preprocesses numerical and categorical student data.  
- Generates **confusion matrix** and performance metrics for model evaluation.  
- Provides interpretable outputs for actionable decision-making.  
- Simple interface for entering student data and receiving predictions.  

---

## Installation
**Requirements:**  
- Python 3.10+  
- Google Colab (optional for experiments)  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`  

**Steps:**  
1. Clone the repository:  
```bash git clone https://github.com/Alison-Kipyegon/Predictive-Analytics-For-Student-Financial-Aid-Eligibility-Using-Machine-Learning.git bash```

---
cd code
---
## Usage

Load the dataset financial_aid_dataset.csv.

Preprocess data (handle missing values, encode categorical features, scale numerical features).

Train the machine learning models (Decision Tree and Logistic Regression).

Evaluate models with metrics such as accuracy, precision, recall, and confusion matrix.

Predict new students’ financial aid eligibility.
---
## Dataset

The dataset financial_aid_dataset.csv contains student records with:

Academic data: GPA, grades, test scores

Demographic data: Age, gender, region

Socioeconomic data: Family income, number of dependents, guardian occupation

Data is preprocessed to handle missing values, encode categorical variables, and normalize numerical features.

---

## Machine Learning Models

## Decision Tree Classifier:

Interpretable rules for predictions.

Suitable for medium-sized datasets.

## Logistic Regression:

Binary classification (eligible vs. not eligible).

Provides probability estimates for eligibility.

---
## Evaluation

Confusion matrix visualizes true positives, false positives, true negatives, and false negatives.

Metrics: Accuracy, Precision, Recall, F1-score.

Cross-validation ensures model generalizability and reduces overfitting risk.

## Project Structure
code/
│
├── eda
  ├──projecteda.py
├── dataset
  ├── financial_aid_dataset.csv  
├──database
  ├──financial_aid
  ├──financial_aid.sqbpro
  ├──setup_db.py
├── models
  ├──dt_model.pkl
  ├──lr_model.pkl
  ├──metrics
  ├──scaler.pkl
├── static
  ├──css
    ├──style.css
  ├──plots
    ├──shap_1001
    ├──shap_1002
    ├──shap_1003
    ├──shap_1004
    ├──shap_1007
    ├──shap_1028
    ├──shap_1051
    ├──shap_1181
    ├──shap_6799
  ├──qrcodes
    ├──alison.kipyegon@strathmore.edu
    ├──chelimoalison0@gmail.com
    ├──kippycheli@gmail.com
  ├──templates
    ├──2fa
    ├──admin_base
    ├──base
    ├──dashboard
    ├──dashboard_admin
    ├──explanation
    ├──landing
    ├──login
    ├──model_admin
    ├──predictions
    ├──settings
    ├──show_qr
    ├──signiup
    ├──students_admin
    ├──upload
  ├──app.py
  ├──generate_qr.py
└── README.md                  
