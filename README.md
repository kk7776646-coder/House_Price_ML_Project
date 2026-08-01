# 🏠 House Price Prediction ML Project

An end-to-end Machine Learning project for predicting house prices using Python and Scikit-learn.

## 📌 Project Overview

This project follows a complete Machine Learning workflow, including data preprocessing, exploratory data analysis, missing value handling, categorical feature encoding, model training, hyperparameter tuning, and deployment.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Gradient Boosting Regressor
* GridSearchCV
* Joblib
* Streamlit
* JupyterLab

## 🔄 Project Workflow

**Data → Preprocessing → EDA → Feature Engineering → Model Training → Hyperparameter Tuning → Evaluation → Deployment**

### 1. Data Preprocessing

* Handled missing values
* Converted categorical features into numerical format
* Prepared features for Machine Learning

### 2. Exploratory Data Analysis

* Analyzed numerical and categorical features
* Visualized distributions and relationships
* Investigated missing values and outliers

### 3. Model Training

The project uses **Gradient Boosting Regressor** for house price prediction.

### 4. Hyperparameter Tuning

Used **GridSearchCV** to find better model parameters and improve performance.

### 5. Model Saving

The trained model was saved using **Joblib** for later use.

### 6. Deployment

The Machine Learning model is integrated with a **Streamlit web application** for making house price predictions.

## 🎯 Objective

The main objective is to build a complete end-to-end Machine Learning project that can predict house prices based on property features.

## 👨‍💻 Author

**Kalamuddin Khan**

🏠 House Price Prediction Pipeline

```
                            HOUSE PRICE PREDICTION
                                      │
     ┌────────────────────────────────┼────────────────────────────────┐
     │                                │                                │
  Dataset                       Preprocessing                     Modeling
     │                                │                                │
     ├── Train.csv                    ├── Missing Values              ├── Train/Test Split
     ├── Test.csv                     ├── Data Cleaning               ├── Model Selection
     ├── 80 Features                  ├── Feature Engineering         ├── Training
     └── SalePrice (Target)           ├── Encoding                    └── Evaluation
                                      ├── Scaling
                                      └── Feature Selection
     │                                │                                │
     └────────────────────────────────┼────────────────────────────────┘
                                      │
                                 Deployment
                                      │
                         ├── model.pkl
                         ├── scaler.pkl
                         ├── Streamlit
                         ├── User Input
                         ├── Prediction
                         └── Output
```


