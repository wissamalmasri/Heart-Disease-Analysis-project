Heart Disease Analysis Project

Overview

This project analyzes a dataset of heart disease patients to perform both supervised and unsupervised learning tasks. The main goals are:
1. Predict the presence of heart disease using classification models.
2. Identify patterns and clusters in the dataset using clustering techniques.
3. Reduce dimensions for visualization using PCA (Principal Component Analysis).

Features
Dataset Exploration: Basic descriptive statistics and data visualizations.

Preprocessing:
Handling class imbalance using SMOTE.
Scaling numeric features.
Encoding categorical variables.

Classification Models:
Logistic Regression
Support Vector Machine (SVM)
Multilayer Perceptron (MLP)
Gradient Boosting Classifier

Evaluation Metrics:
Precision, Recall, F1-Score, Accuracy
ROC Curve and AUC (Area Under Curve)

Clustering Models:
K-Means Clustering
Hierarchical Clustering
DBSCAN Clustering

Dimensionality Reduction:
PCA for visualization

Dataset
The dataset used is a CSV file named heart.csv, containing information about heart disease patients.

Key Columns
target: The target variable indicating the presence or absence of heart disease.
Various clinical and demographic features, including age, cholesterol levels, and others.

Setup
Python 3.x
Libraries: pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn, scipy
Environment: Jupyter Notebook

Library Versions
This project was tested with the following library versions:
Python: 3.9
pandas: 1.3.3
matplotlib: 3.4.3
seaborn: 0.11.2
scikit-learn: 1.0.1
imbalanced-learn: 0.8.0
scipy: 1.7.1

Installation
Clone the repository or download the code files.
Install required Python libraries:
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn scipy
Place the heart.csv dataset in the working directory.
Open the heart_disease_analysis script in Jupyter Notebook and run the cells sequentially.

Usage
Open the project in Jupyter Notebook.
Run the Python script to load and preprocess the dataset.
Perform classification tasks to predict heart disease.
Generate visualizations, including:
Feature importance
Clustering results
PCA scatterplots
ROC curves for classifiers

Outputs
Classification Metrics:
Precision, Recall, F1-Score, Accuracy
Confusion Matrices
ROC Curves and AUC

Clustering Results:
Silhouette Scores
Cluster Scatterplots
Dendrogram (Hierarchical Clustering)
Dimensionality Reduction:
PCA Scatterplot with explained variance ratio

Key Functions
evaluate_model: A helper function to compute and display model metrics and confusion matrix.
PCA implementation for dimensionality reduction.
Clustering methods with visualizations.
Future Work
Enhance feature engineering for better predictions.
Include hyperparameter tuning for models.
Explore additional unsupervised learning techniques.

Acknowledgments
Dataset source: Publicly available heart disease datasets.
Libraries: scikit-learn, pandas, matplotlib, seaborn, imbalanced-learn.