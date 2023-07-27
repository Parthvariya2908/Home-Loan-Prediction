# Loan Approval and Loan Amount Prediction

## Description

This Python machine learning project aims to predict loan approval status and loan amounts based on user inputs using various machine learning models.

## Requirements

- Python 3.x
- Numpy
- Pandas
- Seaborn
- Scikit-learn
- Matplotlib
- Tkinter
- PIL

## Data Preprocessing and Model Training

1. Load and preprocess the dataset.
2. Convert categorical columns to numerical values.
3. Split the data into training and testing sets.
4. Train the following machine learning models:
   - Support Vector Machine (SVM) with a linear kernel for loan approval status prediction.
   - Multiple Linear Regression for loan approval status prediction.
   - Random Forest Regression for loan approval status prediction.
   - Decision Tree Regression for loan approval status prediction.
   - Lasso Regression for loan approval status prediction.
   - Multiple Linear Regression for loan amount prediction.
   - Random Forest Regression for loan amount prediction.
   - Decision Tree Regression for loan amount prediction.
   - Lasso Regression for loan amount prediction.

## Graph Analysis

Graphs are generated using Seaborn to visualize the performance of the multiple linear regression and random forest regression models.

## Prediction Function

The prediction function takes user inputs (e.g., gender, marital status, education, income, credit history, etc.) and uses the trained models to predict loan approval status and loan amounts.

## Graphical User Interface (GUI)

The project includes a GUI built using Tkinter to facilitate user interactions. Users can input their details, and the GUI displays the predicted loan approval status and loan amounts.

## How to Use

1. Install the required libraries mentioned in the Requirements section.
2. Run the Python script "voting_system.py."
3. The GUI window will appear, where you can enter your details and click the "Save" button to get predictions.
4. Click the "Analyze" button to open a new window showing graphs related to the performance of certain regression models.


