# Research Methods Coding Portfolio

This repository contains my Python coding projects and activities from my Research Methods course. Throughout the course, I used Python to work with chemical data, analyze missing values, build machine learning models, and evaluate model predictions.

## Chapter 5 - PubChem

I learned how to retrieve chemical information from PubChem using Python. I worked with molecules such as caffeine, aspirin, nicotine, and theobromine and explored properties such as molecular formulas, molecular weights, and chemical structures.

## Chapter 6 - Molecule Explorer

I used Python and RDKit to explore and compare molecular structures. I learned how molecular properties and structural information can be analyzed computationally.

## Chapter 7 - Alkane Analysis

I analyzed an alkane dataset containing chemical properties such as molecular weight, boiling point, viscosity, thermal conductivity, heat capacity, carbon count, and branching.

I studied missing data using list-wise deletion and imputation. I also tested Log-Linear Regression, KNN, Random Forest, and Ensemble models and compared their performance using MAE. Correlation matrices and bias graphs were created to help analyze the results.

## Chapter 7 - Making Data Whole

I used the Titanic dataset to study missing age data. I tested mean imputation and machine learning methods such as KNN and Random Forest to estimate missing ages.

## Chapter 8 - Residual Analysis

I compared four prediction datasets using MAE, MSE, R2, predicted vs. actual plots, and residual plots. This helped me understand prediction accuracy, model bias, overprediction, and underprediction.

## Chapter 8 - Error Metrics

I calculated residuals, MAE, MSE, and R2 manually using NumPy. I then verified my calculations using scikit-learn and created predicted vs. actual and residual plots.

## What I Learned

These projects helped me understand how Python can be used to analyze scientific data. I learned how to handle missing data, compare machine learning models, evaluate prediction errors, create graphs, and use computational tools to study chemical and biological data.
## Chapter 9: Decision Tree Classifier

In this project, I used a Decision Tree Classifier to predict whether molecules are water soluble. The model used molecular weight, hydrogen bond donors, and hydrogen bond acceptors as features.

I trained the model using scikit-learn and created a visual decision tree showing how the model makes its predictions. I also tested how changing the maximum tree depth affects the model and created another model that uses only hydrogen bond donors.

### What I Learned
This project helped me understand how decision trees make predictions by splitting data based on different features. I also learned that making a tree more complex does not always make it better because deeper trees can overfit the data.

# Chapter 11: Gradient Descent

## Overview
In this project, I used Python to find the best-fitting line through noisy data.

## What I Did
- Generated data using y = 2x + 5.
- Added random noise to the data.
- Used linear regression to find the best-fit line.
- Calculated Mean Squared Error (MSE).
- Created two graphs to show the results.

## Results
- True slope: 2.0
- Predicted slope: 2.0905
- True intercept: 5.0
- Predicted intercept: 4.7301
- MSE: 2.2428

## Regression Graph
![Regression Graph](regression_scatterplot.png)

## Loss Landscape
![Loss Landscape](loss_landscape.png)

## Conclusion
The model found a line close to the original equation. The small differences were caused by random noise. This project helped me understand how models find the best fit and reduce errors.
