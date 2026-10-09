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


## Chapter 11: Gradient Descent

In this project, I used Python to find the best-fitting line through a set of data. I created random data using the equation y = 2x + 5 and added noise to make it more realistic.

I used linear regression to find the best-fit line and calculated the Mean Squared Error (MSE) to measure how accurate the predictions were. I also created graphs to show the results.

### What I Learned
This project helped me understand how models find the best-fitting line and how errors are measured. I also learned how noise can affect the accuracy of predictions.


## Chapter 12: Evolutionary Algorithms

In this project, I used Python to create an evolutionary algorithm that changes random letters into the phrase "METHINKS IT IS LIKE A WEASEL."

The program used mutation and selection to improve the phrase over multiple generations. I also created a graph showing how the fitness score improved until it reached 28/28.

### What I Learned
This project helped me understand how evolutionary algorithms use mutation and selection to improve results. I also learned how these algorithms can find better solutions without knowing the correct answer at the start.

# AR Antagonist Viability Database

## Overview
In this project, I used Python to clean and organize chemical data related to androgen receptor (AR) testing. I used pandas, SQLite, and Matplotlib to create a database and graphs.

## What I Did
- Analyzed 1,835 chemicals.
- Checked for missing values and duplicates.
- Created a searchable database.
- Compared active and inactive chemicals.
- Created graphs showing the results.


## Dataset Columns

1. **DTXSID:** A unique identification number assigned to each chemical by the EPA CompTox database. It helps researchers identify and track chemicals.
2. **Preferred Name:** The name used to identify the chemical in the database.
3. **CASRN:** A unique registry number assigned to a chemical by the Chemical Abstracts Service. It helps identify chemicals that may have multiple names.
4. **Molecular Formula:** Shows which elements are present in a chemical and how many atoms of each element it contains.
5. **Monoisotopic Mass:** The mass of a molecule calculated using the exact masses of its most abundant isotopes.
6. **ToxCast Active:** The number of ToxCast assays in which the chemical was classified as active. This helps show how often the chemical produces measurable biological activity.
7. **ToxCast Total:** The total number of ToxCast assays associated with the chemical. This includes both active and inactive results.
8. **% ToxCast Active:** The percentage of ToxCast assays in which the chemical was active. It is calculated by dividing the number of active assays by the total number of assays and multiplying by 100.
9. **Hit Call:** Classifies the chemical as Active or Inactive in this specific viability assay. Active means the chemical met the assay's activity criteria, while Inactive means it did not.
10. **Continuous Hit Call:** A numerical score that provides more information about assay activity than a simple Active or Inactive classification. The exact meaning depends on the assay's scoring method.
11. **Top:** A value estimated from the fitted concentration-response curve. It represents the upper response level predicted by the model.
12. **Scaled Top:** The fitted Top response after adjustment using the assay's scaling method. This helps express responses on a standardized scale.
13. **AC50:** The estimated concentration at which a chemical produces half of its fitted maximum response. Lower AC50 values can indicate greater potency, but they do not automatically mean greater toxicity.
14. **LOGAC50:** The base-10 logarithm of the AC50 value. Using a logarithmic scale makes it easier to compare concentrations that differ by large amounts.
## Important Note
This dataset measures viability-related assay activity. An Active result does not automatically mean that a chemical blocks the androgen receptor or is harmful to humans.


## Results
- Total chemicals: 1,835
- Active: 552
- Inactive: 1,283
- Duplicates: 0

## What I Learned
This project helped me understand how to clean scientific data, create a database, and use Python to analyze chemical assay results.
