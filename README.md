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


# AR Agonist Master Database

## Project Overview
In this project, I used Python to organize and analyze chemical data from the EPA CompTox Chemicals Dashboard. The dataset focuses on androgen receptor (AR) agonist activity.

The dataset contains 1,830 chemicals and 14 columns. The goal is to identify chemical properties, examine biological activity, and create a searchable database.

## Dataset Columns and Examples

### 1. DTXSID
A unique identification number assigned to each chemical by the EPA.

Example: `DTXSID4020119` identifies Azathioprine.

### 2. Preferred Name
The main name used to identify a chemical.

Example: Azathioprine.

### 3. CASRN
A unique chemical registry number used to identify chemicals.

Example: `446-86-6` is the CASRN for Azathioprine.

### 4. Molecular Formula
Shows the elements present in a chemical and the number of atoms of each element.

Example: `C9H7N7O2S` contains 9 carbon atoms, 7 hydrogen atoms, 7 nitrogen atoms, 2 oxygen atoms, and 1 sulfur atom.

### 5. Monoisotopic Mass
The exact molecular mass calculated using the most abundant isotopes of each element.

Example: Azathioprine has a monoisotopic mass of approximately 277.038194 Da.

### 6. ToxCast Active
The number of ToxCast assays where a chemical was classified as active.

Example: Azathioprine was active in 137 assays.

### 7. ToxCast Total
The total number of ToxCast assays evaluated for a chemical.

Example: Azathioprine had 674 evaluated assay results.

### 8. % ToxCast Active
The percentage of evaluated assays where the chemical was classified as active.

Example: Azathioprine was active in approximately 20% of its evaluated assays.

### 9. Hit Call
Indicates whether a chemical was classified as Active or Inactive in the AR agonist assay.

Example: Azathioprine was classified as Inactive.

### 10. Continuous Hit Call
A numerical score describing the chemical's activity in the assay.

Example: Azathioprine has a Continuous Hit Call of 0.0.

### 11. Top
The estimated upper response level from a fitted concentration-response curve.

Example: Azathioprine has a Top value of approximately -0.000224.

### 12. Scaled Top
The upper response value after applying the assay's scaling method.

Example: Azathioprine has a Scaled Top value of approximately 0.000010.

### 13. AC50
The estimated concentration associated with half of the fitted maximum response.

Example: Azathioprine has a reported AC50 of 49.75. The concentration units should be verified using the EPA assay documentation.

### 14. LOGAC50
The base-10 logarithm of the AC50 value, used to compare chemicals with different concentration values.

Example: Azathioprine has a LOGAC50 of approximately 1.697.

## Dataset Results

| Category | Result |
|----------|--------|
| Total Chemicals | 1,830 |
| Total Columns | 14 |
| Active Chemicals | 87 |
| Inactive Chemicals | 1,743 |
| Duplicate DTXSIDs | 0 |
| Missing Molecular Formulas | 53 |

## Python Libraries Used

- Pandas: Importing, cleaning, and analyzing data.
- SQLite: Creating a searchable chemical database.
- Matplotlib: Generating graphs and visualizations.

## What I Learned

This project helped me understand how chemical data is organized and how biological activity is measured. I learned how to identify missing values, remove duplicate chemical records, create databases, and generate graphs using Python.

## Limitations

An Active result in this assay does not automatically prove that a chemical activates androgen receptors in humans. Additional testing is needed.

The exact interpretation of Continuous Hit Call, Top, Scaled Top, and AC50 units depends on the assay documentation.

