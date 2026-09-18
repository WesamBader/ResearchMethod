# ResearchMethod
Projects for Foundations of Machine Learning for Chemistry
## MoleculeExplorer
### What My Code Does
This program takes a SMILES string and uses RDKit to calculate molecular properties such as exact molecular weight, hydrogen bond donors, TPSA, hydrogen bond acceptors, and rotatable bonds.
### What I Learned
I learned how to use RDKit to turn a SMILES string into a molecule object and calculate different molecular properties. I also learned how these properties can be used to describe and compare molecules.
# What I did: Chapter 6
I made a program that compares different molecules and their properties
# What I learned:
I learned how to use Python and RDKit to find and compare chemical properties 
# Chapter 7: Making Data Whole

## Overview

In Chapter 7, I learned how missing data can affect scientific results and how different methods can be used to fill in missing values. I used Python to analyze an alkane dataset and the Titanic dataset.

## Alkane Data Analysis

I analyzed a dataset containing 250 alkanes and several chemical properties. I found that thermal conductivity had the most missing data at 22%.

I tested list-wise deletion, which removed every row with missing data. This caused 60% of the dataset to be lost, showing how deleting missing data can greatly reduce the amount of information available.

I also used mean imputation to fill missing numeric values and created a correlation matrix to compare chemical properties.

## Machine Learning Models

I tested several models to predict missing chemical properties:

- Log-Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Ensemble

I compared the models using Mean Absolute Error (MAE). A lower MAE means the predictions were closer to the actual values.

Log-Linear Regression worked best for viscosity and thermal conductivity, while Random Forest performed best for heat capacity.

I also created bias graphs to see how prediction errors changed based on carbon count and branching.

## Making Data Whole

For the second part, I used the Titanic dataset to study missing age data. I first used mean imputation to fill missing ages.

I then created KNN and Random Forest models that used other passenger information to predict age. I compared the models using MAE and created graphs comparing actual ages with predicted ages.

## What I Learned

I learned that there are different ways to handle missing data and that each method has advantages and limitations. Deleting missing data can cause information loss, while machine learning can use patterns in other data to estimate missing values. I also learned how MAE, correlations, and graphs can be used to evaluate model performance.
