# What I did:
# I used the Titanic dataset to study missing age data.
# I tested different ways to estimate missing ages.

# What I learned:
# I learned that missing data can affect results and that Python
# can use patterns in other data to estimate missing values.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# LOAD TITANIC DATASET
# -------------------------------------------------

df = sns.load_dataset("titanic")

print("\nTITANIC DATASET")
print("================================")

print(df.head(10))

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
print(list(df.columns))
# -------------------------------------------------
# CHECK MISSING AGE DATA
# -------------------------------------------------

print("\n\nMISSING AGE DATA")
print("================================")

missing_ages = df["age"].isnull().sum()

print("Missing ages:", missing_ages)

print("\nMissing values in all columns:")
print(df.isnull().sum())
# -------------------------------------------------
# MEAN AGE
# -------------------------------------------------

print("\n\nMEAN AGE")
print("================================")

mean_age = df["age"].mean()

print(
    "Mean age:",
    round(mean_age, 2)
)
# -------------------------------------------------
# MEAN IMPUTATION
# -------------------------------------------------

mean_df = df.copy()

mean_df["age"] = mean_df["age"].fillna(
    mean_age
)

print("\n\nMEAN IMPUTATION")
print("================================")

print(
    "Missing ages before:",
    df["age"].isnull().sum()
)

print(
    "Missing ages after:",
    mean_df["age"].isnull().sum()
)

print(
    "Mean used to fill missing ages:",
    round(mean_age, 2)
)
# -------------------------------------------------
# CORRELATION MATRIX
# -------------------------------------------------

numeric_df = df.select_dtypes(
    include=np.number
)

correlation = numeric_df.corr()

print("\n\nCORRELATION MATRIX")
print("================================")

print(
    correlation.round(3)
)

print("\nCORRELATIONS WITH AGE")
print("================================")

age_correlations = (
    correlation["age"]
    .sort_values(ascending=False)
)

print(age_correlations)
plt.figure(figsize=(10, 8))

plt.imshow(
    correlation,
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title(
    "Titanic Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    "Titanic_Correlation_Matrix.png"
)

plt.close()

print(
    "\nSaved: Titanic_Correlation_Matrix.png"
)
# -------------------------------------------------
# PREPARE DATA FOR MACHINE LEARNING
# -------------------------------------------------

features = [
    "pclass",
    "sibsp",
    "parch",
    "fare"
]

known_age = df.dropna(
    subset=["age"]
).copy()

X = known_age[features].copy()
y = known_age["age"].copy()

X = X.fillna(
    X.mean()
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\n\nMACHINE LEARNING DATA")
print("================================")

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)
# -------------------------------------------------
# KNN MODEL
# -------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(
    X_train_scaled,
    y_train
)

knn_predictions = knn.predict(
    X_test_scaled
)

knn_mae = mean_absolute_error(
    y_test,
    knn_predictions
)

print("\n\nKNN RESULTS")
print("================================")

print(
    "KNN MAE:",
    round(knn_mae, 2)
)
# -------------------------------------------------
# KNN PREDICTION GRAPH
# -------------------------------------------------

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    knn_predictions
)

minimum = min(
    y_test.min(),
    knn_predictions.min()
)

maximum = max(
    y_test.max(),
    knn_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)

plt.xlabel(
    "Actual Age"
)

plt.ylabel(
    "KNN Predicted Age"
)

plt.title(
    "Actual Age vs KNN Predicted Age"
)

plt.tight_layout()

plt.savefig(
    "KNN_Age_Predictions.png"
)

plt.close()

print(
    "Saved: KNN_Age_Predictions.png"
)
# -------------------------------------------------
# KNN IMPUTATION
# -------------------------------------------------

knn_df = df.copy()

missing_age_rows = knn_df["age"].isnull()

missing_X = knn_df.loc[
    missing_age_rows,
    features
].copy()

missing_X = missing_X.fillna(
    X.mean()
)

missing_X_scaled = scaler.transform(
    missing_X
)

predicted_missing_ages = knn.predict(
    missing_X_scaled
)

knn_df.loc[
    missing_age_rows,
    "age"
] = predicted_missing_ages

print("\n\nKNN IMPUTATION")
print("================================")

print(
    "Missing ages after KNN:",
    knn_df["age"].isnull().sum()
)

print(
    "Average known age before imputation:",
    round(df["age"].mean(), 2)
)

print(
    "Average age after KNN imputation:",
    round(knn_df["age"].mean(), 2)
)
# -------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_predictions = rf.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

print("\n\nRANDOM FOREST RESULTS")
print("================================")

print(
    "Random Forest MAE:",
    round(rf_mae, 2)
)
# -------------------------------------------------
# RANDOM FOREST GRAPH
# -------------------------------------------------

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    rf_predictions
)

minimum = min(
    y_test.min(),
    rf_predictions.min()
)

maximum = max(
    y_test.max(),
    rf_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)

plt.xlabel(
    "Actual Age"
)

plt.ylabel(
    "Random Forest Predicted Age"
)

plt.title(
    "Actual Age vs Random Forest Predicted Age"
)

plt.tight_layout()

plt.savefig(
    "Random_Forest_Age_Predictions.png"
)

plt.close()

print(
    "Saved: Random_Forest_Age_Predictions.png"
)
# -------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------

print("\n\nMODEL COMPARISON")
print("================================")

print(
    "KNN MAE:",
    round(knn_mae, 2)
)

print(
    "Random Forest MAE:",
    round(rf_mae, 2)
)

if knn_mae < rf_mae:

    print(
        "KNN had the lower MAE."
    )

elif rf_mae < knn_mae:

    print(
        "Random Forest had the lower MAE."
    )

else:

    print(
        "The models had the same MAE."
    )
    # -------------------------------------------------
    # COMPLETE
    # -------------------------------------------------

    print("\n\nMAKING DATA WHOLE COMPLETE")
    print("================================")

    print(
        "Missing ages originally:",
        missing_ages
    )

    print(
        "Missing ages after mean imputation:",
        mean_df["age"].isnull().sum()
    )

    print(
        "Missing ages after KNN imputation:",
        knn_df["age"].isnull().sum()
    )

    print(
        "KNN MAE:",
        round(knn_mae, 2)
    )

    print(
        "Random Forest MAE:",
        round(rf_mae, 2)
    )

    print(
        "\nCheck the project folder for the saved graphs."
    )