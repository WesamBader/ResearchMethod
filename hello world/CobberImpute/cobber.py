# What I did:
# I made a program that analyzes missing data and tests ways to fill it in.

# What I learned:
# I learned that deleting missing data can cause bias and that machine
# learning can be used to estimate missing values.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

file_name = "alkane_dataset.csv"

df = pd.read_csv(file_name)

print("\nORIGINAL DATA")
print("================================")
print(df.head(10))

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
print(list(df.columns))


# -------------------------------------------------
# MISSING DATA REPORT
# -------------------------------------------------

print("\n\nMISSING DATA REPORT")
print("================================")

missing_count = df.isnull().sum()
missing_percent = df.isnull().mean() * 100

missing_report = pd.DataFrame({
    "Missing Values": missing_count,
    "Percent Missing": missing_percent
})

print(missing_report)

most_missing = missing_count.idxmax()

print("\nProperty with the most missing data:")
print(most_missing)


# -------------------------------------------------
# LIST-WISE DELETION
# -------------------------------------------------

print("\n\nLIST-WISE DELETION")
print("================================")

original_rows = len(df)

clean_df = df.dropna()

remaining_rows = len(clean_df)
lost_rows = original_rows - remaining_rows

percent_lost = (lost_rows / original_rows) * 100

print("Original rows:", original_rows)
print("Rows remaining:", remaining_rows)
print("Rows lost:", lost_rows)
print("Percent of dataset lost:", round(percent_lost, 2), "%")


# -------------------------------------------------
# FIND COLUMN NAMES
# -------------------------------------------------

def find_column(words):
    for column in df.columns:
        name = column.lower()

        for word in words:
            if word in name:
                return column

    return None


carbon_col = find_column(["carbon"])
branch_col = find_column(["branch"])
viscosity_col = find_column(["viscos"])
heat_col = find_column(["heat capacity", "molar heat", "heat_capacity"])
thermal_col = find_column(["thermal", "conductivity"])

print("\nDetected columns:")
print("Carbons:", carbon_col)
print("Branches:", branch_col)
print("Viscosity:", viscosity_col)
print("Heat Capacity:", heat_col)
print("Thermal Conductivity:", thermal_col)


# -------------------------------------------------
# BIAS AFTER LIST-WISE DELETION
# -------------------------------------------------

def make_bias_plot(column, title):

    if column is None:
        return

    before = df[column].value_counts(normalize=True).sort_index() * 100
    after = clean_df[column].value_counts(normalize=True).sort_index() * 100

    comparison = pd.concat(
        [before.rename("Before"), after.rename("After")],
        axis=1
    ).fillna(0)

    comparison["Percent Change"] = (
        (comparison["After"] - comparison["Before"])
        / comparison["Before"]
        * 100
    )

    print("\n", title)
    print(comparison)

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison.index.astype(str),
        comparison["Percent Change"]
    )

    plt.axhline(0)

    plt.xlabel(column)
    plt.ylabel("Percent Change")
    plt.title(title)

    plt.xticks(rotation=90)

    plt.tight_layout()

    filename = title.replace(" ", "_") + ".png"

    plt.savefig(filename)
    plt.close()

    print("Saved:", filename)


make_bias_plot(
    carbon_col,
    "Bias_After_Deletion_Carbons"
)

make_bias_plot(
    branch_col,
    "Bias_After_Deletion_Branches"
)


# -------------------------------------------------
# MEAN IMPUTATION
# -------------------------------------------------

print("\n\nMEAN IMPUTATION")
print("================================")

mean_df = df.copy()

numeric_columns = mean_df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    mean_value = mean_df[column].mean()

    mean_df[column] = mean_df[column].fillna(mean_value)

print("Missing values after mean imputation:")
print(mean_df.isnull().sum())


# -------------------------------------------------
# CORRELATION MATRIX
# -------------------------------------------------

print("\n\nCORRELATION MATRIX")
print("================================")

correlation = mean_df[numeric_columns].corr()

print(correlation.round(3))

plt.figure(figsize=(10, 8))

plt.imshow(
    correlation,
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("Correlation_Matrix.png")
plt.close()

print("\nSaved: Correlation_Matrix.png")


# -------------------------------------------------
# CORRELATIONS WITH VISCOSITY
# -------------------------------------------------

if viscosity_col is not None:

    print("\n\nVISCOSITY CORRELATIONS")
    print("================================")

    viscosity_correlations = (
        correlation[viscosity_col]
        .sort_values(ascending=False)
    )

    print(viscosity_correlations)

    if carbon_col:
        print(
            "\nCarbons vs viscosity:",
            round(correlation.loc[carbon_col, viscosity_col], 3)
        )

    if branch_col:
        print(
            "Branches vs viscosity:",
            round(correlation.loc[branch_col, viscosity_col], 3)
        )

    if thermal_col:
        print(
            "Viscosity vs thermal conductivity:",
            round(correlation.loc[viscosity_col, thermal_col], 3)
        )


# -------------------------------------------------
# MODEL TESTING FUNCTION
# -------------------------------------------------

def test_models(target_column):

    if target_column is None:
        return

    print("\n\nMODELS FOR:", target_column.upper())
    print("================================")

    available = df.dropna(
        subset=[target_column]
    ).copy()

    features = []

    if carbon_col:
        features.append(carbon_col)

    if branch_col:
        features.append(branch_col)

    if len(features) == 0:
        print("Could not find carbon/branch columns.")
        return

    X = available[features].copy()
    y = available[target_column].copy()

    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    # ---------------------------------------------
    # LOG-LINEAR REGRESSION
    # ---------------------------------------------

    positive = y_train > 0

    log_model = LinearRegression()

    log_model.fit(
        X_train[positive],
        np.log(y_train[positive])
    )

    log_predictions = np.exp(
        log_model.predict(X_test)
    )

    log_mae = mean_absolute_error(
        y_test,
        log_predictions
    )

    print(
        "Log-Linear Regression MAE:",
        round(log_mae, 4)
    )


    # ---------------------------------------------
    # KNN
    # ---------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for k in [3, 5, 10]:

        knn = KNeighborsRegressor(
            n_neighbors=k
        )

        knn.fit(
            X_train_scaled,
            y_train
        )

        predictions = knn.predict(
            X_test_scaled
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        print(
            "KNN k=",
            k,
            "MAE:",
            round(mae, 4)
        )


    # ---------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------

    for depth in [5, 15]:

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=depth,
            random_state=42
        )

        rf.fit(
            X_train,
            y_train
        )

        predictions = rf.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        print(
            "Random Forest depth=",
            depth,
            "MAE:",
            round(mae, 4)
        )


    # ---------------------------------------------
    # ENSEMBLE
    # ---------------------------------------------

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

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_predictions = rf.predict(
        X_test
    )

    ensemble_predictions = (
        log_predictions
        + knn_predictions
        + rf_predictions
    ) / 3

    ensemble_mae = mean_absolute_error(
        y_test,
        ensemble_predictions
    )

    print(
        "Ensemble MAE:",
        round(ensemble_mae, 4)
    )


    # ---------------------------------------------
    # PREDICTION PLOT
    # ---------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_test,
        ensemble_predictions
    )

    minimum = min(
        y_test.min(),
        ensemble_predictions.min()
    )

    maximum = max(
        y_test.max(),
        ensemble_predictions.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    plt.title(
        "Prediction Quality - " + target_column
    )

    plt.tight_layout()

    filename = (
        "Prediction_"
        + target_column.replace(" ", "_")
        + ".png"
    )

    plt.savefig(filename)
    plt.close()

    print("Saved:", filename)
    # ---------------------------------------------
    # MODEL BIAS ANALYSIS
    # ---------------------------------------------

    print("\nBIAS ANALYSIS FOR:", target_column.upper())
    print("================================")

    models = {
        "Log-Linear": log_predictions,
        "KNN": knn_predictions,
        "Random Forest": rf_predictions,
        "Ensemble": ensemble_predictions
    }

    for model_name, model_predictions in models.items():

        relative_error = (
            (model_predictions - y_test.to_numpy())
            / y_test.to_numpy()
        ) * 100

        print("\n", model_name)

        print(
            "Average Relative Error:",
            round(np.mean(relative_error), 2),
            "%"
        )

        print(
            "Largest Absolute Error:",
            round(np.max(np.abs(relative_error)), 2),
            "%"
        )

        catastrophic = np.sum(
            np.abs(relative_error) > 100
        )

        print(
            "Errors greater than ±100%:",
            catastrophic
        )

        # BIAS VS CARBONS

        if carbon_col:

            plt.figure(figsize=(8, 6))

            plt.scatter(
                X_test[carbon_col],
                relative_error
            )

            plt.axhline(0)

            plt.xlabel("Number of Carbons")
            plt.ylabel("Relative Error (%)")

            plt.title(
                model_name
                + " Bias vs Carbons - "
                + target_column
            )

            plt.tight_layout()

            carbon_filename = (
                "Bias_Carbons_"
                + target_column.replace(" ", "_")
                + "_"
                + model_name.replace(" ", "_")
                + ".png"
            )

            plt.savefig(carbon_filename)
            plt.close()

            print("Saved:", carbon_filename)


        # BIAS VS BRANCHING

        if branch_col:

            plt.figure(figsize=(8, 6))

            plt.scatter(
                X_test[branch_col],
                relative_error
            )

            plt.axhline(0)

            plt.xlabel("Branch Number")
            plt.ylabel("Relative Error (%)")

            plt.title(
                model_name
                + " Bias vs Branching - "
                + target_column
            )

            plt.tight_layout()

            branch_filename = (
                "Bias_Branching_"
                + target_column.replace(" ", "_")
                + "_"
                + model_name.replace(" ", "_")
                + ".png"
            )

            plt.savefig(branch_filename)
            plt.close()

            print("Saved:", branch_filename)

# -------------------------------------------------
# RUN MODELS
# -------------------------------------------------

test_models(viscosity_col)
test_models(heat_col)
test_models(thermal_col)


print("\n\nANALYSIS COMPLETE")
print("Check this folder for the saved graphs.")
