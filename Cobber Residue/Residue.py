# What I did:
# I analyzed prediction errors from four different datasets.

# What I learned:
# I learned how MAE, MSE, R2, and residual plots can be used
# to judge how well a model performs.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -------------------------------------------------
# DATASETS
# -------------------------------------------------

datasets = [
    "dataset_low_scatter.csv",
    "dataset_high_scatter.csv",
    "dataset_positive_deviation.csv",
    "dataset_negative_deviation.csv"
]


# -------------------------------------------------
# ANALYZE EACH DATASET
# -------------------------------------------------

def analyze_dataset(file_name):

    print("\n")
    print("=" * 60)
    print("ANALYZING:", file_name)
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(file_name)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(list(df.columns))


    # -------------------------------------------------
    # FIND ACTUAL AND PREDICTED COLUMNS
    # -------------------------------------------------

    actual_col = None
    predicted_col = None

    for column in df.columns:

        name = column.lower()

        if "actual" in name:
            actual_col = column

        if "predict" in name:
            predicted_col = column


    if actual_col is None or predicted_col is None:

        print(
            "Could not automatically find the actual "
            "and predicted columns."
        )

        return None


    actual = df[actual_col]
    predicted = df[predicted_col]


    # -------------------------------------------------
    # RESIDUALS
    # -------------------------------------------------

    residuals = predicted - actual


    # -------------------------------------------------
    # ERROR METRICS
    # -------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )


    print("\nERROR METRICS")
    print("------------------------------")

    print(
        "MAE:",
        round(mae, 4)
    )

    print(
        "MSE:",
        round(mse, 4)
    )

    print(
        "R2:",
        round(r2, 4)
    )


    # -------------------------------------------------
    # PREDICTED VS ACTUAL PLOT
    # -------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.scatter(
        actual,
        predicted
    )

    minimum = min(
        actual.min(),
        predicted.min()
    )

    maximum = max(
        actual.max(),
        predicted.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--"
    )

    plt.xlabel(
        "Actual Value"
    )

    plt.ylabel(
        "Predicted Value"
    )

    plt.title(
        "Predicted vs Actual - "
        + file_name.replace(".csv", "")
    )

    plt.tight_layout()

    prediction_file = (
        file_name.replace(
            ".csv",
            "_Predicted_vs_Actual.png"
        )
    )

    plt.savefig(
        prediction_file
    )

    plt.close()

    print(
        "Saved:",
        prediction_file
    )


    # -------------------------------------------------
    # RESIDUAL PLOT
    # -------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.scatter(
        actual,
        residuals
    )

    plt.axhline(
        0
    )

    plt.xlabel(
        "Actual Value"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Plot - "
        + file_name.replace(".csv", "")
    )

    plt.tight_layout()

    residual_file = (
        file_name.replace(
            ".csv",
            "_Residual.png"
        )
    )

    plt.savefig(
        residual_file
    )

    plt.close()

    print(
        "Saved:",
        residual_file
    )


    # -------------------------------------------------
    # EXTRA INFORMATION
    # -------------------------------------------------

    print("\nRESIDUAL INFORMATION")
    print("------------------------------")

    print(
        "Average residual:",
        round(residuals.mean(), 4)
    )

    print(
        "Largest positive residual:",
        round(residuals.max(), 4)
    )

    print(
        "Largest negative residual:",
        round(residuals.min(), 4)
    )

    print(
        "Largest absolute error:",
        round(np.abs(residuals).max(), 4)
    )


    return {
        "Dataset": file_name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "Average Residual": residuals.mean()
    }


# -------------------------------------------------
# RUN ALL DATASETS
# -------------------------------------------------

results = []

for dataset in datasets:

    if os.path.exists(dataset):

        result = analyze_dataset(
            dataset
        )

        if result is not None:
            results.append(result)

    else:

        print(
            "\nFILE NOT FOUND:",
            dataset
        )


# -------------------------------------------------
# RESULTS TABLE
# -------------------------------------------------

print("\n\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(
    results
)

if len(results_df) > 0:

    print(
        results_df.to_string(
            index=False
        )
    )


# -------------------------------------------------
# COMPLETE
# -------------------------------------------------

print("\n\nCHAPTER 8 RESIDUAL ANALYSIS COMPLETE")
print("Check the CobberResidue folder for the graphs.")