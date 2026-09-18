# What I did:
# I calculated MAE, MSE, R2, and residuals using Python.

# What I learned:
# I learned how different error measurements can be used
# to evaluate the accuracy of predictions.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -------------------------------------------------
# ACTUAL AND PREDICTED VALUES
# -------------------------------------------------

actual = np.array([
    2, 4, 5, 4, 5, 7, 9
])

predicted = np.array([
    2.5, 3.5, 4, 5, 6, 8, 8
])


print("\nACTUAL VALUES")
print(actual)

print("\nPREDICTED VALUES")
print(predicted)
# -------------------------------------------------
# RESIDUALS
# -------------------------------------------------

residuals = predicted - actual

print("\nRESIDUALS")
print(residuals)
# -------------------------------------------------
# MAE FROM SCRATCH
# -------------------------------------------------

absolute_errors = np.abs(
    residuals
)

mae_manual = np.mean(
    absolute_errors
)

print(
    "\nManual MAE:",
    round(mae_manual, 4)
)
# -------------------------------------------------
# R2 FROM SCRATCH
# -------------------------------------------------

ss_residual = np.sum(
    (actual - predicted) ** 2
)

actual_mean = np.mean(
    actual
)

ss_total = np.sum(
    (actual - actual_mean) ** 2
)

r2_manual = (
    1 - (ss_residual / ss_total)
)

print(
    "Manual R2:",
    round(r2_manual, 4)
)
# -------------------------------------------------
# SCIKIT-LEARN CHECK
# -------------------------------------------------

mae_library = mean_absolute_error(
    actual,
    predicted
)

mse_library = mean_squared_error(
    actual,
    predicted
)

r2_library = r2_score(
    actual,
    predicted
)

print("\nSCIKIT-LEARN RESULTS")
print("------------------------------")

print(
    "MAE:",
    round(mae_library, 4)
)

print(
    "MSE:",
    round(mse_library, 4)
)

print(
    "R2:",
    round(r2_library, 4)
)


print("\nDO THE RESULTS MATCH?")
# -------------------------------------------------
# MSE FROM SCRATCH
# -------------------------------------------------

squared_errors = residuals ** 2

mse_manual = np.mean(
    squared_errors
)

print(
    "Manual MSE:",
    round(mse_manual, 4)
)
print(
    "MAE matches:",
    np.isclose(
        mae_manual,
        mae_library
    )
)

print(
    "MSE matches:",
    np.isclose(
        mse_manual,
        mse_library
    )
)

print(
    "R2 matches:",
    np.isclose(
        r2_manual,
        r2_library
    )
)
# MAE
absolute_errors = np.abs(residuals)
mae_manual = np.mean(absolute_errors)

# MSE
squared_errors = residuals ** 2
mse_manual = np.mean(squared_errors)

# R2
ss_residual = np.sum((actual - predicted) ** 2)
actual_mean = np.mean(actual)
ss_total = np.sum((actual - actual_mean) ** 2)
r2_manual = 1 - (ss_residual / ss_total)
# Chapter 8 - Error Metrics
# How Good Were Your Guesses?

# What I did:
# I calculated residuals, MAE, MSE, and R2 using NumPy.
# I then checked my calculations using scikit-learn.
# I also created graphs to visualize prediction accuracy.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -------------------------------------------------
# ACTUAL AND PREDICTED VALUES
# -------------------------------------------------

actual = np.array([
    2, 4, 5, 4, 5, 7, 9
], dtype=float)

predicted = np.array([
    2.5, 3.5, 4, 5, 6, 8, 8
], dtype=float)

print("\nACTUAL VALUES")
print(actual)

print("\nPREDICTED VALUES")
print(predicted)


# -------------------------------------------------
# RESIDUALS
# Residual = Predicted - Actual
# -------------------------------------------------

residuals = predicted - actual

print("\nRESIDUALS")
print(residuals)


# -------------------------------------------------
# MAE FROM SCRATCH
# -------------------------------------------------

absolute_errors = np.abs(residuals)

mae_manual = np.mean(
    absolute_errors
)

print(
    "\nManual MAE:",
    round(mae_manual, 4)
)


# -------------------------------------------------
# MSE FROM SCRATCH
# -------------------------------------------------

squared_errors = residuals ** 2

mse_manual = np.mean(
    squared_errors
)

print(
    "Manual MSE:",
    round(mse_manual, 4)
)


# -------------------------------------------------
# R2 FROM SCRATCH
# -------------------------------------------------

actual_mean = np.mean(actual)

ss_residual = np.sum(
    (actual - predicted) ** 2
)

ss_total = np.sum(
    (actual - actual_mean) ** 2
)

r2_manual = 1 - (
    ss_residual / ss_total
)

print(
    "Manual R2:",
    round(r2_manual, 4)
)


# -------------------------------------------------
# CHECK RESULTS WITH SCIKIT-LEARN
# -------------------------------------------------

mae_library = mean_absolute_error(
    actual,
    predicted
)

mse_library = mean_squared_error(
    actual,
    predicted
)

r2_library = r2_score(
    actual,
    predicted
)

print("\nSCIKIT-LEARN RESULTS")
print("--------------------------------")

print(
    "MAE:",
    round(mae_library, 4)
)

print(
    "MSE:",
    round(mse_library, 4)
)

print(
    "R2:",
    round(r2_library, 4)
)


# -------------------------------------------------
# CHECK IF MANUAL RESULTS MATCH
# -------------------------------------------------

print("\nDO THE RESULTS MATCH?")
print("--------------------------------")

print(
    "MAE matches:",
    np.isclose(
        mae_manual,
        mae_library
    )
)

print(
    "MSE matches:",
    np.isclose(
        mse_manual,
        mse_library
    )
)

print(
    "R2 matches:",
    np.isclose(
        r2_manual,
        r2_library
    )
)


# -------------------------------------------------
# IMPROVEMENT 1
# DISPLAY ACTUAL, PREDICTED, AND RESIDUAL VALUES
# -------------------------------------------------

print("\nACTUAL | PREDICTED | RESIDUAL")
print("--------------------------------")

for a, p, r in zip(
    actual,
    predicted,
    residuals
):
    print(
        f"{a:6.1f} | {p:9.1f} | {r:8.1f}"
    )


# -------------------------------------------------
# FIND WORST PREDICTION
# -------------------------------------------------

worst_index = np.argmax(
    np.abs(residuals)
)

print("\nWORST PREDICTION")
print("--------------------------------")

print(
    "Actual:",
    actual[worst_index]
)

print(
    "Predicted:",
    predicted[worst_index]
)

print(
    "Residual:",
    residuals[worst_index]
)


# -------------------------------------------------
# PREDICTED VS ACTUAL GRAPH
# -------------------------------------------------

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    actual,
    predicted,
    s=80,
    label="Predictions"
)

minimum = min(
    actual.min(),
    predicted.min()
)

maximum = max(
    actual.max(),
    predicted.max()
)

# Ideal fit line
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--",
    label="Ideal Fit"
)


# -------------------------------------------------
# IMPROVEMENT 2
# HIGHLIGHT WORST PREDICTION
# -------------------------------------------------

plt.scatter(
    actual[worst_index],
    predicted[worst_index],
    s=160,
    color="red",
    label="Worst Prediction"
)

plt.xlabel(
    "Actual Value",
    fontsize=12
)

plt.ylabel(
    "Predicted Value",
    fontsize=12
)

plt.title(
    "Predicted vs Actual",
    fontsize=14
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Predicted_vs_Actual.png",
    dpi=300
)

plt.close()

print(
    "\nSaved: Predicted_vs_Actual.png"
)


# -------------------------------------------------
# RESIDUAL PLOT
# -------------------------------------------------

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    actual,
    residuals,
    s=80
)

# Zero error line
plt.axhline(
    0,
    linestyle="--"
)

# Highlight worst prediction
plt.scatter(
    actual[worst_index],
    residuals[worst_index],
    s=160,
    color="red",
    label="Worst Prediction"
)

plt.xlabel(
    "Actual Value",
    fontsize=12
)

plt.ylabel(
    "Residual",
    fontsize=12
)

plt.title(
    "Residual Plot",
    fontsize=14
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Residual_Plot.png",
    dpi=300
)

plt.close()

print(
    "Saved: Residual_Plot.png"
)


# -------------------------------------------------
# FINAL RESULTS
# -------------------------------------------------

print("\n\nERROR METRICS PROJECT COMPLETE")
print("================================")

print(
    "MAE:",
    round(mae_manual, 4)
)

print(
    "MSE:",
    round(mse_manual, 4)
)

print(
    "R2:",
    round(r2_manual, 4)
)

print(
    "\nCheck the ErrorMetrics folder for your two graphs."
)