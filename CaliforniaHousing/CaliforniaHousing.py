
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Save all files inside the CaliforniaHousing folder
FOLDER = Path(__file__).resolve().parent

print("CHAPTER 16: CALIFORNIA HOUSING REGRESSION")
print("=" * 50)

# 1. Load the dataset
# The first run may require internet access to download it.
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("\nDataset description:")
print(housing.DESCR[:1500])

print("\nFirst five rows:")
print(df.head().to_string())

print("\nBasic statistics:")
print(df.describe().round(2).to_string())

# 2. Separate input features and target prices
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Target prices are measured in units of $100,000.
feature_names = list(X.columns)

# 3. Create graphs for each feature
X.hist(figsize=(14, 10), bins=30)
plt.suptitle("California Housing Feature Distributions")
plt.tight_layout()
plt.savefig(FOLDER / "feature_histograms.png", dpi=150)
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("California Housing Correlation Heatmap")
plt.tight_layout()
plt.savefig(FOLDER / "correlation_heatmap.png", dpi=150)
plt.close()

# 5. Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 6. Create the four regression models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Support Vector Regression": make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=10, epsilon=0.1)
    )
}

results = []
trained_models = {}

print("\nMODEL COMPARISON")
print("=" * 50)

# 7. Train and evaluate every model
for name, model in models.items():
    print(f"\nTraining {name}...")

    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "Training Time (s)": training_time,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"Training Time: {training_time:.2f} seconds")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

    # Predicted vs. actual graph
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.25, s=12)

    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())

    plt.plot([low, high], [low, high], "r--")
    plt.xlabel("Actual Price ($100,000 units)")
    plt.ylabel("Predicted Price ($100,000 units)")
    plt.title(f"{name}: Predicted vs. Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(FOLDER / f"{filename}_predictions.png", dpi=150)
    plt.close()

# 8. Compare all model results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("R2", ascending=False)

print("\nFINAL MODEL RESULTS")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(FOLDER / "model_results.csv", index=False)

# 9. Select and save the best model
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print(f"\nBest Model: {best_name}")
print(f"Best R2: {results_df.iloc[0]['R2']:.4f}")

joblib.dump(
    {
        "model": best_model,
        "features": feature_names,
        "model_name": best_name
    },
    FOLDER / "best_housing_model.joblib"
)

# 10. Display the strongest correlations with housing prices
correlations = df.corr()["MedHouseVal"].drop("MedHouseVal")
correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

print("\nFEATURE CORRELATIONS WITH HOUSING PRICE")
print(correlations.round(3).to_string())

# 11. Feature importance if the best model supports it
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(
        best_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\nFEATURE IMPORTANCE")
    print(importance.round(4).to_string())

    plt.figure(figsize=(9, 5))
    importance.plot(kind="bar")
    plt.title(f"Feature Importance: {best_name}")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(FOLDER / "feature_importance.png", dpi=150)
    plt.close()

# 12. Predict prices for three example neighborhoods
# These values are illustrative, not actual property listings.
examples = pd.DataFrame([
    [3.0, 30, 5.0, 1.1, 1200, 3.0, 34.05, -118.25],
    [6.0, 20, 6.0, 1.0, 900, 2.5, 37.77, -122.42],
    [9.0, 15, 7.0, 1.2, 700, 2.2, 37.45, -122.15]
], columns=feature_names)

print("\nEXAMPLE HOUSING PREDICTIONS")

example_predictions = best_model.predict(examples)

for index, prediction in enumerate(example_predictions, start=1):
    print(f"Example {index}: ${prediction * 100000:,.0f}")

# 13. Allow the user to enter custom housing values
print("\nCUSTOM HOUSING PREDICTIONS")
print("Enter all eight features, or type 'q' to quit.")

while True:
    answer = input("\nMake another prediction? (y/n): ").strip().lower()

    if answer not in ("y", "yes"):
        break

    values = []

    try:
        for feature in feature_names:
            value = float(input(f"Enter {feature}: "))
            if not np.isfinite(value):
                raise ValueError("Values must be finite numbers.")
            values.append(value)

        custom_house = pd.DataFrame(
            [values], columns=feature_names
        )

        predicted_price = best_model.predict(custom_house)[0]

        print(f"\nPredicted Housing Value: ${predicted_price * 100000:,.0f}")

    except ValueError:
        print("Invalid input. Please enter numbers only.")

print("\nFinished! Results, graphs, and model saved.")
