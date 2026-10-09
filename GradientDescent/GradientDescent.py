
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate noisy data
rng = np.random.default_rng(42)

x = np.linspace(0, 10, 50)
true_slope = 2.0
true_intercept = 5.0

true_y = true_slope * x + true_intercept
noise = rng.normal(0, 2.0, size=x.size)
y = true_y + noise

# Save the dataset
pd.DataFrame({
    "x": x,
    "true_y": true_y,
    "noisy_y": y
}).to_csv("gradient_data.csv", index=False)

# Step 2: Train the regression model
X = x.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

predicted_y = model.predict(X)

learned_slope = float(model.coef_[0])
learned_intercept = float(model.intercept_)

# Step 3: Print the results
print(f"True slope: {true_slope:.4f}")
print(f"Learned slope: {learned_slope:.4f}")
print(f"True intercept: {true_intercept:.4f}")
print(f"Learned intercept: {learned_intercept:.4f}")

print(
    f"Best-fit MSE: "
    f"{mean_squared_error(y, predicted_y):.4f}"
)

# Step 4: Plot the regression
plt.figure(figsize=(9, 6))

plt.scatter(x, y, label="Noisy observations", alpha=0.8)
plt.plot(x, true_y, label="True line: y = 2x + 5")
plt.plot(
    x, predicted_y,
    label="Learned best-fit line",
    linestyle="--"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression on Noisy Data")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("regression_scatterplot.png", dpi=300)
plt.close()

# Step 5: Create an MSE function
def calculate_mse(slope, intercept):
    guesses = slope * x + intercept
    return float(np.mean((y - guesses) ** 2))

print(
    "MSE at slope=1, intercept=1:",
    round(calculate_mse(1, 1), 4)
)

# Step 6: Build the loss landscape
slope_values = np.linspace(0, 4, 180)
intercept_values = np.linspace(0, 10, 180)

slopes, intercepts = np.meshgrid(
    slope_values, intercept_values
)

losses = np.zeros_like(slopes)

for row in range(slopes.shape[0]):
    for col in range(slopes.shape[1]):
        losses[row, col] = calculate_mse(
            slopes[row, col],
            intercepts[row, col]
        )

# Step 7: Plot the loss landscape
plt.figure(figsize=(9, 7))

contour = plt.contourf(
    slopes,
    intercepts,
    losses,
    levels=40,
    cmap="viridis_r"
)

plt.colorbar(contour, label="Mean Squared Error (MSE)")

plt.scatter(
    [true_slope],
    [true_intercept],
    color="lime",
    marker="*",
    s=220,
    edgecolors="black",
    label="True parameters"
)

plt.scatter(
    [learned_slope],
    [learned_intercept],
    color="deepskyblue",
    marker="X",
    s=160,
    edgecolors="black",
    label="Best-fit parameters"
)

plt.xlabel("Slope (m)")
plt.ylabel("Intercept (b)")
plt.title("Loss Landscape for Linear Regression")
plt.legend()
plt.tight_layout()

plt.savefig("loss_landscape.png", dpi=300)
plt.close()

# Step 8: Let the user test different MSE values
print("\nEnter a slope and intercept to calculate MSE.")
print("Press Enter without typing anything to finish.")

while True:
    entry = input("Slope (blank to exit): ").strip()

    if not entry:
        break

    try:
        slope = float(entry)
        intercept = float(input("Intercept: ").strip())

        print(
            f"MSE: {calculate_mse(slope, intercept):.4f}"
        )

    except ValueError:
        print("Please enter valid numbers.")

print("\nFinished!")
print("Saved: gradient_data.csv")
print("Saved: regression_scatterplot.png")
print("Saved: loss_landscape.png")
