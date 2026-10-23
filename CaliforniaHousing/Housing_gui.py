
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# Load trained model
folder = Path(__file__).resolve().parent
model_path = folder / "best_housing_model.joblib"

if not model_path.exists():
    raise SystemExit("Run housing_regression.py first.")

saved = joblib.load(model_path)
model = saved["model"]
features = saved["features"]
model_name = saved["model_name"]

defaults = {
    "MedInc": "4.0",
    "HouseAge": "25",
    "AveRooms": "4.5",
    "AveBedrms": "1.1",
    "Population": "1300",
    "AveOccup": "3.0",
    "Latitude": "34.05",
    "Longitude": "-118.25"
}

labels = {
    "MedInc": "Median Income ($10,000 units)",
    "HouseAge": "Median House Age",
    "AveRooms": "Average Rooms",
    "AveBedrms": "Average Bedrooms",
    "Population": "Population",
    "AveOccup": "Average Occupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}

root = tk.Tk()
root.title("California Housing Predictor")
root.geometry("620x590")
root.resizable(True, True)

# Title
tk.Label(
    root,
    text="California Housing Predictor",
    font=("Arial", 19, "bold")
).pack(pady=(12, 3))

tk.Label(
    root,
    text=f"Model: {model_name}",
    font=("Arial", 11)
).pack()

# Prediction displayed near top
result_var = tk.StringVar(value="No prediction yet")

tk.Label(
    root,
    textvariable=result_var,
    font=("Arial", 17, "bold"),
    fg="green",
    wraplength=550
).pack(pady=12)

# Input fields
frame = tk.Frame(root)
frame.pack(pady=5)

entries = {}

for i, feature in enumerate(features):
    tk.Label(
        frame,
        text=labels.get(feature, feature),
        width=30,
        anchor="w"
    ).grid(row=i, column=0, padx=5, pady=6)

    entry = tk.Entry(frame, width=20)
    entry.insert(0, defaults.get(feature, "0"))
    entry.grid(row=i, column=1, padx=5, pady=6)
    entries[feature] = entry


def predict_price():
    try:
        values = []

        for feature in features:
            value = float(entries[feature].get())

            if not np.isfinite(value):
                raise ValueError(f"Invalid value for {feature}")

            values.append(value)

        house = pd.DataFrame([values], columns=features)

        prediction = model.predict(house)[0]
        price = prediction * 100000

        result_var.set(f"Predicted Price: ${price:,.0f}")

        print(f"Predicted Price: ${price:,.0f}")

    except Exception as error:
        result_var.set("Prediction failed")
        messagebox.showerror("Prediction Error", str(error))
        print("Prediction Error:", error)


def reset_fields():
    for feature, entry in entries.items():
        entry.delete(0, tk.END)
        entry.insert(0, defaults.get(feature, "0"))

    result_var.set("No prediction yet")
    print("Fields reset")


# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=15)

tk.Button(
    button_frame,
    text="PREDICT PRICE",
    font=("Arial", 12, "bold"),
    command=predict_price,
    width=17
).grid(row=0, column=0, padx=10)

tk.Button(
    button_frame,
    text="RESET",
    font=("Arial", 12, "bold"),
    command=reset_fields,
    width=12
).grid(row=0, column=1, padx=10)

tk.Label(
    root,
    text="Predictions are estimates based on California census data.",
    font=("Arial", 9)
).pack(pady=5)

root.mainloop()
