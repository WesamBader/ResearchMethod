import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = {
    "Molecule": [
        "Molecule 1", "Molecule 2", "Molecule 3", "Molecule 4",
        "Molecule 5", "Molecule 6", "Molecule 7", "Molecule 8",
        "Molecule 9", "Molecule 10", "Molecule 11", "Molecule 12"
    ],
    "Molecular Weight": [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

print(df)
X = df[
    ["Molecular Weight", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors"]
]

y = df["Water Solubility"]
model = DecisionTreeClassifier(random_state=42)

model.fit(X, y)

predictions = model.predict(X)

results = df[["Molecule", "Water Solubility"]].copy()
results["Predicted Solubility"] = predictions

print("\nDecision Tree Results:")
print(results.to_string(index=False))


# Step 8: Visualize the trained decision tree

plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Soluble", "Soluble"],
    filled=True,
    rounded=True,
    fontsize=12
)

plt.title("Decision Tree for Water Solubility")
plt.tight_layout()


# Step 9: Automatically save the decision tree image

plt.savefig("decision_tree.png", dpi=300)

plt.show()


# Step 10: Train another model using only Hydrogen Bond Donors

X_donors = df[["Hydrogen Bond Donors"]]

donor_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

donor_model.fit(X_donors, y)

donor_predictions = donor_model.predict(X_donors)

donor_results = df[["Molecule", "Water Solubility"]].copy()
donor_results["Predicted Using H-Bond Donors"] = donor_predictions

print("\nHydrogen Bond Donor Only Results:")
print(donor_results.to_string(index=False))


# Visualize the Hydrogen Bond Donor model

plt.figure(figsize=(12, 7))

plot_tree(
    donor_model,
    feature_names=["Hydrogen Bond Donors"],
    class_names=["Not Soluble", "Soluble"],
    filled=True,
    rounded=True,
    fontsize=12
)

plt.title("Decision Tree Using Hydrogen Bond Donors Only")
plt.tight_layout()

plt.savefig("decision_tree_donors_only.png", dpi=300)

plt.show()

print("\nFinished!")
print("Saved: decision_tree.png")
print("Saved: decision_tree_donors_only.png")