
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# STEP 1: SET UP PROJECT FOLDER
# -----------------------------------

ROOT = Path(__file__).resolve().parent
CSV_FILE = ROOT / "alkali_metals.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(
        "alkali_metals.csv was not found. "
        "Put the CSV in the ElementClustering folder."
    )

# -----------------------------------
# STEP 2: LOAD AND PREVIEW DATA
# -----------------------------------

df = pd.read_csv(CSV_FILE)

print("\nCHAPTER 10: ELEMENT CLUSTERING")
print("\nOriginal Dataset:")
print(df.to_string(index=False))

print("\nDataset Information:")
print("Total elements:", len(df))
print("Total columns:", len(df.columns))

print("\nMissing Values:")
print(df.isna().sum())

required_columns = [
    "Element",
    "Symbol",
    "Atomic_Number",
    "Atomic_Radius_pm",
    "First_Ionization_Energy_kJ_mol"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

if df[required_columns].isna().any().any():
    raise ValueError("The dataset contains missing values.")

# -----------------------------------
# STEP 3: SELECT FEATURES
# -----------------------------------

features = [
    "Atomic_Radius_pm",
    "First_Ionization_Energy_kJ_mol"
]

X = df[features]

# Normalize features for distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# STEP 4: ORIGINAL SCATTER PLOT
# -----------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Atomic_Radius_pm"],
    df["First_Ionization_Energy_kJ_mol"],
    color="royalblue",
    s=130
)

for i, row in df.iterrows():
    plt.annotate(
        row["Symbol"],
        (
            row["Atomic_Radius_pm"],
            row["First_Ionization_Energy_kJ_mol"]
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.title("Group 1: Atomic Radius vs Ionization Energy")
plt.xlabel("Atomic Radius (pm)")
plt.ylabel("First Ionization Energy (kJ/mol)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "periodic_trends.png",
    dpi=300
)
plt.close()

# -----------------------------------
# STEP 5: RUN K-MEANS CLUSTERING
# -----------------------------------

def run_clustering(k):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    # Convert centroids back to original units
    centers = scaler.inverse_transform(
        model.cluster_centers_
    )

    results = df.copy()
    results["Cluster"] = clusters + 1

    print(f"\nK-MEANS RESULTS: k = {k}")
    print(
        results[
            ["Element", "Symbol", "Cluster"]
        ].to_string(index=False)
    )

    print("\nInertia:", round(model.inertia_, 4))

    # -----------------------------------
    # STEP 6: PLOT CLUSTERS
    # -----------------------------------

    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["Atomic_Radius_pm"],
        df["First_Ionization_Energy_kJ_mol"],
        c=clusters,
        cmap="tab10",
        s=150,
        edgecolors="black"
    )

    # Add element labels
    for i, row in df.iterrows():
        plt.annotate(
            row["Symbol"],
            (
                row["Atomic_Radius_pm"],
                row["First_Ionization_Energy_kJ_mol"]
            ),
            xytext=(6, 6),
            textcoords="offset points"
        )

    # Display cluster centers as black X markers
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        color="black",
        marker="X",
        s=280,
        label="Cluster Centers"
    )

    plt.title(f"K-Means Clustering of Alkali Metals (k={k})")
    plt.xlabel("Atomic Radius (pm)")
    plt.ylabel("First Ionization Energy (kJ/mol)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"element_clusters_k{k}.png"

    plt.savefig(
        ROOT / filename,
        dpi=300
    )

    plt.close()

    # Save cluster results
    results.to_csv(
        ROOT / f"cluster_results_k{k}.csv",
        index=False
    )

    print("Graph saved:", filename)

    return model.inertia_

# -----------------------------------
# STEP 7: INITIAL CLUSTERING
# -----------------------------------

run_clustering(2)

# -----------------------------------
# STEP 8: TEST MULTIPLE K VALUES
# -----------------------------------

inertia_results = []

for k in range(1, len(df) + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_results.append({
        "k": k,
        "Inertia": model.inertia_
    })

inertia_df = pd.DataFrame(inertia_results)

print("\nINERTIA RESULTS:")
print(inertia_df.to_string(index=False))

inertia_df.to_csv(
    ROOT / "inertia_results.csv",
    index=False
)

# -----------------------------------
# STEP 9: CREATE ELBOW GRAPH
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    inertia_df["k"],
    inertia_df["Inertia"],
    marker="o",
    color="darkorange"
)

plt.title("Elbow Method: K-Means Inertia")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(range(1, len(df) + 1))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROOT / "elbow_method.png",
    dpi=300
)

plt.close()

# -----------------------------------
# STEP 10: INTERACTIVE K LOOP
# -----------------------------------

while True:

    answer = input(
        "\nEnter a new k value (1-6), "
        "or press Enter to exit: "
    ).strip()

    if answer == "":
        break

    try:
        k = int(answer)

        if 1 <= k <= len(df):
            run_clustering(k)
        else:
            print("Choose a number between 1 and 6.")

    except ValueError:
        print("Please enter a whole number.")

print("\nCHAPTER 10 PROJECT COMPLETE")

print("\nFiles created:")
print("periodic_trends.png")
print("element_clusters_k2.png")
print("cluster_results_k2.csv")
print("inertia_results.csv")
print("elbow_method.png")

print("\nAdditional graphs are saved when you enter new k values.")
