from pathlib import Path
import sqlite3

import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Find the project folder
ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
DATASETS.mkdir(exist_ok=True)

# STEP 2: Find the CSV file
files = sorted(DATASETS.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {DATASETS}\n"
        "Move your AR agonist CSV into the datasets folder."
    )

# Use the newest CSV if multiple files are present
csv_file = max(files, key=lambda file: file.stat().st_mtime)
print("Reading:", csv_file.name)

# STEP 3: Load the data
df = pd.read_csv(csv_file, dtype={"DTXSID": str})

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("%", "percent", regex=False)
    .str.replace(" ", "_", regex=False)
)

# STEP 4: Check required columns
required = ["dtxsid", "preferred_name", "hit_call"]

for column in required:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Clean identifiers and activity labels
df["dtxsid"] = df["dtxsid"].astype("string").str.strip()
df["preferred_name"] = df["preferred_name"].astype("string").str.strip()
df["hit_call"] = df["hit_call"].astype("string").str.strip().str.title()

# STEP 5: Check missing IDs and duplicates
missing_ids = df["dtxsid"].isna() | df["dtxsid"].eq("")
duplicate_count = df.loc[~missing_ids, "dtxsid"].duplicated().sum()

print("\nDATA QUALITY CHECK")
print("Original rows:", len(df))
print("Missing DTXSIDs:", missing_ids.sum())
print("Duplicate DTXSIDs:", duplicate_count)

# Remove rows without an ID
df = df.loc[~missing_ids].copy()

# Remove duplicate IDs, keeping the first record
df = df.drop_duplicates(subset="dtxsid", keep="first")

# STEP 6: Convert measurements to numbers
numeric_columns = [
    "monoisotopic_mass",
    "toxcast_active",
    "toxcast_total",
    "percent_toxcast_active",
    "continuous_hit_call",
    "top",
    "scaled_top",
    "ac50",
    "logac50",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# STEP 7: Save the master CSV
master_csv = ROOT / "master_ar_agonist.csv"
df.to_csv(master_csv, index=False)

# STEP 8: Create the SQLite database
database_path = ROOT / "master_ar_agonist.db"

with sqlite3.connect(database_path) as connection:
    df.to_sql(
        "ar_agonist_master",
        connection,
        if_exists="replace",
        index=False
    )

# STEP 9: Save missing-value report
missing_report = df.isna().sum().reset_index()
missing_report.columns = ["column", "missing_values"]
missing_report.to_csv(ROOT / "missing_values.csv", index=False)

# STEP 10: Save activity summary
hit_counts = df["hit_call"].value_counts()
hit_summary = hit_counts.rename_axis("hit_call").reset_index(name="count")
hit_summary.to_csv(ROOT / "hit_call_summary.csv", index=False)

# STEP 11: Create activity graph
plt.figure(figsize=(7, 5))
hit_counts.plot(kind="bar", color=["steelblue", "orange"])
plt.title("AR Agonist Activity Results")
plt.xlabel("Hit Call")
plt.ylabel("Number of Chemicals")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(ROOT / "agonist_hit_calls.png", dpi=300)
plt.close()

# STEP 12: Create molecular mass graph
if "monoisotopic_mass" in df.columns:
    masses = df["monoisotopic_mass"].dropna()

    if not masses.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(masses, bins=30, color="steelblue", edgecolor="black")
        plt.title("Molecular Mass Distribution")
        plt.xlabel("Monoisotopic Mass (Da)")
        plt.ylabel("Number of Chemicals")
        plt.tight_layout()
        plt.savefig(ROOT / "agonist_mass_distribution.png", dpi=300)
        plt.close()

# STEP 13: Display results
print("\nAR AGONIST DATABASE COMPLETE")
print("Total chemicals:", len(df))
print("Total columns:", len(df.columns))
print("\nActivity results:")
print(hit_counts)

print("\nFiles created:")
print("master_ar_agonist.csv")
print("master_ar_agonist.db")
print("missing_values.csv")
print("hit_call_summary.csv")
print("agonist_hit_calls.png")
print("agonist_mass_distribution.png (if mass data is available)")

# STEP 14: Search for chemicals
while True:
    search = input("\nEnter a chemical name to search (Enter to exit): ").strip()

    if not search:
        break

    results = df[
        df["preferred_name"].str.contains(
            search, case=False, na=False, regex=False
        )
    ]

    if results.empty:
        print("No chemicals found.")
    else:
        columns = ["dtxsid", "preferred_name", "hit_call"]
        if "ac50" in results.columns:
            columns.append("ac50")

        print(results[columns].to_string(index=False))

print("\nProgram finished.")