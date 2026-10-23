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

from pathlib import Path
import time
import requests
import pandas as pd

# =====================================
# STEP 1: PROJECT FILES
# =====================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "master_ar_agonist.csv"
OUTPUT_FILE = ROOT / "master_ar_agonist_with_smiles.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "master_ar_agonist.csv was not found. "
        "Run AR_Agonist_Database.py first."
    )

# =====================================
# STEP 2: LOAD DATABASE
# =====================================

df = pd.read_csv(INPUT_FILE, dtype=str)

df.columns = df.columns.str.strip().str.lower()

required = ["dtxsid", "preferred_name", "casrn"]

for column in required:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

print("\nFETCHING SMILES FROM PUBCHEM")
print("Total chemicals:", len(df))

# =====================================
# STEP 3: SET UP PUBCHEM API
# =====================================

BASE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    "compound/name"
)

session = requests.Session()
session.headers.update({
    "User-Agent": "AcademicChemicalResearch/1.0",
    "Accept": "application/json"
})

# =====================================
# STEP 4: FUNCTION TO FETCH SMILES
# =====================================

def fetch_smiles(identifier):

    if pd.isna(identifier):
        return None, None, "Not found"

    identifier = str(identifier).strip()

    if not identifier:
        return None, None, "Not found"

    url = (
        f"{BASE_URL}/"
        "property/SMILES/JSON"
    )

    for attempt in range(3):

        try:
            response = session.post(
                url,
                data={"name": identifier},
                timeout=30
            )

            if response.status_code == 404:
                return None, None, "Not found"

            if response.status_code in [429, 500, 502, 503, 504]:
                time.sleep(3 * (attempt + 1))
                continue

            response.raise_for_status()

            data = response.json()

            properties = data.get(
                "PropertyTable", {}
            ).get("Properties", [])

            if properties:
                record = properties[0]

                return (
                    record.get("SMILES"),
                    record.get("CID"),
                    "Found"
                )

            return None, None, "Not found"

        except requests.RequestException as error:
            print("Request error:", error)
            time.sleep(3 * (attempt + 1))

    return None, None, "Request failed"

# =====================================
# STEP 5: LOAD PREVIOUS PROGRESS
# =====================================

completed = {}

if OUTPUT_FILE.exists():

    old = pd.read_csv(
        OUTPUT_FILE,
        dtype=str
    )

    if "dtxsid" in old.columns and "SMILES_Status" in old.columns:

        for _, row in old.iterrows():

            if row["SMILES_Status"] in ["Found", "Not found"]:

                completed[row["dtxsid"]] = {
                    "SMILES": row.get("SMILES"),
                    "PubChem_CID": row.get("PubChem_CID"),
                    "SMILES_Status": row["SMILES_Status"],
                    "SMILES_Search_Method": row.get(
                        "SMILES_Search_Method"
                    )
                }

print("Previously completed:", len(completed))

# =====================================
# STEP 6: FETCH CHEMICAL STRUCTURES
# =====================================

results = []

for index, row in df.iterrows():

    dtxsid = row["dtxsid"]

    if dtxsid in completed:
        result = completed[dtxsid]

    else:

        casrn = row["casrn"]
        name = row["preferred_name"]

        smiles = None
        cid = None
        status = "Not found"
        method = "None"

        # First search by CASRN
        if pd.notna(casrn) and str(casrn).strip():

            smiles, cid, status = fetch_smiles(casrn)
            method = "CASRN"

        # If not found, search by chemical name
        if status == "Not found" and pd.notna(name):

            smiles, cid, status = fetch_smiles(name)
            method = "Preferred Name"

        result = {
            "SMILES": smiles,
            "PubChem_CID": cid,
            "SMILES_Status": status,
            "SMILES_Search_Method": method
        }

    results.append(result)

    # Save every 25 chemicals
    if len(results) % 25 == 0:

        progress = df.iloc[:len(results)].copy()

        for column in [
            "SMILES",
            "PubChem_CID",
            "SMILES_Status",
            "SMILES_Search_Method"
        ]:
            progress[column] = [
                item[column] for item in results
            ]

        progress.to_csv(OUTPUT_FILE, index=False)

        print(
            f"Processed {len(results)} / {len(df)} chemicals"
        )

    time.sleep(0.25)

# =====================================
# STEP 7: CREATE FINAL DATABASE
# =====================================

for column in [
    "SMILES",
    "PubChem_CID",
    "SMILES_Status",
    "SMILES_Search_Method"
]:
    df[column] = [
        item[column] for item in results
    ]

df.to_csv(OUTPUT_FILE, index=False)

# =====================================
# STEP 8: DISPLAY RESULTS
# =====================================

print("\nSMILES DATABASE COMPLETE")

print("Total chemicals:", len(df))

print(
    "SMILES found:",
    df["SMILES"].notna().sum()
)

print(
    "SMILES missing:",
    df["SMILES"].isna().sum()
)

print("\nSearch status:")
print(df["SMILES_Status"].value_counts())

print("\nPreview:")

print(
    df[
        [
            "dtxsid",
            "preferred_name",
            "SMILES",
            "PubChem_CID"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved file:")
print(OUTPUT_FILE)
