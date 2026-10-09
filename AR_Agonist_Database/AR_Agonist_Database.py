
from pathlib import Path
import sqlite3
import re
import pandas as pd
import matplotlib.pyplot as plt

# Find the project folder
ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
DATASETS.mkdir(exist_ok=True)

# Find all CSV files
files = sorted(DATASETS.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        "Put your AR agonist CSV inside the datasets folder."
    )

master = None
metadata = [
    "dtxsid",
    "preferred_name",
    "casrn",
    "molecular_formula",
    "monoisotopic_mass",
    "toxcast_active",
    "toxcast_total",
    "percent_toxcast_active"
]

reports = []

# Step 1: Read and clean every dataset
for number, file in enumerate(files):

    df = pd.read_csv(file, dtype=str)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("%", "percent", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # Clean chemical identifiers
    if "dtxsid" not in df.columns:
        raise ValueError(f"{file.name} has no DTXSID column")

    df["dtxsid"] = (
        df["dtxsid"]
        .str.extract(r"(DTXSID\d+)", expand=False)
    )

    if df["dtxsid"].isna().any():
        raise ValueError(f"Missing DTXSID in {file.name}")

    if df["dtxsid"].duplicated().any():
        raise ValueError(f"Duplicate DTXSID in {file.name}")

    # Convert measurement columns to numbers
    for col in df.columns:
        if col not in [
            "dtxsid",
            "preferred_name",
            "casrn",
            "molecular_formula",
            "hit_call"
        ]:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    if "hit_call" in df.columns:
        df["hit_call"] = df["hit_call"].str.strip()

    # Give each assay its own column names
    prefix = re.sub(
        r"[^a-z0-9]+",
        "_",
        file.stem.lower()
    ).strip("_")

    if number == 0:
        # Keep chemical information once
        keep = [c for c in metadata if c in df.columns]

        df = df.rename(columns={
            c: f"{prefix}__{c}"
            for c in df.columns
            if c not in keep
        })

        master = df

    else:
        # Add new assay results without overwriting old ones
        new_metadata = [
            c for c in metadata
            if c in df.columns and c != "dtxsid"
        ]

        df = df.drop(columns=new_metadata)

        df = df.rename(columns={
            c: f"{prefix}__{c}"
            for c in df.columns
            if c != "dtxsid"
        })

        master = pd.merge(
            master,
            df,
            on="dtxsid",
            how="outer",
            validate="one_to_one"
        )

    reports.append({
        "file": file.name,
        "chemicals": len(df)
    })

# Step 2: Save master dataset
master = master.sort_values("dtxsid")

master.to_csv(
    ROOT / "master_ar_agonist.csv",
    index=False
)

# Step 3: Create SQLite database
with sqlite3.connect(
    ROOT / "master_ar_agonist.db"
) as conn:

    master.to_sql(
        "ar_agonist_master",
        conn,
        if_exists="replace",
        index=False
    )

    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_dtxsid "
        "ON ar_agonist_master(dtxsid)"
    )

# Step 4: Save summaries
pd.DataFrame(reports).to_csv(
    ROOT / "source_summary.csv",
    index=False
)

master.isna().sum().to_csv(
    ROOT / "missing_values.csv",
    header=["missing_values"]
)

# Step 5: Find assay activity columns
hit_columns = [
    c for c in master.columns
    if c.endswith("__hit_call")
]

if hit_columns:

    summary = (
        master[hit_columns]
        .apply(lambda x: x.value_counts())
        .fillna(0)
        .astype(int)
    )

    summary.to_csv(
        ROOT / "hit_call_summary.csv"
    )

    # Graph first assay
    counts = (
        master[hit_columns[0]]
        .value_counts()
        .reindex(["Active", "Inactive"], fill_value=0)
    )

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)

    plt.title("AR Agonist: Active vs Inactive")
    plt.xlabel("Hit Call")
    plt.ylabel("Number of Chemicals")

    plt.tight_layout()
    plt.savefig(
        ROOT / "agonist_hit_calls.png",
        dpi=180
    )
    plt.close()

    print("\nActivity results:")
    print(counts)

# Step 6: Graph molecular mass
if "monoisotopic_mass" in master.columns:

    masses = pd.to_numeric(
        master["monoisotopic_mass"],
        errors="coerce"
    ).dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(masses, bins=40)

    plt.title("Molecular Mass Distribution")
    plt.xlabel("Monoisotopic Mass")
    plt.ylabel("Number of Chemicals")

    plt.tight_layout()
    plt.savefig(
        ROOT / "agonist_mass_distribution.png",
        dpi=180
    )
    plt.close()

# Step 7: Display results
print("\nMASTER DATABASE COMPLETE")
print("Datasets processed:", len(files))
print("Total chemicals:", len(master))
print("Total columns:", len(master.columns))
print("Duplicate DTXSIDs:", master["dtxsid"].duplicated().sum())

print("\nFiles created:")
print("master_ar_agonist.csv")
print("master_ar_agonist.db")
print("source_summary.csv")
print("missing_values.csv")
print("hit_call_summary.csv")
print("agonist_hit_calls.png")
print("agonist_mass_distribution.png")

# Step 8: Search chemicals
search = input(
    "\nEnter a chemical name to search (Enter to exit): "
).strip()

if search and "preferred_name" in master.columns:

    results = master[
        master["preferred_name"]
        .str.contains(search, case=False, na=False, regex=False)
    ]

    if results.empty:
        print("No chemicals found.")
    else:
        columns = [
            "dtxsid",
            "preferred_name"
        ] + hit_columns

        print(results[columns].head(20).to_string(index=False))
