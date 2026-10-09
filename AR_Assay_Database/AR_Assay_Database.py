
from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Find the CSV in the same folder as this Python file
FOLDER = Path(__file__).resolve().parent

CSV_FILES = sorted(
    FOLDER.glob("Assay List ACEA_AR_antagonist_AUC_viability*.csv")
)

if not CSV_FILES:
    raise FileNotFoundError(
        "Place your AR viability CSV file in this folder."
    )

# Step 1: Load the dataset
raw = pd.read_csv(
    CSV_FILES[0],
    encoding="utf-8-sig",
    skipinitialspace=True
)

raw.columns = raw.columns.str.strip()
raw = raw.replace(r"^\s*$", pd.NA, regex=True)

# Extract chemical IDs
raw["DTXSID"] = (
    raw["DTXSID"]
    .astype("string")
    .str.extract(r"(DTXSID\d+)", expand=False)
)

print("Original rows:", len(raw))
print("Duplicate rows:", raw.duplicated().sum())
print("Duplicate chemical IDs:", raw["DTXSID"].duplicated().sum())

# Step 2: Clean the dataset
clean = raw.drop_duplicates().copy()

clean.columns = [
    column.lower().replace("%", "percent").replace(" ", "_")
    for column in clean.columns
]

clean["hit_call"] = clean["hit_call"].astype("string").str.strip()

numeric_cols = [
    "monoisotopic_mass",
    "toxcast_active",
    "toxcast_total",
    "percent_toxcast_active",
    "continuous_hit_call",
    "top",
    "scaled_top",
    "ac50",
    "logac50"
]

for column in numeric_cols:
    clean[column] = pd.to_numeric(
        clean[column], errors="coerce"
    )

# Step 3: Create the SQLite database
database = FOLDER / "ar_viability.db"

with sqlite3.connect(database) as conn:
    clean.to_sql(
        "ar_viability",
        conn,
        if_exists="replace",
        index=False
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dtxsid "
        "ON ar_viability(dtxsid)"
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hit_call "
        "ON ar_viability(hit_call)"
    )

    summary = pd.read_sql_query(
        """
        SELECT hit_call, COUNT(*) AS chemicals
        FROM ar_viability
        GROUP BY hit_call
        ORDER BY chemicals DESC
        """,
        conn
    )

# Step 4: Save cleaned data and summaries
clean.to_csv(
    FOLDER / "cleaned_ar_viability.csv",
    index=False
)

missing = (
    clean.isna()
    .sum()
    .rename_axis("column")
    .reset_index(name="missing_values")
)

missing.to_csv(
    FOLDER / "missing_values.csv",
    index=False
)

summary.to_csv(
    FOLDER / "activity_summary.csv",
    index=False
)

# Step 5: Create graph of Active vs Inactive
plt.figure(figsize=(7, 5))
plt.bar(summary["hit_call"], summary["chemicals"])

plt.title("AR Antagonist Viability: Hit Calls")
plt.xlabel("Hit Call")
plt.ylabel("Number of Chemicals")

plt.tight_layout()
plt.savefig(FOLDER / "hit_call_counts.png", dpi=180)
plt.close()

# Step 6: Create molecular mass graph
plt.figure(figsize=(7, 5))

clean["monoisotopic_mass"].dropna().hist(bins=40)

plt.title("Monoisotopic Mass Distribution")
plt.xlabel("Monoisotopic Mass")
plt.ylabel("Number of Chemicals")

plt.tight_layout()
plt.savefig(FOLDER / "mass_distribution.png", dpi=180)
plt.close()

# Step 7: Display results
print("\nHit calls:")
print(summary.to_string(index=False))

print("\nMissing values:")
print(
    missing[missing["missing_values"] > 0].to_string(index=False)
)

print("\nDatabase created:", database)
print("Cleaned CSV, summaries, and graphs saved.")

# Step 8: Search the database
search = input(
    "\nSearch chemical name (or press Enter to exit): "
).strip()

if search:
    with sqlite3.connect(database) as conn:
        results = pd.read_sql_query(
            """
            SELECT dtxsid, preferred_name, hit_call, ac50
            FROM ar_viability
            WHERE preferred_name LIKE ?
            LIMIT 20
            """,
            conn,
            params=(f"%{search}%",)
        )

    if results.empty:
        print("No matches found.")
    else:
        print(results.to_string(index=False))
