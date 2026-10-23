
from pathlib import Path
import time
import requests
import pandas as pd

# =====================================
# STEP 1: FILE LOCATIONS
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
# STEP 2: LOAD THE DATABASE
# =====================================

df = pd.read_csv(INPUT_FILE, dtype=str)
df.columns = df.columns.str.strip().str.lower()

required = ["dtxsid", "preferred_name", "casrn"]

for column in required:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

df["dtxsid"] = df["dtxsid"].str.strip()

print("\nFETCHING SMILES FROM PUBCHEM", flush=True)
print("Total chemicals:", len(df), flush=True)

# =====================================
# STEP 3: PUBCHEM CONNECTION
# =====================================

BASE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    "compound/name/property/"
    "CanonicalSMILES,IsomericSMILES/JSON"
)

session = requests.Session()

session.headers.update({
    "User-Agent": "AcademicChemicalResearch/1.0",
    "Accept": "application/json"
})

# =====================================
# STEP 4: FETCH SMILES FUNCTION
# =====================================

def fetch_smiles(identifier):

    if pd.isna(identifier):
        return None, None, "Not found"

    identifier = str(identifier).strip()

    if not identifier:
        return None, None, "Not found"

    for attempt in range(3):

        try:
            response = session.post(
                BASE_URL,
                data={"name": identifier},
                timeout=40
            )

            if response.status_code in (400, 404):
                return None, None, "Not found"

            if response.status_code in (
                429, 500, 502, 503, 504
            ):
                time.sleep(5 * (attempt + 1))
                continue

            response.raise_for_status()

            data = response.json()

            records = data.get(
                "PropertyTable", {}
            ).get("Properties", [])

            if records:
                record = records[0]

                smiles = (
                    record.get("SMILES")
                    or record.get("ConnectivitySMILES")
                    or record.get("CanonicalSMILES")
                    or record.get("IsomericSMILES")
                )

                return (
                    smiles,
                    record.get("CID"),
                    "Found" if smiles else "Not found"
                )

            return None, None, "Not found"

        except requests.RequestException as error:
            print(
                f"Request error for {identifier}: {error}",
                flush=True
            )
            time.sleep(3 * (attempt + 1))

    return None, None, "Request failed"

# =====================================
# STEP 5: LOAD PREVIOUS PROGRESS
# =====================================

completed = {}

if OUTPUT_FILE.exists():

    previous = pd.read_csv(
        OUTPUT_FILE,
        dtype=str
    )

    needed = {"dtxsid", "SMILES_Status"}

    if needed.issubset(previous.columns):

        for _, row in previous.iterrows():

            if row["SMILES_Status"] in (
                "Found", "Not found"
            ):

                completed[row["dtxsid"]] = {
                    "SMILES": row.get("SMILES"),
                    "PubChem_CID": row.get("PubChem_CID"),
                    "SMILES_Status": row.get("SMILES_Status"),
                    "SMILES_Search_Method": row.get(
                        "SMILES_Search_Method"
                    )
                }

print(
    "Previously completed:",
    len(completed),
    flush=True
)

# =====================================
# STEP 6: PROCESS CHEMICALS
# =====================================

results = []

for index, row in df.iterrows():

    dtxsid = row["dtxsid"]

    if dtxsid in completed:
        result = completed[dtxsid]

    else:
        smiles = None
        cid = None
        status = "Not found"
        method = "None"

        casrn = row["casrn"]
        name = row["preferred_name"]

        # First try the CAS registry number
        if pd.notna(casrn) and casrn.strip():

            smiles, cid, status = fetch_smiles(casrn)
            method = "CASRN"

        # Try chemical name if CASRN was not found
        if (
            status == "Not found"
            and pd.notna(name)
            and name.strip()
        ):

            smiles, cid, status = fetch_smiles(name)
            method = "Preferred Name"

        result = {
            "SMILES": smiles,
            "PubChem_CID": cid,
            "SMILES_Status": status,
            "SMILES_Search_Method": method
        }

        # Be considerate of PubChem's API limits
        time.sleep(0.25)

    results.append(result)

    # Save progress every 25 chemicals
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

        progress.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"Processed {len(results)} / {len(df)}",
            flush=True
        )

# =====================================
# STEP 7: SAVE FINAL DATABASE
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

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =====================================
# STEP 8: DISPLAY FINAL RESULTS
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

print("\nFirst 10 chemicals:")

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

print("\nFile saved:")
print(OUTPUT_FILE)

print("\nPROGRAM FINISHED")
