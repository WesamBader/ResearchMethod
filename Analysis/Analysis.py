from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem


molecules = {
    "water": "O",
    "ammonia": "N",
    "ethanol": "CCO",
    "glucose": "OCC1OC(O)C(O)C(O)1O",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "morphine": "CN1CCC[C@]23[C@H]4Oc5c(O)ccc(C[C@H]1[C@H]2C=C[C@H]34)c5O"
}


def get_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)

    return {
        "Molecular Weight": Descriptors.MolWt(mol),
        "Exact Weight": Descriptors.ExactMolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "XLogP": Crippen.MolLogP(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol)
    }


for name, smiles in molecules.items():
    properties = get_properties(smiles)

    print("\n", name.upper())
    print("----------------------")

    for property_name, value in properties.items():
        print(property_name, ":", round(value, 2))
print("\n\nACTIVITY A - MOLECULAR WEIGHT")
print("================================")

water = get_properties(molecules["water"])["Molecular Weight"]
ammonia = get_properties(molecules["ammonia"])["Molecular Weight"]

print("Water molecular weight:", round(water, 2))
print("Ammonia molecular weight:", round(ammonia, 2))
print("Difference:", round(abs(water - ammonia), 2))
print("\nClosest molecular weight to ethanol:")

ethanol_weight = get_properties(molecules["ethanol"])["Molecular Weight"]

closest_name = None
closest_difference = float("inf")

for name, smiles in molecules.items():

    if name == "ethanol":
        continue

    weight = get_properties(smiles)["Molecular Weight"]
    difference = abs(ethanol_weight - weight)

    print(name, "difference =", round(difference, 2))

    if difference < closest_difference:
        closest_difference = difference
        closest_name = name

print("\nClosest:", closest_name)
print("\nFarthest molecular weight from glucose:")

glucose_weight = get_properties(molecules["glucose"])["Molecular Weight"]

farthest_name = None
largest_difference = -1

for name, smiles in molecules.items():

    if name == "glucose":
        continue

    weight = get_properties(smiles)["Molecular Weight"]
    difference = abs(glucose_weight - weight)

    if difference > largest_difference:
        largest_difference = difference
        farthest_name = name

print("Least similar:", farthest_name)
print("Weight difference:", round(largest_difference, 2))
print("\n\nACTIVITY B - WATER VS CAFFEINE")
print("================================")

water_properties = get_properties(molecules["water"])
caffeine_properties = get_properties(molecules["caffeine"])

print("Water XLogP:", round(water_properties["XLogP"], 2))
print("Caffeine XLogP:", round(caffeine_properties["XLogP"], 2))

print()

print("Water TPSA:", round(water_properties["TPSA"], 2))
print("Caffeine TPSA:", round(caffeine_properties["TPSA"], 2))

print()

print(
    "TPSA difference:",
    round(abs(water_properties["TPSA"] - caffeine_properties["TPSA"]), 2)
)

print(
    "XLogP difference:",
    round(abs(water_properties["XLogP"] - caffeine_properties["XLogP"]), 2)
)
print("\n\nACTIVITY C - FLEXIBILITY")
print("================================")

for name, smiles in molecules.items():

    rotatable = get_properties(smiles)["Rotatable Bonds"]

    print(name, ":", rotatable, "rotatable bonds")
    print("\n\nACTIVITY D - DRUG XLOGP")
    print("================================")

    drug_names = [
        "ibuprofen",
        "aspirin",
        "acetaminophen",
        "morphine"
    ]

    for name in drug_names:
        xlogp = get_properties(molecules[name])["XLogP"]

        print(name, "XLogP:", round(xlogp, 2))
ibuprofen = get_properties(molecules["ibuprofen"])["XLogP"]
aspirin = get_properties(molecules["aspirin"])["XLogP"]
acetaminophen = get_properties(molecules["acetaminophen"])["XLogP"]

aspirin_difference = abs(ibuprofen - aspirin)
acetaminophen_difference = abs(ibuprofen - acetaminophen)

print("\nIbuprofen vs aspirin difference:", round(aspirin_difference, 2))
print(
    "Ibuprofen vs acetaminophen difference:",
    round(acetaminophen_difference, 2)
)

if aspirin_difference < acetaminophen_difference:
    print("Aspirin is more similar to ibuprofen in XLogP.")
else:
    print("Acetaminophen is more similar to ibuprofen in XLogP.")
    print("\nIBUPROFEN VS MORPHINE")

    for name in ["ibuprofen", "morphine"]:
        props = get_properties(molecules[name])

        print("\n", name)
        print("XLogP:", round(props["XLogP"], 2))
        print("TPSA:", round(props["TPSA"], 2))
        print("Molecular Weight:", round(props["Molecular Weight"], 2))
        print("H-Bond Donors:", props["H-Bond Donors"])
        print("H-Bond Acceptors:", props["H-Bond Acceptors"])
amino_acids = {
    "glycine": "NCC(=O)O",
    "alanine": "CC(N)C(=O)O",
    "serine": "NC(CO)C(=O)O",
    "threonine": "CC(O)C(N)C(=O)O",
    "asparagine": "NC(=O)CC(N)C(=O)O",
    "glutamine": "NC(=O)CCC(N)C(=O)O",
    "tryptophan": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "lysine": "NCCCCC(N)C(=O)O",
    "aspartic acid": "NC(CC(=O)O)C(=O)O",
    "glutamic acid": "NC(CCC(=O)O)C(=O)O"
}


print("\n\nACTIVITY E - AMINO ACID TPSA")
print("================================")

for name, smiles in amino_acids.items():

    tpsa = get_properties(smiles)["TPSA"]

    print(name, "TPSA:", round(tpsa, 2))
glutamine_tpsa = get_properties(
    amino_acids["glutamine"]
)["TPSA"]

closest_name = None
closest_difference = float("inf")

for name, smiles in amino_acids.items():

    if name == "glutamine":
        continue

    tpsa = get_properties(smiles)["TPSA"]

    difference = abs(glutamine_tpsa - tpsa)

    if difference < closest_difference:
        closest_difference = difference
        closest_name = name

print(
    "\nAmino acid closest to glutamine based on TPSA:",
    closest_name
)
tryptophan_tpsa = get_properties(
    amino_acids["tryptophan"]
)["TPSA"]

farthest_name = None
largest_difference = -1

for name, smiles in amino_acids.items():

    if name == "tryptophan":
        continue

    tpsa = get_properties(smiles)["TPSA"]

    difference = abs(tryptophan_tpsa - tpsa)

    if difference > largest_difference:
        largest_difference = difference
        farthest_name = name

print(
    "Amino acid farthest from tryptophan based on TPSA:",
    farthest_name
)
