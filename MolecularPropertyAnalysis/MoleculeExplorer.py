#chapter 6
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen

# Common chemicals and their SMILES
chemical_library = {
    "water": "O",
    "ammonia": "N",
    "ethanol": "CCO",
    "glucose": "OCC1OC(O)C(O)C(O)1O",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "morphine": "CN1CCC[C@]23[C@H]4Oc5c(O)ccc(C[C@H]1[C@H]2C=C[C@H]34)c5O",
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


def analyze_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        print("Invalid molecule or SMILES string.")
        return

    print("\nMOLECULAR PROPERTIES")
    print("-------------------------")
    print("Molecular Weight:", round(Descriptors.MolWt(molecule), 2))
    print("Exact Molecular Weight:", round(Descriptors.ExactMolWt(molecule), 2))
    print("TPSA:", round(rdMolDescriptors.CalcTPSA(molecule), 2))
    print("XLogP:", round(Crippen.MolLogP(molecule), 2))
    print("Rotatable Bonds:", Lipinski.NumRotatableBonds(molecule))
    print("H-Bond Donors:", Lipinski.NumHDonors(molecule))
    print("H-Bond Acceptors:", Lipinski.NumHAcceptors(molecule))


while True:

    user_input = input(
        "\nEnter a chemical name or SMILES string "
        "(type quit to stop): "
    ).strip()

    if user_input.lower() == "quit":
        print("Program ended.")
        break

    # Check if user typed a known chemical name
    if user_input.lower() in chemical_library:
        smiles = chemical_library[user_input.lower()]

        print("\nChemical:", user_input.title())
        print("SMILES:", smiles)

        analyze_molecule(smiles)

    else:
        # If it is not a known name, try treating it as SMILES
        print("\nTrying input as a SMILES string...")
        analyze_molecule(user_input)