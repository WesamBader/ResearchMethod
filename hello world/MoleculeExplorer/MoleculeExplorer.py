from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors

smiles = input("Enter a SMILES string: ")

molecule = Chem.MolFromSmiles(smiles)

if molecule:
    print("\nMolecular Properties")
    print("--------------------")
    print("Exact Molecular Weight:", Descriptors.ExactMolWt(molecule))
    print("Hydrogen Bond Donors:", Lipinski.NumHDonors(molecule))
    print("TPSA:", rdMolDescriptors.CalcTPSA(molecule))
    print("Hydrogen Bond Acceptors:", Lipinski.NumHAcceptors(molecule))
    print("Rotatable Bonds:", Lipinski.NumRotatableBonds(molecule))

else:
    print("Invalid SMILES string.")