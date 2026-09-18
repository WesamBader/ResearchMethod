import pubchempy as pcp

compound_name = input("Enter the name of a compound: ")

compounds = pcp.get_compounds(compound_name, "name")

if compounds:
    compound = compounds[0]

    print("\nCompound:", compound_name)
    print("Molecular Weight:", compound.molecular_weight)
    print("Molecular Formula:", compound.molecular_formula)
    print("Canonical SMILES:", compound.smiles)
    print("H-Bond Donors:", compound.h_bond_donor_count)
    print("H-Bond Acceptors:", compound.h_bond_acceptor_count)
    print("XLogP:", compound.xlogp)

else:
    print("Compound not found.")