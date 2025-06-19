import numpy as np
import pandas as pd
from src.color_code_atoms import get_color 

def get_atomic_coordinates_from_row(row, structure):
    """
    Extracts atomic coordinates and RGB colors for two nucleotides described in a DataFrame row.

    Args:
        row (pd.Series): A row with nt1/nt2, chain1/chain2, pos1/pos2.
        structure: BioPython structure object.

    Returns:
        pd.DataFrame: Combined atomic coordinates and RGB values for both residues.
    """
    coords1, coords2 = [], []
    atom1, atom2 = [], []
    atom_n1, atom_n2 = [], []
    rn1, rn2 = [], []

    chain_id1, residue_name1, residue_number1 = row['chain1'], row['nt1'], row['pos1']
    chain_id2, residue_name2, residue_number2 = row['chain2'], row['nt2'], row['pos2']

    for model in structure:
        for chain in model:
            if chain.id == chain_id1:
                for residue in chain:
                    if residue.resname == residue_name1 and str(residue.id[1]) == str(residue_number1):
                        for atom in residue:
                            coords1.append(atom.coord)
                            atom1.append(get_color(atom.name[0]))
                            atom_n1.append(atom.name + "_1")
                            rn1.append(residue_name1)
            if chain.id == chain_id2:
                for residue in chain:
                    if residue.resname == residue_name2 and str(residue.id[1]) == str(residue_number2):
                        for atom in residue:
                            coords2.append(atom.coord)
                            atom2.append(get_color(atom.name[0]))
                            atom_n2.append(atom.name + "_2")
                            rn2.append(residue_name2)

    coords1, coords2 = np.array(coords1), np.array(coords2)
    atom1, atom2 = np.array(atom1), np.array(atom2)

    df1 = pd.DataFrame({
        'res': rn1, 'atom': atom_n1,
        'X': coords1[:, 0], 'Y': coords1[:, 1], 'Z': coords1[:, 2],
        'R': atom1[:, 0], 'G': atom1[:, 1], 'B': atom1[:, 2]
    })
    df2 = pd.DataFrame({
        'res': rn2, 'atom': atom_n2,
        'X': coords2[:, 0], 'Y': coords2[:, 1], 'Z': coords2[:, 2],
        'R': atom2[:, 0], 'G': atom2[:, 1], 'B': atom2[:, 2]
    })

    return pd.concat([df1, df2]).reset_index(drop=True)
