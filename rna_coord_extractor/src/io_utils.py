import os
import pandas as pd

def save_coordinates_and_label(coords_df, row, output_dir, pdb_id, index=None):
    """
    Save the atomic coordinates and optional label into appropriate files.

    Args:
        coords_df (pd.DataFrame): DataFrame with atom data.
        row (pd.Series): Input row from the CSV.
        output_dir (str): Path to save outputs.
        pdb_id (str): Current PDB ID.
        index (int, optional): Row index (not used in file naming).
    """
    bp = row.get("bp", "unknown")
    os.makedirs(os.path.join(output_dir, bp), exist_ok=True)

    # Construct filename from DSSR columns
    fname_core = f"{pdb_id.lower()}_{row['chain1']}_{row['pos1']}_{row['chain2']}_{row['pos2']}_{row['bp1']}_{row['bp2']}"
    result_filename = os.path.join(output_dir, bp, f"{fname_core}.txt")
    label_filename = os.path.join(output_dir, bp, f"{fname_core}_label.txt")

    coords_df.to_csv(result_filename, sep='\t', index=False, header=None)

    if 'label' in row:
        with open(label_filename, 'w') as f:
            f.write(str(row['label']))
