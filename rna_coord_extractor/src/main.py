import argparse
import pandas as pd
import os
import sys
from src.pdb_utils import retrieve_pdb
from src.structure_utils import get_atomic_coordinates_from_row
from src.io_utils import save_coordinates_and_label
from src.extract_pdbID import extract_pdb
from src.swap_labels import swap_class 

def main(args):
    pdb_list_path = '/home/s081p868/scratch/RNA_annotations/bp_annotations/RNA_chain_list_final'
    pdb_list, _ = extract_pdb(pdb_list_path)
    print(f"Found {len(pdb_list)} PDB entries")
    
    # Read all DSSR files in the input folder
    for pdb_id in pdb_list:
        file_path = os.path.join(args.input_csv, f"{pdb_id.lower()}_dssr.csv")
        if not os.path.exists(file_path):
            print(f"Skipping {pdb_id}: No file found")
            continue

        print(f"Processing {pdb_id}")
        try:
            df = pd.read_csv(file_path).astype(str)
            df["pdb_id"] = pdb_id
        except Exception as e:
            print(f"Error reading CSV for {pdb_id}: {e}")
            continue

        try:
            structure = retrieve_pdb(pdb_id, args.pdb_dir)
        except Exception as e:
            print(f"Error retrieving structure for {pdb_id}: {e}")
            continue

        for i, row in df.iterrows():
            try:
                # Apply swap_class to get standardized base pair label and base pair
                bp, label = swap_class(row['bp1'] + row['bp2'], row['label'])
                row['bp'] = bp
                row['label'] = label

                coords_df = get_atomic_coordinates_from_row(row, structure)
                save_coordinates_and_label(coords_df, row, args.output_dir, pdb_id, i)

            except Exception as e:
                print(f"Error processing {pdb_id}_{i}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Folder of DSSR CSVs (not a single CSV anymore)")
    parser.add_argument("--output_dir", required=True, help="Where to store coordinate files")
    parser.add_argument("--pdb_dir", default="./pdbs", help="Directory to cache PDB/mmCIF files")
    args = parser.parse_args()
    main(args)
