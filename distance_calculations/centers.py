import os
import csv
import sys
import numpy as np
from Bio.PDB import MMCIFParser, PDBExceptions

RNA_RESIDUE_NAMES = {'A', 'U', 'C', 'G', 'RA', 'RU', 'RC', 'RG', 'I'}
CIF_INPUT_FOLDER = "cif"
CSV_OUTPUT_FOLDER = "rna_center_distance_csvs_2"

# Dictionary defining the set of atoms for each base
BASE_ATOMS = {
    'A': {'N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C6', 'N6', 'C2', 'N1'},
    'G': {'N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C6', 'O6', 'N1', 'C25h', 'N2'},
    'C': {'N1', 'C5', 'C6', 'C2', 'O2', 'N3', 'C4', 'N4'},
    'U': {'N1', 'C5', 'C6', 'C2', 'O2', 'N3', 'C4', 'O4'}
}

# Add aliases for common alternatives
BASE_ATOMS['RA'] = BASE_ATOMS['A']
BASE_ATOMS['RG'] = BASE_ATOMS['G']
BASE_ATOMS['RC'] = BASE_ATOMS['C']
BASE_ATOMS['RU'] = BASE_ATOMS['U']

def get_residue_center(residue):
    res_name = residue.get_resname().strip()
    target_atom_names = BASE_ATOMS.get(res_name)
    
    atom_coords = []

    if target_atom_names:
        # Standard case: Use the predefined set of base atoms
        for atom in residue.get_atoms():
            if atom.get_id() in target_atom_names:
                atom_coords.append(atom.get_coord())
        
        # Check if we found any of the target atoms (for malformed files)
        if not atom_coords:
            print(f"Warning: For residue {res_name} {residue.get_id()}, none of the target base atoms were found. "
                  "Falling back to all-atom centroid.", file=sys.stderr)
            
            for atom in residue.get_atoms():
                atom_coords.append(atom.get_coord())
    else:
        # Fallback case: Residue name not in our dictionary
        print(f"Warning: Residue type '{res_name}' {residue.get_id()} has no defined base atom set. "
              "Calculating center of ALL atoms as a fallback.", file=sys.stderr)
        for atom in residue.get_atoms():
            atom_coords.append(atom.get_coord())

    if not atom_coords:
        print(f"Warning: No atoms found at all for residue {residue.get_id()}.", file=sys.stderr)
        return None
    
    # Calculate the mean of coordinates along each axis (x, y, z)
    center = np.mean(np.array(atom_coords), axis=0)
    return center

def calculate_rna_nucleotide_center_distances(cif_filepath, output_csv_filepath):
    """
    Parses a CIF file, identifies RNA nucleotides, calculates distances
    between their geometric centers, and writes them to a CSV file.

    Args:
        cif_filepath (str): Path to the input CIF file.
        output_csv_filepath (str): Path to save the output CSV file.
    """
    parser = MMCIFParser(QUIET=True)

    try:
        structure = parser.get_structure("rna_structure", cif_filepath)
    except PDBExceptions.PDBConstructionException as e:
        print(f"Error parsing CIF file {cif_filepath}: {e}. Skipping.")
        return
    except FileNotFoundError:
        print(f"CIF file not found: {cif_filepath}. Skipping.")
        return
    except Exception as e:
        print(f"An unexpected error occurred with {cif_filepath}: {e}. Skipping.")
        return

    rna_nucleotide_centers = [] # Stores (label, center_coord_array)
    rna_nucleotide_labels = []  # Stores "ChainID_ResidueID" for CSV headers/rows

    print(f"\nProcessing {cif_filepath} for RNA nucleotide centers...")

    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                hetfield, resseq, icode = residue.id
                res_name_original = residue.get_resname().strip()

                if res_name_original in RNA_RESIDUE_NAMES:
                    center = get_residue_center(residue)
                    if center is not None:
                        res_id_str = f"{res_name_original}{resseq}"
                        if icode.strip():
                            res_id_str += icode.strip()
                        
                        full_label = f"{chain_id}_{res_id_str}"
                        
                        rna_nucleotide_centers.append(center) # Store the center coordinates
                        rna_nucleotide_labels.append(full_label)
                    # else:
                    #     print(f"  RNA nucleotide Chain {chain_id}, {res_name_original}{resseq}{icode.strip()} has no atoms to calculate center. Skipping.")
                # else:
                #     if hetfield == ' ':
                #         print(f"  Skipping non-RNA residue: Chain {chain_id}, {res_name_original}{resseq}{icode.strip()}")


    if not rna_nucleotide_centers:
        print(f"  No RNA nucleotides found or centers calculated in {cif_filepath}. No CSV generated.")
        return
    
    if len(rna_nucleotide_centers) < 2:
        print(f"  Only one or zero RNA nucleotides with calculated centers found. Cannot calculate distances. No CSV generated.")
        return

    num_nucleotides = len(rna_nucleotide_centers)
    distance_matrix = [[0.0 for _ in range(num_nucleotides)] for _ in range(num_nucleotides)]

    print(f"  Found {num_nucleotides} RNA nucleotides. Calculating center-to-center distances...")

    for i in range(num_nucleotides):
        center_i = rna_nucleotide_centers[i]
        for j in range(num_nucleotides):
            if i == j:
                distance_matrix[i][j] = 0.0
            else:
                center_j = rna_nucleotide_centers[j]
                # Calculate Euclidean distance between the two center coordinate vectors
                distance = np.linalg.norm(center_i - center_j)
                distance_matrix[i][j] = float(f"{distance:.3f}")

    # Write to CSV
    try:
        with open(output_csv_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + rna_nucleotide_labels)
            for i in range(num_nucleotides):
                writer.writerow([rna_nucleotide_labels[i]] + distance_matrix[i])
        print(f"  Successfully generated RNA center-to-center distance matrix: {output_csv_filepath}")
    except IOError as e:
        print(f"  Error writing CSV file {output_csv_filepath}: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred while writing CSV {output_csv_filepath}: {e}")


def process_cif_folder(cif_folder_path, output_csv_folder_path):
    if not os.path.exists(cif_folder_path):
        print(f"Error: CIF input folder '{cif_folder_path}' not found.")
        return

    if not os.path.exists(output_csv_folder_path):
        os.makedirs(output_csv_folder_path)
        print(f"Created output CSV folder: '{output_csv_folder_path}'")

    for filename in os.listdir(cif_folder_path):
        if filename.lower().endswith(".cif"):
            cif_filepath = os.path.join(cif_folder_path, filename)
            base_filename = os.path.splitext(filename)[0]
            output_csv_filename = f"{base_filename}_rna_center_distances.csv" # Updated filename
            output_csv_filepath = os.path.join(output_csv_folder_path, output_csv_filename)
            
            calculate_rna_nucleotide_center_distances(cif_filepath, output_csv_filepath)
    print("\nFinished processing all CIF files for RNA center-to-center distances.")

if __name__ == "__main__":
    if not os.path.exists(CIF_INPUT_FOLDER):
        os.makedirs(CIF_INPUT_FOLDER)
        print(f"Created CIF input folder: '{CIF_INPUT_FOLDER}'.")
        print(f"Please place your .cif files in the '{CIF_INPUT_FOLDER}' directory and rerun the script.")
    elif not os.listdir(CIF_INPUT_FOLDER) :
        print(f"CIF input folder '{CIF_INPUT_FOLDER}' is empty.")
        print(f"Please place your .cif files in the '{CIF_INPUT_FOLDER}' directory and rerun the script.")
    else:
        process_cif_folder(CIF_INPUT_FOLDER, CSV_OUTPUT_FOLDER)