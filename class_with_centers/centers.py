import os
import csv
import numpy as np
from Bio.PDB import MMCIFParser, PDBExceptions
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import concurrent.futures
from tqdm import tqdm

CIF_INPUT_FOLDER = "../../../cif" # change to your csv location 
CSV_OUTPUT_FOLDER = "rna_center_distance_csvs_3_robust"

# Low num of cores because it's fast enough, and the second half mem ballons
NUM_CORES = 4

# This list is a little random, emperically expanded
BASE_ATOMS = {
    'A': {'N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C6', 'N6', 'C2', 'N1'}, 
    'G': {'N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C6', 'O6', 'N1', 'C2', 'N2'},
    'C': {'N1', 'C5', 'C6', 'C2', 'O2', 'N3', 'C4', 'N4'}, 
    'U': {'N1', 'C5', 'C6', 'C2', 'O2', 'N3', 'C4', 'O4'},
}

for base in list(BASE_ATOMS.keys()): BASE_ATOMS['R' + base] = BASE_ATOMS[base] # Add aliases

def get_auth_to_label_chain_map(cif_filepath):
    """Parses CIF to map author chain IDs to the PDB's label_asym_id."""
    chain_map = {}
    try:
        mmcif_dict = MMCIF2Dict(cif_filepath)
        auth_ids = mmcif_dict.get("_atom_site.auth_asym_id", [])
        label_ids = mmcif_dict.get("_atom_site.label_asym_id", [])
        # Use a dictionary to ensure a unique mapping from auth to label
        chain_map = dict(zip(auth_ids, label_ids))
    except Exception:
        pass
    return chain_map

def get_residue_center(residue):
    """Calculates the geometric center of the base atoms."""
    res_name = residue.get_resname().strip()
    # If res_name is not a standard base, default to using all atoms for the center calculation, might change this behaviour later, unsure if this has effects
    target_atom_names = BASE_ATOMS.get(res_name, {a.id for a in residue})
    atom_coords = [atom.get_coord() for atom in residue if atom.id in target_atom_names]
    return np.mean(atom_coords, axis=0) if atom_coords else None

def is_nucleotide(residue):
    """Determines if a residue is a nucleotide by checking for key backbone or sugar atoms."""
    return any(atom_id in residue for atom_id in ["P", "O5'", "C5'", "C4'", "O4'", "C1'"])

def create_residue_label(chain_label, residue):
    """Creates a standardized residue label: Chain_ResNameResSeq[Icode]"""
    res_name = residue.get_resname().strip()
    _, res_seq, i_code = residue.id
    i_code_str = i_code.strip() # Use strip to remove whitespace
    return f"{chain_label}_{res_name}{res_seq}{i_code_str}"

def calculate_distances_for_cif(cif_filepath, output_csv_filepath):
    """Processes a single CIF file to generate an RNA distance matrix."""
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("s", cif_filepath)
        chain_map = get_auth_to_label_chain_map(cif_filepath)
    except (PDBExceptions.PDBConstructionException, FileNotFoundError, Exception):
        return None # Gracefully fail on parsing errors, undo for debug

    nucleotides = []
    for model in structure:
        for chain in model:
            chain_label = chain_map.get(chain.id, chain.id)
            for residue in chain:
                if is_nucleotide(residue):
                    center = get_residue_center(residue)
                    if center is not None:
                        label = create_residue_label(chain_label, residue)
                        nucleotides.append({'label': label, 'center': center})
    
    if len(nucleotides) < 2: return None

    num_nuc = len(nucleotides)
    labels = [n['label'] for n in nucleotides]
    centers = np.array([n['center'] for n in nucleotides])
    
    distance_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

    try:
        with open(output_csv_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + labels)
            for i in range(num_nuc):
                formatted_row = [f"{dist:.3f}" for dist in distance_matrix[i]]
                writer.writerow([labels[i]] + formatted_row)
    except (IOError, Exception):
        return None
    return cif_filepath

def process_cif_folder(cif_folder, output_folder, max_workers):
    """Processes all CIF files in a folder using a specified number of workers."""
    os.makedirs(output_folder, exist_ok=True)
    
    all_files = os.listdir(cif_folder)
    tasks = []
    for filename in all_files:
        if filename.lower().endswith(".cif"):
            cif_path = os.path.join(cif_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_rna_center_distances.csv")
            tasks.append((cif_path, output_path))
    
    if not tasks:
        print(f"No .cif files found in '{cif_folder}'.")
        return

    print(f"Found {len(tasks)} files to process using a max of {max_workers} core(s).")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks), desc="Generating Robust Distance Matrices") as pbar:
            futures = {executor.submit(calculate_distances_for_cif, cif, out): cif for cif, out in tasks}
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

if __name__ == "__main__":
    process_cif_folder(CIF_INPUT_FOLDER, CSV_OUTPUT_FOLDER, NUM_CORES)
    print(f"\nFinished generating robust distance matrices in '{CSV_OUTPUT_FOLDER}'.")