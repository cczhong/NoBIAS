import os
import csv
import pandas as pd
import numpy as np
from Bio import PDB
import random
import logging

# --- Configuration ---
CIF_FILES_DIR = "cif"
BASE_PAIRS_INPUT_FOLDER = "base_pairs"
GLOBAL_STACK_INPUT_FILE = "output_fr3.csv"

# <<< UPDATED: New output file for the binary dataset >>>
OUTPUT_DATASET_CSV = "ann_dataset_binary.csv"
SAMPLES_PER_CLASS = 10000 # Let's get more samples for the two classes

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Simplified Feature Extraction ---
def get_base_plane_simple(base_residue):
    ring_atom_coords = [a.coord for a in base_residue if a.element in ('C', 'N') and a.name[0] in ('C', 'N')]
    if len(ring_atom_coords) < 3: return None, None
    coords = np.array(ring_atom_coords)
    center = np.mean(coords, axis=0)
    coords_centered = coords - center
    try:
        _, _, vh = np.linalg.svd(coords_centered); normal = vh[2, :]
        return center, normal / np.linalg.norm(normal)
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return None, None

def extract_simple_features(res1, res2):
    """Extracts only anchor distance and orientation angle."""
    center1, normal1 = get_base_plane_simple(res1)
    center2, normal2 = get_base_plane_simple(res2)
    if center1 is None or center2 is None or normal1 is None:
        return None

    # Feature 1: Anchor Distance (using geometric centers)
    anchor_dist = np.linalg.norm(center1 - center2)
    
    # Feature 2: Orientation Angle
    inter_anchor_vector = center2 - center1
    if np.allclose(inter_anchor_vector, 0): return None
    inter_anchor_vector_norm = inter_anchor_vector / np.linalg.norm(inter_anchor_vector)
    dot_prod = np.clip(np.dot(normal1, inter_anchor_vector_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_prod)
    orientation_angle_deg = min(np.degrees(angle_rad), 180 - np.degrees(angle_rad))

    return [anchor_dist, orientation_angle_deg]

# --- Data Loading (Unchanged, robust versions) ---
def load_structure_bio(cif_filepath): # ... (identical to previous versions)
    parser = PDB.MMCIFParser(QUIET=True);
    try: return parser.get_structure(os.path.splitext(os.path.basename(cif_filepath))[0], cif_filepath);
    except Exception: return None;

def load_known_base_pairs_for_molecule_ann(bp_filepath): # ... (identical)
    known_pairs = [];
    if not os.path.exists(bp_filepath): return known_pairs;
    try:
        with open(bp_filepath, 'r', newline='') as f:
            reader = csv.DictReader(f);
            for row in reader:
                try:
                    c1, p1s, n1, c2, p2s, n2 = row['chain1'].strip(), row['pos1'].strip(), row['nt1'].strip().upper(), row['chain2'].strip(), row['pos2'].strip(), row['nt2'].strip().upper();
                    if not (p1s.isdigit() and p2s.isdigit()): continue;
                    res1, res2 = (c1, int(p1s), n1), (c2, int(p2s), n2);
                    if res1 != res2: known_pairs.append(frozenset([res1, res2]));
                except (KeyError, ValueError): continue;
    except Exception: pass;
    return known_pairs;

def load_known_base_stacks_for_molecule_ann(mol_name_filter, global_stack_df): # ... (identical)
    known_stacks = [];
    if global_stack_df is None: return known_stacks;
    df = global_stack_df[global_stack_df['mol_name'].astype(str) == str(mol_name_filter)];
    for _, row in df.iterrows():
        try:
            c1, p1s, n1 = row.get('chain1','').strip(), str(row.get('position1','')).strip(), row.get('nt1','').strip().upper();
            c2, p2s, n2 = row.get('chain2','').strip(), str(row.get('position2','')).strip(), row.get('nt2','').strip().upper();
            if not n1 or not n2:
                parts = str(row.get('base/residue_name', '')).strip().split('-', 1);
                if len(parts) == 2:
                    if not n1 and parts[0]: n1 = parts[0].strip().upper()[0];
                    if not n2 and parts[1]: n2 = parts[1].strip().upper()[0];
            if not all([c1, p1s, n1, c2, p2s, n2, p1s.isdigit(), p2s.isdigit()]): continue;
            res1, res2 = (c1, int(p1s), n1), (c2, int(p2s), n2);
            if res1 != res2: known_stacks.append(frozenset([res1, res2]));
        except (KeyError, ValueError, IndexError): continue;
    return known_stacks;


# --- Main Dataset Building Logic for Binary Classification ---
def build_dataset():
    all_data = []
    # <<< UPDATED: Counters for binary classification >>>
    counts = {'interacting': 0, 'non-interacting': 0}
    
    global_stack_df = pd.read_csv(GLOBAL_STACK_INPUT_FILE, dtype=str) if os.path.exists(GLOBAL_STACK_INPUT_FILE) else None
    cif_files = [f for f in os.listdir(CIF_FILES_DIR) if f.lower().endswith(".cif")]
    random.shuffle(cif_files)

    for cif_filename in cif_files:
        if all(c >= SAMPLES_PER_CLASS for c in counts.values()): break
        mol_name = os.path.splitext(cif_filename)[0]
        logging.info(f"--- Processing {mol_name} | Status: {counts} ---")
        structure = load_structure_bio(os.path.join(CIF_FILES_DIR, cif_filename))
        if not structure or 0 not in structure: continue
        model = structure[0]
        
        processed_residue_sets_in_mol = set()

        # Combine pairs and stacks into the 'interacting' class
        if counts['interacting'] < SAMPLES_PER_CLASS:
            pairs = load_known_base_pairs_for_molecule_ann(os.path.join(BASE_PAIRS_INPUT_FOLDER, f"{mol_name}.csv"))
            stacks = load_known_base_stacks_for_molecule_ann(mol_name, global_stack_df)
            all_interactions = pairs + stacks
            
            for interaction_fs in all_interactions:
                if counts['interacting'] >= SAMPLES_PER_CLASS: break
                res_info1, res_info2 = list(interaction_fs)
                try:
                    res1_obj, res2_obj = model[res_info1[0]][(' ', res_info1[1], ' ')], model[res_info2[0]][(' ', res_info2[1], ' ')]
                    fs_id = frozenset([res1_obj.get_full_id(), res2_obj.get_full_id()])
                    if fs_id in processed_residue_sets_in_mol: continue
                    
                    features = extract_simple_features(res1_obj, res2_obj)
                    if features:
                        all_data.append(features + ["interacting"]) # <<< NEW LABEL
                        counts['interacting'] += 1
                        processed_residue_sets_in_mol.add(fs_id)
                except KeyError: continue
        
        # Collect non-interacting class
        if counts['non-interacting'] < SAMPLES_PER_CLASS:
            rna_residues = [r for c in model for r in c if r.id[0]==' ' and r.resname.strip() in ['A','C','G','U','I']]
            if len(rna_residues) >= 2:
                for _ in range(len(rna_residues) * 4):
                    if counts['non-interacting'] >= SAMPLES_PER_CLASS: break
                    res1, res2 = random.sample(rna_residues, 2)
                    fs_id = frozenset([res1.get_full_id(), res2.get_full_id()])
                    if fs_id not in processed_residue_sets_in_mol:
                        features = extract_simple_features(res1, res2)
                        if features:
                            all_data.append(features + ['non-interacting']) # <<< NEW LABEL
                            counts['non-interacting'] += 1
                            processed_residue_sets_in_mol.add(fs_id)

    logging.info(f"Final Collection: {counts}")
    
    # <<< UPDATED: Final DataFrame columns >>>
    df = pd.DataFrame(all_data, columns=["anchor_dist", "orientation_angle_deg", "label"])
    df.to_csv(OUTPUT_DATASET_CSV, index=False)
    logging.info(f"Binary feature dataset saved to {OUTPUT_DATASET_CSV}")

if __name__ == "__main__":
    build_dataset()