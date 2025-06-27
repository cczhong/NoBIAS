import os
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from Bio import PDB
import logging
import argparse

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Geometric Feature Extraction (Copied from the data generation script) ---
def get_base_plane_simple(base_residue):
    """Calculates the geometric center and normal vector of a base plane."""
    ring_atom_coords = [a.coord for a in base_residue if a.element in ('C', 'N') and a.name[0] in ('C', 'N')]
    if len(ring_atom_coords) < 3:
        return None, None
    coords = np.array(ring_atom_coords)
    center = np.mean(coords, axis=0)
    coords_centered = coords - center
    try:
        _, _, vh = np.linalg.svd(coords_centered)
        normal = vh[2, :]
        return center, normal / np.linalg.norm(normal)
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return None, None

def extract_simple_features(res1, res2):
    """Extracts anchor distance and orientation angle for a pair of residues."""
    center1, normal1 = get_base_plane_simple(res1)
    center2, _ = get_base_plane_simple(res2) # We only need the normal of the first base

    if center1 is None or center2 is None or normal1 is None:
        return None

    anchor_dist = np.linalg.norm(center1 - center2)

    inter_anchor_vector = center2 - center1
    if np.allclose(inter_anchor_vector, 0):
        return None # Same residue or overlapping centers

    inter_anchor_vector_norm = inter_anchor_vector / np.linalg.norm(inter_anchor_vector)
    dot_prod = np.clip(np.dot(normal1, inter_anchor_vector_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_prod)
    orientation_angle_deg = min(np.degrees(angle_rad), 180 - np.degrees(angle_rad))

    return [anchor_dist, orientation_angle_deg]

# --- Main Inference Function ---
def predict_interactions_in_cif(model_path, cif_path):
    """
    Loads a CIF file and an XGBoost model, then predicts the interaction
    type for every possible pair of RNA bases in the structure.
    """
    # 1. Load the trained XGBoost model
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logging.info(f"Successfully loaded XGBoost model from {model_path}")
    except (xgb.core.XGBoostError, IOError) as e:
        logging.error(f"Failed to load model from {model_path}. Error: {e}")
        return

    # 2. Load the PDB/CIF structure
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        structure_id = os.path.splitext(os.path.basename(cif_path))[0]
        structure = parser.get_structure(structure_id, cif_path)
        logging.info(f"Successfully loaded structure {structure_id} from {cif_path}")
    except Exception as e:
        logging.error(f"Failed to load or parse CIF file {cif_path}. Error: {e}")
        return

    # 3. Collect all standard RNA residues from the structure
    rna_residues = []
    valid_resnames = ['A', 'C', 'G', 'U', 'I']
    for model_obj in structure:
        for chain in model_obj:
            for residue in chain:
                # Standard residues have a blank hetflag ' '
                if residue.id[0] == ' ' and residue.get_resname().strip() in valid_resnames:
                    rna_residues.append(residue)

    num_residues = len(rna_residues)
    if num_residues < 2:
        logging.warning("Found fewer than 2 RNA residues. No pairs to analyze.")
        return

    logging.info(f"Found {num_residues} standard RNA residues. Analyzing all possible pairs...")

    # 4. Generate all unique pairs and extract features
    all_pairs_info = []
    feature_vectors = []

    # Use itertools.combinations to get all unique pairs efficiently
    for res1, res2 in itertools.combinations(rna_residues, 2):
        features = extract_simple_features(res1, res2)
        if features:
            feature_vectors.append(features)
            # Store information to identify the pair later
            all_pairs_info.append({
                'chain1': res1.parent.id,
                'resname1': res1.get_resname(),
                'resid1': res1.id[1],
                'chain2': res2.parent.id,
                'resname2': res2.get_resname(),
                'resid2': res2.id[1],
            })

    if not feature_vectors:
        logging.error("Could not extract features for any residue pairs.")
        return

    # 5. Perform batch prediction
    logging.info(f"Extracted features for {len(feature_vectors)} pairs. Predicting...")
    X_pred = np.array(feature_vectors)
    predictions = model.predict(X_pred)
    # 0='interacting', 1='non-interacting'
    class_map = {0: 'interacting', 1: 'non-interacting'}
    predicted_labels = [class_map[p] for p in predictions]

    # 6. Create and save the results DataFrame
    results_df = pd.DataFrame(all_pairs_info)
    results_df['predicted_interaction'] = predicted_labels

    # Also add the feature values to the output for analysis
    features_df = pd.DataFrame(X_pred, columns=["anchor_dist", "orientation_angle_deg"])
    final_df = pd.concat([results_df, features_df], axis=1)

    # Filter to show only the interesting 'interacting' pairs
    interacting_df = final_df[final_df['predicted_interaction'] == 'interacting'].copy()

    # Save results to a CSV file
    output_filename = f"{structure_id}_predicted_interactions.csv"
    interacting_df.to_csv(output_filename, index=False)
    logging.info(f"Analysis complete. Found {len(interacting_df)} predicted interactions.")
    logging.info(f"Results saved to: {output_filename}")


if __name__ == '__main__':
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Predict all residue interactions in a CIF file using a trained XGBoost model.")
    
    # Required positional argument for the input file
    parser.add_argument("cif_path", type=str, help="Path to the input CIF file to analyze.")
    
    # Optional argument for the model path, with a new default value
    parser.add_argument(
        "--model-path",
        type=str,
        default="binary_interaction_classifier_xgb.json",
        help="Path to the trained XGBoost model file (default: %(default)s)."
    )

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    predict_interactions_in_cif(args.model_path, args.cif_path)