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

# --- Geometric Feature Extraction ---
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
    center2, _ = get_base_plane_simple(res2)

    if center1 is None or center2 is None or normal1 is None:
        return None

    anchor_dist = np.linalg.norm(center1 - center2)

    inter_anchor_vector = center2 - center1
    if np.allclose(inter_anchor_vector, 0):
        return None

    inter_anchor_vector_norm = inter_anchor_vector / np.linalg.norm(inter_anchor_vector)
    dot_prod = np.clip(np.dot(normal1, inter_anchor_vector_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_prod)
    orientation_angle_deg = min(np.degrees(angle_rad), 180 - np.degrees(angle_rad))

    return [anchor_dist, orientation_angle_deg]

# --- Main Inference Function ---
def predict_interactions_in_cif(model_path, cif_path):
    """
    Loads a CIF file and an XGBoost model, then predicts RNA base interactions.

    For NMR multi-model structures, each conformer is processed independently:
      - Each model gets its own numeric index (matching BioPython model IDs: 0, 1, 2 ...)
      - Each model writes its own CSV:  {structure_id}_m{idx}_predicted_interactions.csv
      - A combined CSV across all models is always written: {structure_id}_predicted_interactions.csv
    """
    # 1. Load the trained XGBoost model
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_path)
        logging.info(f"Successfully loaded XGBoost model from {model_path}")
    except (xgb.core.XGBoostError, IOError) as e:
        logging.error(f"Failed to load model from {model_path}. Error: {e}")
        return

    # 2. Load the PDB/CIF structure
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        structure_id = os.path.splitext(os.path.basename(cif_path))[0]
        structure = parser.get_structure(structure_id, cif_path)
        logging.info(f"Successfully loaded structure '{structure_id}' from {cif_path}")
    except Exception as e:
        logging.error(f"Failed to parse CIF file {cif_path}. Error: {e}")
        return

    valid_resnames = {'A', 'C', 'G', 'U', 'I'}
    class_map = {0: 'interacting', 1: 'non-interacting'}

    # 3. Report ensemble size
    all_models = list(structure)
    num_models = len(all_models)
    is_nmr = num_models > 1
    logging.info(
        f"Structure '{structure_id}' has {num_models} model(s) — "
        f"{'NMR ensemble, processing each conformer independently.' if is_nmr else 'single model.'}"
    )

    all_results = []  # collects per-model DataFrames for the combined output

    # 4. Iterate model-first — each NMR conformer gets its own independent pass
    for model_obj in all_models:
        model_idx = model_obj.id  # BioPython assigns 0, 1, 2, … per conformer

        # Collect valid RNA residues within this model only
        model_residues = [
            residue
            for chain in model_obj
            for residue in chain
            if residue.id[0] == ' ' and residue.get_resname().strip() in valid_resnames
        ]

        num_res = len(model_residues)
        if num_res < 2:
            logging.warning(f"Model {model_idx}: only {num_res} RNA residue(s) found — skipping.")
            continue

        total_pairs = num_res * (num_res - 1) // 2
        logging.info(f"Model {model_idx}: {num_res} RNA residues → {total_pairs:,} candidate pairs.")

        # Extract geometric features for every within-model pair
        pairs_info = []
        feature_vectors = []
        for res1, res2 in itertools.combinations(model_residues, 2):
            features = extract_simple_features(res1, res2)
            if features:
                feature_vectors.append(features)
                pairs_info.append({
                    'model_idx':  model_idx,
                    'chain1':     res1.parent.id,
                    'resname1':   res1.get_resname(),
                    'resid1':     res1.id[1],
                    'chain2':     res2.parent.id,
                    'resname2':   res2.get_resname(),
                    'resid2':     res2.id[1],
                })

        if not feature_vectors:
            logging.warning(f"Model {model_idx}: no valid features extracted — skipping.")
            continue

        # Predict for this model only
        X_pred = np.array(feature_vectors)
        predictions = xgb_model.predict(X_pred)

        results_df = pd.DataFrame(pairs_info)
        results_df['predicted_interaction'] = [class_map[p] for p in predictions]
        features_df = pd.DataFrame(X_pred, columns=["anchor_dist", "orientation_angle_deg"])
        final_df = pd.concat([results_df, features_df], axis=1)
        interacting_df = final_df[final_df['predicted_interaction'] == 'interacting'].copy()

        logging.info(
            f"Model {model_idx}: {len(interacting_df):,} interactions predicted "
            f"(out of {len(feature_vectors):,} valid pairs)."
        )

        # Per-model CSV — only written for NMR ensembles with multiple conformers
        if is_nmr:
            per_model_file = f"{structure_id}_m{model_idx}_predicted_interactions.csv"
            interacting_df.to_csv(per_model_file, index=False)
            logging.info(f"Model {model_idx} results → {per_model_file}")

        all_results.append(interacting_df)

    # 5. Write combined CSV (always produced, across all models)
    if not all_results:
        logging.error("No interactions found in any model.")
        return

    combined_df = pd.concat(all_results, ignore_index=True)
    output_filename = f"{structure_id}_predicted_interactions.csv"
    combined_df.to_csv(output_filename, index=False)
    logging.info(
        f"Done. {len(combined_df):,} total interactions across {len(all_results)} model(s) → {output_filename}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict RNA base interactions in a CIF file using a trained XGBoost model. "
                    "NMR multi-model structures produce one CSV per conformer plus a combined CSV."
    )
    parser.add_argument("cif_path", type=str, help="Path to the input CIF / mmCIF file.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="binary_interaction_classifier_xgb.json",
        help="Path to the trained XGBoost model file (default: %(default)s).",
    )
    args = parser.parse_args()
    predict_interactions_in_cif(args.model_path, args.cif_path)