import os
import csv
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

from Bio.PDB import MMCIFParser

# --- Configuration ---
DISTANCE_MATRIX_FOLDER = "rna_center_distance_csvs_2"
BASE_PAIRS_FOLDER = "base_pairs"
STACKS_BY_MOL_FOLDER = "stacks_by_mol"
CIF_FOLDER = "../cif"
ANALYSIS_OUTPUT_FOLDER_ROOT = "threshold_analysis_structured_v10"  # Incremented version

THRESHOLDS = [float(x) for x in range(1, 15)]
NUM_CORES = 18

def get_resolution_from_cif(cif_filepath):
    """Parses a CIF file to get its resolution."""
    if not os.path.exists(cif_filepath):
        return None
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("mol", cif_filepath)
        resolution = structure.header.get("resolution")
        return float(resolution) if resolution is not None else None
    except Exception as e:
        # Mostly here for errors and misparses
        print(f"Could not parse resolution from {cif_filepath}: {e}")


def load_known_base_pairs_for_molecule(bp_filepath):
    known_items_with_info = []
    original_header = []
    if not os.path.exists(bp_filepath):
        return known_items_with_info, original_header
    try:
        with open(bp_filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            original_header = reader.fieldnames if reader.fieldnames else []
            for row_number, row in enumerate(reader, 1):
                try:
                    required_keys = ["chain1", "pos1", "nt1", "chain2", "pos2", "nt2"]
                    if not all(
                        key in row and row[key] and row[key].strip()
                        for key in required_keys
                    ):
                        continue
                    chain1, pos1, nt1 = (
                        row["chain1"].strip(),
                        row["pos1"].strip(),
                        row["nt1"].strip().upper(),
                    )
                    chain2, pos2, nt2 = (
                        row["chain2"].strip(),
                        row["pos2"].strip(),
                        row["nt2"].strip().upper(),
                    )
                    label1, label2 = f"{chain1}_{nt1}{pos1}", f"{chain2}_{nt2}{pos2}"
                    pair_labels_fs = frozenset([label1, label2])
                    known_items_with_info.append(
                        {
                            "pair_fs": pair_labels_fs,
                            "original_row": row,
                            "id": row_number,
                        }
                    )
                except Exception:
                    pass
    except Exception as e:
        print(f"Error reading BP file {bp_filepath}: {e}")
        return [], []
    return known_items_with_info, original_header


def load_known_base_stacks_for_molecule(mol_name_filter):
    mol_stack_filepath = os.path.join(STACKS_BY_MOL_FOLDER, f"{mol_name_filter}.csv")
    if not os.path.exists(mol_stack_filepath):
        return [], []
    known_stacks_with_info = []
    original_header = []
    try:
        mol_specific_df = pd.read_csv(mol_stack_filepath, dtype=str)
        original_header = list(mol_specific_df.columns)
        for index, row in mol_specific_df.iterrows():
            pair_str = str(row.get("base/residue_name", "")).strip()
            if "-" not in pair_str:
                continue
            nt1, nt2 = (
                pair_str.split("-", 1)[0].strip().upper(),
                pair_str.split("-", 1)[1].strip().upper(),
            )
            c1, p1_raw = (
                str(row.get("chain1", "")).strip(),
                str(row.get("position1", "")).strip(),
            )
            c2, p2_raw = (
                str(row.get("chain2", "")).strip(),
                str(row.get("position2", "")).strip(),
            )
            if not all([c1, p1_raw, nt1, c2, p2_raw, nt2]):
                continue
            p1, p2 = str(int(float(p1_raw))), str(int(float(p2_raw)))
            l1, l2 = f"{c1}_{nt1}{p1}", f"{c2}_{nt2}{p2}"
            fs = frozenset([l1, l2])
            known_stacks_with_info.append(
                {"pair_fs": fs, "original_row": row.to_dict(), "id": index}
            )
    except Exception as e:
        print(f"Error processing stack file {mol_stack_filepath}: {e}")
        return [], []
    return known_stacks_with_info, original_header


def analyze_and_log_removals(
    distance_matrix_filepath,
    known_items_data_list,
    original_csv_header,
    thresholds_list,
    output_txt_filepath,
    item_type_name="Items",
):
    counts_remaining_at_threshold = []
    if not os.path.exists(distance_matrix_filepath):
        for _ in thresholds_list:
            counts_remaining_at_threshold.append(0)
        return counts_remaining_at_threshold, 0
    try:
        dist_df = pd.read_csv(distance_matrix_filepath, index_col=0)
    except Exception as e:
        print(f"Error reading dist matrix {distance_matrix_filepath}: {e}")
        for _ in thresholds_list:
            counts_remaining_at_threshold.append(0)
        return counts_remaining_at_threshold, 0
    all_labels_in_matrix = set(dist_df.index).union(set(dist_df.columns))
    calculated_distances, initially_missing = {}, []
    for item in known_items_data_list:
        fs = item["pair_fs"]
        lbls = list(fs)
        l1, l2 = (lbls[0], lbls[0]) if len(lbls) == 1 else (lbls[0], lbls[1])
        dist_val = float("inf")
        if l1 not in all_labels_in_matrix or l2 not in all_labels_in_matrix:
            initially_missing.append(item)
        else:
            try:
                val = dist_df.loc[l1, l2]
                if isinstance(val, pd.Series):
                    dist_val = (
                        float(val.iloc[0])
                        if not val.empty and not pd.isna(val.iloc[0])
                        else float("inf")
                    )
                elif not pd.isna(val):
                    dist_val = float(val)
                if pd.isna(dist_val) or dist_val == float("inf"):
                    if item not in initially_missing:
                        initially_missing.append(item)
                    dist_val = float("inf")
            except Exception:
                if item not in initially_missing:
                    initially_missing.append(item)
        calculated_distances[fs] = dist_val
    initially_missing = [
        item
        for item in known_items_data_list
        if calculated_distances.get(item["pair_fs"], float("inf")) == float("inf")
    ]
    num_valid = len(known_items_data_list) - len(initially_missing)
    os.makedirs(os.path.dirname(output_txt_filepath), exist_ok=True)
    with open(output_txt_filepath, "w") as txt_f:
        # ... (logging logic is unchanged)
        txt_f.write(f"{item_type_name} Removal Log\n")
        if original_csv_header:
            txt_f.write(
                f"Original CSV Header: {', '.join(map(str, original_csv_header))}\n"
            )
        txt_f.write(
            f"Total {item_type_name.lower()} loaded: {len(known_items_data_list)}\n"
        )
        txt_f.write(f"Initially missing/invalid distances: {len(initially_missing)}\n")
        txt_f.write(f"Valid for thresholding: {num_valid}\n\n")
        if initially_missing:
            txt_f.write(
                f"--- {len(initially_missing)} {item_type_name.upper()} WITH COMPONENTS NOT IN DISTANCE MATRIX ---\n"
            )
            for item_i in initially_missing:
                r_d = item_i["original_row"]
                r_v = (
                    [str(r_d.get(h, "")) for h in original_csv_header]
                    if original_csv_header
                    else [str(v) for v in r_d.values()]
                )
                txt_f.write(f"  {', '.join(r_v)}\n")
            txt_f.write("\n")
        for thold in thresholds_list:
            rem, rem_info = 0, []
            for item_d in known_items_data_list:
                d = calculated_distances.get(item_d["pair_fs"], float("inf"))
                if d == float("inf") or pd.isna(d):
                    continue
                if float(d) <= thold:
                    rem += 1
                else:
                    rem_info.append((item_d, float(d)))
            counts_remaining_at_threshold.append(rem)
            txt_f.write(f"--- THRESHOLD: <= {thold:.1f} A ---\n")
            txt_f.write(
                f"{item_type_name} meeting threshold (from {num_valid} valid): {rem}\n"
            )
            if num_valid > 0:
                txt_f.write(f"  Percentage: {(rem/num_valid)*100:.2f}%\n")
            txt_f.write(f"{item_type_name} 'removed' (dist > {thold:.1f} A):\n")
            if rem_info:
                for item_i, d_v in sorted(rem_info, key=lambda x: x[1]):
                    r_d_ = item_i["original_row"]
                    r_v_ = (
                        [str(r_d_.get(h, "")) for h in original_csv_header]
                        if original_csv_header
                        else [str(v) for v in r_d_.values()]
                    )
                    txt_f.write(f"  {', '.join(r_v_)} (Calc Dist: {d_v:.3f} A)\n")
            else:
                txt_f.write("  None removed.\n")
            txt_f.write("\n")
    return counts_remaining_at_threshold, num_valid


def plot_remaining_items_graph_percent(
    thresholds,
    counts_remaining,
    total_processable,
    item_name_for_title,
    item_type_suffix,
    output_graph_folder,
):
    os.makedirs(output_graph_folder, exist_ok=True)
    plt.figure(figsize=(12, 7))
    percentages = [
        (c / total_processable) * 100 if total_processable > 0 else 0
        for c in counts_remaining
    ]
    thold_labels = [f"{int(t)}" for t in thresholds]
    bar_color = "skyblue" if item_type_suffix == "bp" else "mediumseagreen"
    plt.bar(thold_labels, percentages, color=bar_color, width=0.8)
    plt.xlabel("Distance Threshold (A)")
    plt.ylabel(f"Known {item_type_suffix.capitalize()} Remaining (%)")
    plt.title(
        f"% Known {item_type_suffix.capitalize()} Remaining vs. Threshold for {item_name_for_title}"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_filename = f"{item_name_for_title}_{item_type_suffix}_percent_remaining.png"
    plot_filepath = os.path.join(output_graph_folder, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

def find_first_loss_threshold(counts_remaining, num_valid, thresholds):
    if num_valid == 0:
        return None
    highest_threshold_with_a_loss = None
    for count, threshold in zip(counts_remaining, thresholds):
        if count < num_valid:
            highest_threshold_with_a_loss = threshold
    return highest_threshold_with_a_loss

def process_molecule(distance_matrix_filepath):
    LOGS_FOLDER = os.path.join(ANALYSIS_OUTPUT_FOLDER_ROOT, "removal_logs")
    BP_GRAPHS_FOLDER = os.path.join(ANALYSIS_OUTPUT_FOLDER_ROOT, "base_pair_graphs")
    STACK_GRAPHS_FOLDER = os.path.join(ANALYSIS_OUTPUT_FOLDER_ROOT, "base_stack_graphs")

    mol_name = os.path.basename(distance_matrix_filepath).replace(
        "_rna_center_distances.csv", ""
    )

    first_loss_bp = None
    first_loss_stack = None

    # Get resolution
    cif_filepath = os.path.join(CIF_FOLDER, f"{mol_name}.cif")
    resolution = get_resolution_from_cif(cif_filepath)

    # Process Base Pairs
    bp_filepath = os.path.join(BASE_PAIRS_FOLDER, f"{mol_name}.csv")
    if os.path.exists(bp_filepath):
        known_bp_data, bp_csv_header = load_known_base_pairs_for_molecule(bp_filepath)
        if known_bp_data:
            bp_log_filepath = os.path.join(
                LOGS_FOLDER, f"{mol_name}_bp_removal_log.txt"
            )
            bp_counts, bp_valid_count = analyze_and_log_removals(
                distance_matrix_filepath,
                known_bp_data,
                bp_csv_header,
                THRESHOLDS,
                bp_log_filepath,
                "Base Pairs",
            )
            if bp_valid_count > 0:
                plot_remaining_items_graph_percent(
                    THRESHOLDS,
                    bp_counts,
                    bp_valid_count,
                    mol_name,
                    "bp",
                    BP_GRAPHS_FOLDER,
                )
                first_loss_bp = find_first_loss_threshold(
                    bp_counts, bp_valid_count, THRESHOLDS
                )

    # Process Base Stacks
    known_stacks_data, stack_csv_header = load_known_base_stacks_for_molecule(mol_name)
    if known_stacks_data:
        stack_log_filepath = os.path.join(
            LOGS_FOLDER, f"{mol_name}_stack_removal_log.txt"
        )
        stack_counts, stack_valid_count = analyze_and_log_removals(
            distance_matrix_filepath,
            known_stacks_data,
            stack_csv_header,
            THRESHOLDS,
            stack_log_filepath,
            "Base Stacks",
        )
        if stack_valid_count > 0:
            plot_remaining_items_graph_percent(
                THRESHOLDS,
                stack_counts,
                stack_valid_count,
                mol_name,
                "stack",
                STACK_GRAPHS_FOLDER,
            )
            first_loss_stack = find_first_loss_threshold(
                stack_counts, stack_valid_count, THRESHOLDS
            )

    print(f"Finished: {mol_name} (Res: {resolution if resolution else 'N/A'})")
    return mol_name, first_loss_bp, first_loss_stack, resolution


if __name__ == "__main__":
    for folder in [
        DISTANCE_MATRIX_FOLDER,
        CIF_FOLDER,
        STACKS_BY_MOL_FOLDER,
        BASE_PAIRS_FOLDER,
    ]:
        if not os.path.isdir(folder):
            print(f"Error: Required folder '{folder}' was not found.")
            exit()

    os.makedirs(ANALYSIS_OUTPUT_FOLDER_ROOT, exist_ok=True)
    os.makedirs(
        os.path.join(ANALYSIS_OUTPUT_FOLDER_ROOT, "removal_logs"), exist_ok=True
    )
    os.makedirs(
        os.path.join(ANALYSIS_OUTPUT_FOLDER_ROOT, "base_pair_graphs"), exist_ok=True
    )
    os.makedirs(
        os.path.join(ANALYSIS_OUTPUT_FOLDER_ROOT, "base_stack_graphs"), exist_ok=True
    )

    print(
        f"Analysis outputs will be saved in subfolders under: {ANALYSIS_OUTPUT_FOLDER_ROOT}"
    )

    dist_matrix_files = [
        os.path.join(DISTANCE_MATRIX_FOLDER, f)
        for f in os.listdir(DISTANCE_MATRIX_FOLDER)
        if f.endswith("_rna_center_distances.csv")
    ]

    if not dist_matrix_files:
        print(f"\nNo distance matrices found in '{DISTANCE_MATRIX_FOLDER}'. Exiting.")
        exit()

    print(
        f"\nFound {len(dist_matrix_files)} molecules to process using {NUM_CORES} cores."
    )

    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(process_molecule, dist_matrix_files)

    # Aggregation logic by resolution category
    print("\n--- Analysis Complete. Aggregating results... ---")

    resolution_categories = {
        "0-2.0 A": {"bp_losses": [], "stack_losses": [], "mol_count": 0},
        "2.0-3.5 A": {"bp_losses": [], "stack_losses": [], "mol_count": 0},
        "3.5+ A": {"bp_losses": [], "stack_losses": [], "mol_count": 0},
        "Unknown/No Resolution": {"bp_losses": [], "stack_losses": [], "mol_count": 0},
    }

    # Sort results into categories
    for mol_name, bp_loss, stack_loss, resolution in results:
        category_key = None
        if resolution is None:
            category_key = "Unknown/No Resolution"
        elif resolution <= 2.0:
            category_key = "0-2.0 A"
        elif resolution <= 3.5:
            category_key = "2.0-3.5 A"
        else:
            category_key = "3.5+ A"

        if category_key:
            resolution_categories[category_key]["mol_count"] += 1
            if bp_loss is not None:
                resolution_categories[category_key]["bp_losses"].append(bp_loss)
            if stack_loss is not None:
                resolution_categories[category_key]["stack_losses"].append(stack_loss)

    # Print categorized results
    print("\n--- Average First Loss Threshold by Resolution ---")
    for category, data in resolution_categories.items():
        if data["mol_count"] == 0:
            continue

        print(f"\n--- Category: {category} ({data['mol_count']} structures) ---")

        # Base Pairs
        bp_losses = data["bp_losses"]
        if bp_losses:
            avg_bp_loss = sum(bp_losses) / len(bp_losses)
            print(
                f"Base Pairs:  {avg_bp_loss:.2f} A (average threshold to lose the first BP, based on {len(bp_losses)} molecules)"
            )
        else:
            print("Base Pairs:  No data available in this category.")

        # Base Stacks
        stack_losses = data["stack_losses"]
        if stack_losses:
            avg_stack_loss = sum(stack_losses) / len(stack_losses)
            print(
                f"Base Stacks: {avg_stack_loss:.2f} A (average threshold to lose the first stack, based on {len(stack_losses)} molecules)"
            )
        else:
            print("Base Stacks: No data available in this category.")

    print(f"\nFinished all analyses. Processed {len(dist_matrix_files)} structures.")
