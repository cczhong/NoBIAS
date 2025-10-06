import os
import csv
import sys
from tqdm import tqdm
import concurrent.futures
import logging
import numpy as np
import re
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

CENTER_CSV_DIR = "rec_pipe/rna_center_distance_csvs_3_robust" # change folder
PAIRING_CSV_DIR = "Parsed_union_pairs/Parsed_union_pairs" # should be the exact ones you gave me
STACKING_CSV_DIR = "parsed_union_stackings/parsed_union_stackings"
CIF_FILES_DIR = "../../cif" # change folder
OUTPUT_DIR = "analysis_results_discovery_20A_3"
DISTANCE_THRESHOLD = 20.0
NUM_CORES = 10
BATCH_WRITE_SIZE = 10

ALL_PASSED_INTERACTIONS_FILE = os.path.join(OUTPUT_DIR, f"all_interactions_under_{int(DISTANCE_THRESHOLD)}A.csv")
MISSED_KNOWN_INTERACTIONS_FILE = os.path.join(OUTPUT_DIR, "missed_known_interactions.csv")
SUMMARY_LOG_FILE = os.path.join(OUTPUT_DIR, "summary_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(SUMMARY_LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

def get_chain_translation_map(cif_filepath):
    """
    Creates a case-insensitive map from author chain IDs to PDB label chain IDs.
    """
    auth_to_label = {}
    try:
        mmcif_dict = MMCIF2Dict(cif_filepath)
        auth_ids = mmcif_dict.get("_atom_site.auth_asym_id", [])
        label_ids = mmcif_dict.get("_atom_site.label_asym_id", [])
        auth_to_label = {auth.upper(): label.upper() for auth, label in zip(auth_ids, label_ids)} # thought the I to i res thing might have been a case issue, didnt help much
    except Exception as e:
        print(f"WARNING: Could not create chain map for {os.path.basename(cif_filepath)}: {e}", file=sys.stderr, flush=True)
    return auth_to_label

def check_recall_against_known(pdb_id, interaction_dir, discovered_set, chain_map):
    def _parse_residue_id_local(res_id_str):
        match = re.match(r"(-?\d+)([A-Za-z]*)", str(res_id_str).strip())
        if match:
            num_part, icode_part = match.groups()
            return int(num_part), icode_part
        raise ValueError(f"Could not parse residue ID: '{res_id_str}'")

    stats = {"total_known": 0, "recalled": 0, "missed": 0, "missed_lines": []}
    interaction_csv_path = os.path.join(interaction_dir, f"{pdb_id}.csv")
    if not os.path.exists(interaction_csv_path):
        return stats

    with open(interaction_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_known"] += 1
            try:
                auth_chain1 = row["chain1"].upper()
                auth_chain2 = row["chain2"].upper()
                
                label_chain1 = chain_map.get(auth_chain1, auth_chain1)
                label_chain2 = chain_map.get(auth_chain2, auth_chain2)
                
                pos1_key, pos2_key = ("pos1", "pos2") if "pos1" in row else ("position1", "position2")
                res_num1, i_code1 = _parse_residue_id_local(row[pos1_key])
                res_num2, i_code2 = _parse_residue_id_local(row[pos2_key])
                
                key1 = (label_chain1, res_num1, i_code1)
                key2 = (label_chain2, res_num2, i_code2)
                canonical_key = tuple(sorted((key1, key2)))

                if canonical_key in discovered_set:
                    stats["recalled"] += 1
                else:
                    stats["missed"] += 1
                    missed_info = [pdb_id, row["chain1"], f"{res_num1}{i_code1}", row["chain2"], f"{res_num2}{i_code2}"]
                    stats["missed_lines"].append(missed_info)

            except (KeyError, ValueError):
                continue
    return stats


def process_pdb_id(center_csv_path, cif_dir):
    def _parse_residue_id(res_id_str):
        match = re.match(r"(-?\d+)([A-Za-z]*)", str(res_id_str).strip())
        if match:
            num_part, icode_part = match.groups()
            return int(num_part), icode_part
        raise ValueError(f"Could not parse residue ID: '{res_id_str}'")

    def _parse_label_to_canonical_key(label):
        try:
            chain_id, nt_info = label.split('_', 1)
            chain_id = chain_id.upper()
            
            res_id_str = re.sub(r"^[A-Z]+", "", nt_info)
            res_num, i_code = _parse_residue_id(res_id_str)
            return (chain_id, res_num, i_code)
        except (ValueError, IndexError):
            raise ValueError(f"Could not parse label: '{label}'")
    
    pdb_id = os.path.basename(center_csv_path).split('_')[0]
    
    cif_filepath = os.path.join(cif_dir, f"{pdb_id}.cif")
    if not os.path.exists(cif_filepath):
        print(f"WARNING: Source CIF file not found for chain mapping: {cif_filepath}", file=sys.stderr, flush=True)
        return None, None, None
    chain_map = get_chain_translation_map(cif_filepath)

    all_passed_lines = []
    discovered_set = set()
    
    try:
        with open(center_csv_path, 'r') as f:
            reader = csv.reader(f)
            try:
                labels = next(reader)[1:]
            except StopIteration:
                return None, None, None

            for i, row in enumerate(reader):
                distances = row[1:]
                for j in range(i + 1, len(distances)):
                    try:
                        distance = float(distances[j])
                        if distance <= DISTANCE_THRESHOLD:
                            label1_full, label2_full = labels[i], labels[j]
                            
                            key1 = _parse_label_to_canonical_key(label1_full)
                            key2 = _parse_label_to_canonical_key(label2_full)
                            canonical_key = tuple(sorted((key1, key2)))
                            discovered_set.add(canonical_key)
                            
                            chain1, resnum1, icode1 = key1
                            chain2, resnum2, icode2 = key2
                            all_passed_lines.append([pdb_id, chain1, f"{resnum1}{icode1}", chain2, f"{resnum2}{icode2}", "N", "N", f"{distance:.3f}"])

                    except (ValueError, IndexError):
                        continue
            
    except Exception as e:
        print(f"ERROR: FATAL_PROCESS_FAIL - Processing {pdb_id}.csv: {e}", file=sys.stderr, flush=True)
        return None, None, None

    pair_recall_stats = check_recall_against_known(pdb_id, PAIRING_CSV_DIR, discovered_set, chain_map)
    stack_recall_stats = check_recall_against_known(pdb_id, STACKING_CSV_DIR, discovered_set, chain_map)
    
    return all_passed_lines, pair_recall_stats, stack_recall_stats

def print_recall_summary(final_stats, interaction_type):
    total_known = final_stats['total_known']
    if total_known == 0: return
    recalled = final_stats['recalled']
    missed = final_stats['missed']
    recall_percentage = (recalled / total_known) * 100 if total_known > 0 else 0
    logging.info("\n" + "=" * 50)
    logging.info(f"RECALL SUMMARY FOR: {interaction_type.upper()}")
    logging.info("=" * 50)
    logging.info(f"Total Known Interactions (in Union files): {total_known:,}")
    logging.info(f"  - Recalled (Found by distance search):     {recalled:,}")
    logging.info(f"  - Missed (Not found by distance search):   {missed:,}")
    logging.info("=" * 50)
    logging.info(f"RECALL PERCENTAGE:                           {recall_percentage:.2f}%")
    logging.info("=" * 50 + "\n")

def write_batch_data(passed_lines, missed_pairs, missed_stacks):
    try:
        if passed_lines:
            with open(ALL_PASSED_INTERACTIONS_FILE, 'a', newline='') as f:
                csv.writer(f).writerows(passed_lines)
        if missed_pairs or missed_stacks:
            with open(MISSED_KNOWN_INTERACTIONS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if missed_pairs: writer.writerows(missed_pairs)
                if missed_stacks: writer.writerows(missed_stacks)
    except IOError as e:
        print(f"ERROR: File write error: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    HEADER_PASSED = ["pdb_id", "chain1", "res_num1", "chain2", "res_num2", "nt1", "nt2", "distance_A"]
    HEADER_MISSED = ["pdb_id", "chain1", "res_num1", "chain2", "res_num2"]
    with open(ALL_PASSED_INTERACTIONS_FILE, 'w', newline='') as f: csv.writer(f).writerow(HEADER_PASSED)
    with open(MISSED_KNOWN_INTERACTIONS_FILE, 'w', newline='') as f: csv.writer(f).writerow(["type"] + HEADER_MISSED)

    center_csv_files = [os.path.join(CENTER_CSV_DIR, f) for f in os.listdir(CENTER_CSV_DIR) if f.endswith(".csv")]
    if not center_csv_files:
        logging.error(f"No center distance CSV files found in '{CENTER_CSV_DIR}'. Exiting.")
        exit()

    logging.info(f"Found {len(center_csv_files)} PDB structures to analyze using up to {NUM_CORES} cores.")

    final_pair_stats = {"total_known": 0, "recalled": 0, "missed": 0}
    final_stack_stats = final_pair_stats.copy()
    total_discovered_count = 0
    
    batch_passed_lines, batch_missed_pair_lines, batch_missed_stack_lines = [], [], []
    processed_in_batch_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        future_to_path = {executor.submit(process_pdb_id, path, CIF_FILES_DIR): path for path in center_csv_files}
        
        with tqdm(total=len(center_csv_files), desc="Analyzing Structures") as pbar:
            for future in concurrent.futures.as_completed(future_to_path):
                pbar.update(1)
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result is None: continue
                    
                    passed_lines, pair_stats, stack_stats = result
                    if pair_stats:
                        for key in final_pair_stats: final_pair_stats[key] += pair_stats[key]
                    if stack_stats:
                        for key in final_stack_stats: final_stack_stats[key] += stack_stats[key]

                    if passed_lines:
                        total_discovered_count += len(passed_lines)
                        batch_passed_lines.extend(passed_lines)
                    if pair_stats and pair_stats.get("missed_lines"):
                        batch_missed_pair_lines.extend([["pair"] + line for line in pair_stats["missed_lines"]])
                    if stack_stats and stack_stats.get("missed_lines"):
                        batch_missed_stack_lines.extend([["stack"] + line for line in stack_stats["missed_lines"]])
                    
                    processed_in_batch_count += 1
                    if processed_in_batch_count >= BATCH_WRITE_SIZE:
                        write_batch_data(batch_passed_lines, batch_missed_pair_lines, batch_missed_stack_lines)
                        batch_passed_lines.clear(); batch_missed_pair_lines.clear(); batch_missed_stack_lines.clear()
                        processed_in_batch_count = 0
                
                except Exception as exc:
                    print(f"ERROR: CRITICAL_MAIN_LOOP_ERROR for {os.path.basename(path)}: {exc}", file=sys.stderr, flush=True)
                    continue
    
    logging.info("Writing final batch of data to disk...")
    if processed_in_batch_count > 0:
        write_batch_data(batch_passed_lines, batch_missed_pair_lines, batch_missed_stack_lines)

    logging.info("\n\n--- ANALYSIS COMPLETE ---")
    logging.info(f"Total interactions discovered under {DISTANCE_THRESHOLD} Å: {total_discovered_count:,}")
    print_recall_summary(final_pair_stats, "Base Pairs")
    print_recall_summary(final_stack_stats, "Base Stacks")