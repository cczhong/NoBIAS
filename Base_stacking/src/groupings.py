import os
import csv
from itertools import combinations

INPUT_FILES = {
    "ClaRNA": "output_clarna.csv",
    "DSSR": "output_dssr.csv",
    "FR3D": "output_fr3.csv",
    "MCA": "output_mca.csv"
}

OUTPUT_DIR = "comparison_results"

def create_canonical_key(row):
    """
    Creates a standardized key for an interaction to handle swapped residue orders.
    
    For an interaction between (chain1, pos1) and (chain2, pos2), this ensures the key is the same regardless of which residue is listed first.
    """
    mol_name, chain1, chain2, _, _, pos1, pos2, _ = row
    
    # Create two pairs and sort them to ensure consistent order
    res1_tuple = (chain1, pos1)
    res2_tuple = (chain2, pos2)
    
    # Sort the tuples themselves to handle cases like ('B', '50') vs ('A', '10')
    sorted_residues = sorted([res1_tuple, res2_tuple])
    
    return f"{mol_name}_{sorted_residues[0][0]}_{sorted_residues[0][1]}_{sorted_residues[1][0]}_{sorted_residues[1][1]}"

def write_csv(filepath, header, rows):
    
    unique_rows = sorted(list(set(rows)))
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(unique_rows)
        
    print(f"  -> Saved {len(unique_rows)} rows to {filepath}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_interactions = {}
    row_data_map = {}
    csv_header = []

    # Read all input files and aggregate the data
    for tool_name, filepath in INPUT_FILES.items():
        if not os.path.exists(filepath):
            print(f"  [Warning] File not found, skipping: {filepath}")
            continue
        
        print(f"  -> Processing {filepath} for tool '{tool_name}'")
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            if not csv_header:
                csv_header = next(reader)
            else:
                next(reader) # Skip header for other files

            for row in reader:
                if len(row) != len(csv_header):
                    continue # Skip malformed rows
                
                key = create_canonical_key(row)
                label = row[-1]

                all_interactions.setdefault(key, {})
                all_interactions[key].setdefault(label, set())
                all_interactions[key][label].add(tool_name)
                
                if key not in row_data_map:
                    row_data_map[key] = row

    if not all_interactions:
        return

    # Generate threshold-based CSVs (pass_all, pass_3, pass_2)
    pass_all_rows = []
    pass_3_rows = []
    pass_2_rows = []
    
    num_tools_total = len(INPUT_FILES)

    for key, label_info in all_interactions.items():
        for label, tools in label_info.items():
            count = len(tools)
            
            # Get the representative row and replace its label with the current one
            base_row = row_data_map[key]
            output_row = tuple(base_row[:-1] + [label])
            
            if count == num_tools_total:
                pass_all_rows.append(output_row)
            if count >= 3:
                pass_3_rows.append(output_row)
            if count >= 2:
                pass_2_rows.append(output_row)
                
    write_csv(os.path.join(OUTPUT_DIR, "pass_all.csv"), csv_header, pass_all_rows)
    write_csv(os.path.join(OUTPUT_DIR, "pass_3.csv"), csv_header, pass_3_rows)
    write_csv(os.path.join(OUTPUT_DIR, "pass_2.csv"), csv_header, pass_2_rows)

    # 3. Generate combination-based CSVs for groups of 3 and 2
    tool_names = list(INPUT_FILES.keys())

    # --- Groups of 3 ---
    for combo in combinations(tool_names, 3):
        combo_set = set(combo)
        combo_rows = []
        
        for key, label_info in all_interactions.items():
            for label, tools in label_info.items():
                # Check for an EXACT match with the current combination
                if tools == combo_set:
                    base_row = row_data_map[key]
                    output_row = tuple(base_row[:-1] + [label])
                    combo_rows.append(output_row)
        
        filename = f"group_3_{'_'.join(sorted(combo))}.csv"
        write_csv(os.path.join(OUTPUT_DIR, filename), csv_header, combo_rows)

    # --- Groups of 2 ---
    for combo in combinations(tool_names, 2):
        combo_set = set(combo)
        combo_rows = []

        for key, label_info in all_interactions.items():
            for label, tools in label_info.items():
                # Check for an EXACT match with the current combination
                if tools == combo_set:
                    base_row = row_data_map[key]
                    output_row = tuple(base_row[:-1] + [label])
                    combo_rows.append(output_row)

        filename = f"group_2_{'_'.join(sorted(combo))}.csv"
        write_csv(os.path.join(OUTPUT_DIR, filename), csv_header, combo_rows)

    print("\n--- Grouping Complete ---")

if __name__ == "__main__":
    main()