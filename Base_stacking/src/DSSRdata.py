import os
import re
import sys
import csv
from Bio.PDB import MMCIFParser

DSSR_DIR = "DSSR/DSSR"
CIF_DIR = "cif/cif"
OUTPUT_CSV = "output_dssr.csv"

def parse_cif(molecule_name, cif_dir):
    """
    Parses a CIF file to create a residue-to-base map and a chain-to-model map.
    """
    cif_path = os.path.join(cif_dir, f"{molecule_name}.cif")
        
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(molecule_name, cif_path)
        residue_map = {}
        chain_to_model_map = {}
        
        # Bio.PDB iterates through models, then chains, then residues
        for model in structure:
            for chain in model:
                chain_id = chain.id
                if chain_id not in chain_to_model_map:
                    chain_to_model_map[chain_id] = model.id 
                for residue in chain:
                    residue_name = residue.resname.strip()
                    if residue_name in ['A', 'C', 'G', 'U']:
                        position = str(residue.id[1])
                        residue_map[(chain_id, position)] = residue_name
                        
        return residue_map, chain_to_model_map
        
    except Exception as e:
        print(f"[Error] Failed to parse CIF {cif_path}: {e}", file=sys.stderr)
        return None, None

def extract_position(residue_part):
    match = re.search(r'\d+', residue_part)
    return match.group() if match else None

def main():
    with open(OUTPUT_CSV, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['molecule_name', 'chain1', 'chain2', 'chain_number',
                         'base_residue_name', 'position1', 'position2', 'label'])

        processed_files_count = 0
        total_rows_written = 0

        for filename in os.listdir(DSSR_DIR):
            if not filename.endswith('.out'):
                continue

            molecule_name = os.path.splitext(filename)[0]
            print(f"Processing: {molecule_name}")

            residue_map, chain_number_map = parse_cif(molecule_name, CIF_DIR)
            if not residue_map or not chain_number_map:
                print(f"  -> Skipping {molecule_name} due to missing or failed CIF parse.")
                continue

            processed_files_count += 1
            dssr_path = os.path.join(DSSR_DIR, filename)

            try:
                with open(dssr_path, 'r') as f:
                    content = f.read()

                # Find all "List of...helix" sections in the file
                helix_sections = re.finditer(r"List of \d+ helix.*?(?=\*{5,}|List of|$)", content, re.DOTALL)

                for section in helix_sections:
                    # Process each line within the found helix section
                    for line in section.group(0).split('\n'):
                        line = line.strip()
                        
                        # A valid data line starts with a number (1 B.G201)
                        if not re.match(r"^\d+\s+", line):
                            continue
                        
                        parts = line.split()
                        if len(parts) < 3:
                            continue

                        # Extract the two residue identifiers ("B.G201", "B.C220")
                        residue1_str = parts[1]
                        residue2_str = parts[2]

                        if '.' not in residue1_str: continue
                        chain1, res_part1 = residue1_str.split('.', 1)
                        position1 = extract_position(res_part1)

                        if '.' not in residue2_str: continue
                        chain2, res_part2 = residue2_str.split('.', 1)
                        position2 = extract_position(res_part2)

                        if not (position1 and position2):
                            continue

                        base1 = residue_map.get((chain1, position1))
                        base2 = residue_map.get((chain2, position2))

                        if not base1 or not base2:
                            continue

                        chain_number = chain_number_map.get(chain1, '?')
                        base_residue_name = f"{base1}-{base2}"
                        
                        writer.writerow([
                            molecule_name, chain1, chain2, chain_number,
                            base_residue_name, position1, position2, "up/down" # all helixes are s35, up/down class
                        ])
                        total_rows_written += 1
            
            except Exception as e:
                print(f"[Error] Failed to process file {dssr_path}: {e}", file=sys.stderr)

    print(f"Processed {processed_files_count} annotation files.")
    print(f"Total rows written to {OUTPUT_CSV}: {total_rows_written}")

if __name__ == "__main__":
    main()