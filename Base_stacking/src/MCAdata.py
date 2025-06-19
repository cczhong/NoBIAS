import os
import csv
import re
import sys
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

ANNO_DIR = "MCA/MCA"
CIF_DIR = "cif/cif"
OUTPUT_CSV = "output_mca.csv"

# RNA base mappings
RNA_BASES_3_TO_1_LETTER = {
    "A": "A", "ADE": "A",
    "U": "U", "URA": "U",
    "C": "C", "CYT": "C",
    "G": "G", "GUA": "G",
}
RNA_SINGLE_LETTERS = {'A', 'U', 'C', 'G'}

# 4->3 class fixer
LABEL_MAP = {
    "upward": "up/down",
    "downward": "up/down",
    "inward": "inward",
    "outward": "outward",
}

def parse_cif(molecule_name, cif_dir):
    """
    Parses a CIF file to extract a residue map and a chain-to-entity map.
    
    Returns:
        A tuple (residue_map, chain_number_map).
        Returns (None, None) if parsing fails or the file doesn't exist.
    """
    
    cif_path = os.path.join(cif_dir, f"{molecule_name}.cif")

    try:
        cif_dict = MMCIF2Dict(cif_path)

        # Create a map from chain ID to its entity number
        chain_ids_cif = cif_dict.get('_atom_site.auth_asym_id', [])
        entity_ids_cif = cif_dict.get('_atom_site.label_entity_id', [])
        chain_number_map = {
            cid: eid for cid, eid in set(zip(chain_ids_cif, entity_ids_cif))
        }

        # Get all atom-level data
        model_numbers = cif_dict.get('_atom_site.pdbx_PDB_model_num', [])
        auth_asym_ids = cif_dict.get('_atom_site.auth_asym_id', [])
        auth_seq_ids = cif_dict.get('_atom_site.auth_seq_id', [])
        label_comp_ids = cif_dict.get('_atom_site.label_comp_id', [])

        # Ensures data consistency, havent had any cases trigger this one yet
        if not (len(auth_asym_ids) == len(auth_seq_ids) == len(label_comp_ids)):
            print(f"[Warning] Mismatch in CIF data lengths for {molecule_name}.", file=sys.stderr)
            return None, chain_number_map

        # some files have multiple versions, use the main one only
        atom_indices_to_use = range(len(auth_asym_ids))
        if model_numbers:
            first_model_value = next((m for m in model_numbers if m is not None), None)
            if first_model_value:
                atom_indices_to_use = [i for i, num in enumerate(model_numbers) if num == first_model_value]

        # Create a map from (chain, position) to residue name
        residue_map = {}
        processed_residues = set()
        
        for i in atom_indices_to_use:
            
            chain = auth_asym_ids[i]
            pos = auth_seq_ids[i]
            residue = label_comp_ids[i]
            
            if (chain, pos) not in processed_residues:
                residue_map[(chain, pos)] = residue
                processed_residues.add((chain, pos))

        return residue_map, chain_number_map

    except Exception as e:
        print(f"[Error] Exception parsing CIF {cif_path}: {e}", file=sys.stderr)
        return None, None

def parse_residue_identifier(residue_id_str):
    """Splits a residue string like A101 into chain A and position 101"""
    match = re.match(r"([A-Za-z0-9]+?)(\d+)$", residue_id_str)
    if match:
        return match.group(1), match.group(2)
    return None, None

def main():

    with open(OUTPUT_CSV, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['molecule_name', 'chain1', 'chain2', 'chain_number',
                         'base_residue_name', 'position1', 'position2', 'label'])

        processed_files_count = 0
        total_rows_written = 0

        # Iterate over each file in the annotation directory
        for anno_filename in os.listdir(ANNO_DIR):
            if not anno_filename.endswith('_MC'):
                continue

            molecule_name = anno_filename[:-3]  # Remove '_MC' suffix
            print(f"Processing: {molecule_name}")

            # Get residue and chain info from the corresponding CIF file
            residue_map, chain_number_map = parse_cif(molecule_name, CIF_DIR)
            if not residue_map or not chain_number_map:
                print(f"  -> Skipping {molecule_name} due to missing or incomplete CIF data.")
                continue

            processed_files_count += 1
            rows_written_for_file = 0
            anno_path = os.path.join(ANNO_DIR, anno_filename)

            # Read the annotation file line by line
            try:
                with open(anno_path, 'r') as anno_file:
                    in_adjacent_stacking_section = False
                    for line in anno_file:
                        line = line.strip()

                        # Start parsing only after this header is found
                        if "Adjacent stackings" in line:
                            in_adjacent_stacking_section = True
                            continue
                        
                        # Stop parsing if a new section starts
                        if "Non-Adjacent stackings" in line:
                            break

                        if not in_adjacent_stacking_section or ':' not in line:
                            continue

                        # 3. Extract and process data for each interaction
                        pair_str, interaction_details = [part.strip() for part in line.split(':', 1)]
                        if '-' not in pair_str:
                            continue

                        residue1_id_str, residue2_id_str = pair_str.split('-', 1)
                        chain1, pos1 = parse_residue_identifier(residue1_id_str)
                        chain2, pos2 = parse_residue_identifier(residue2_id_str)

                        if not (chain1 and pos1 and chain2 and pos2):
                            continue

                        # Use CIF data to validate and get base names
                        base1_3_letter = residue_map.get((chain1, pos1))
                        base2_3_letter = residue_map.get((chain2, pos2))
                        if not base1_3_letter or not base2_3_letter:
                            continue

                        base1_1_letter = RNA_BASES_3_TO_1_LETTER.get(base1_3_letter.upper())
                        base2_1_letter = RNA_BASES_3_TO_1_LETTER.get(base2_3_letter.upper())
                        
                        if not (base1_1_letter and base2_1_letter):
                            continue
                        
                        # Extract and normalize the label
                        interaction_words = interaction_details.split()
                        raw_label = ""
                        if interaction_words:
                            # Handle cases like "anti upward pairing" -> "upward"
                            if interaction_words[-1].lower() == "pairing" and len(interaction_words) > 1:
                                raw_label = interaction_words[-2].lower()
                            else:
                                raw_label = interaction_words[-1].lower()
                        
                        final_label = LABEL_MAP.get(raw_label)
                        
                        # Write to CSV
                        if final_label:
                            base_residue_name = f"{base1_1_letter}-{base2_1_letter}"
                            chain_number = chain_number_map.get(chain1, '?')
                            
                            writer.writerow([
                                molecule_name, chain1, chain2, chain_number,
                                base_residue_name, pos1, pos2, final_label
                            ])
                            rows_written_for_file += 1
                
                total_rows_written += rows_written_for_file

            except Exception as e:
                print(f"[Error] Exception while processing annotation file {anno_path}: {e}", file=sys.stderr)
                continue
    
    print("\n----- Processing Complete -----")
    print(f"Processed {processed_files_count} annotation files.")
    print(f"Total rows written to {OUTPUT_CSV}: {total_rows_written}")

if __name__ == "__main__":
    main()