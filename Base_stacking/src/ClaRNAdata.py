import os
import re
from Bio.PDB import MMCIFParser
from collections import defaultdict

clarna_dir = "CLARNA/CLARNA"
cif_dir = "cif/cif"
output_file = "output_clarna.csv"

def parse_cif(molecule_name, cif_dir):
    """Parse CIF file and return residue and chain mappings."""
    cif_path = os.path.join(cif_dir, f"{molecule_name}.cif")

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(molecule_name, cif_path)
        residue_map = {}
        chain_num_map = {}

        for model in structure:
            for chain in model:
                chain_id = chain.id
                chain_num_map[chain_id] = model.id
                for residue in chain:
                    res_name = residue.resname.strip()
                    if res_name in ('A', 'C', 'G', 'U', 'PSU', 'H2U', '5MC', '5MU'):
                        res_num = str(residue.id[1])
                        residue_map[(chain_id, res_num)] = res_name
        return residue_map, chain_num_map

    except Exception as e:
        print(f"Error parsing CIF file {cif_path}: {e}")
        return None, None

def parse_clarna_output(file_path, residue_map, chain_num_map):
    """Parse CLARNA output file and classify interactions."""
    
    molecule_name = os.path.basename(file_path).split('.')[0]
    interaction_counts = defaultdict(int)

    with open(file_path, 'r') as f:
        content = f.read()

    # Extract contacts section
    contacts_match = re.search(r"find contacts .*?DONE", content, re.DOTALL)
    if not contacts_match:
        print(f"Warning: No contacts section found in {file_path}")
        return []

    contacts_section = contacts_match.group()

    # Regex to capture the full interaction line
    interaction_pattern = re.compile(
        r"^\s*\d+_\d+,\s*([A-Z]):\s+(\d+)\s+([AUCG]+)-([AUCG]+)\s+(\d+)\s+[A-Z]:\s+([^\[\n]+)",
        re.MULTILINE
    )

    results = []
    for match in interaction_pattern.finditer(contacts_section):
        chain1, pos1, base1, base2, pos2, interaction_part = match.groups()
        
        # Skip if contains question mark OR doesn't contain any arrows
        if '?' in interaction_part or not any(sym in interaction_part for sym in ['<<', '>>', '<>', '><']):
            continue

        # Extract the first arrow pattern found
        arrow_match = re.search(r'([<>]{2})', interaction_part)
        if not arrow_match:
            continue
            
        interaction = arrow_match.group(1)

        # Classify interaction type
        if interaction in ['<<', '>>']:
            direction = 'up/down'
        elif interaction == '><':
            direction = 'inward'
        elif interaction == '<>':
            direction = 'outward'
        else:
            continue  # Skip if not a recognized pattern

        # Verify residues exist in structure
        if (chain1, pos1) not in residue_map or (chain1, pos2) not in residue_map:
            continue

        # Count interaction types
        interaction_counts[direction] += 1

        # Prepare output row
        chain_no = chain_num_map.get(chain1, '?')
        results.append(
            f"{molecule_name},{chain1},{chain1},{chain_no},"
            f"{base1}-{base2},{pos1},{pos2},{direction}"
        )

    # Print interaction statistics for this file
    print(f"\nInteraction counts for {molecule_name}:")
    for direction, count in sorted(interaction_counts.items()):
        print(f"{direction}: {count}")

    return results

def main():
    with open(output_file, 'w') as outfile:
        outfile.write("mol_name, chain1, chain2, chain_no, base/residue_name, position1, position2, label\n")

        for filename in sorted(os.listdir(clarna_dir)):
            if not filename.endswith(".out"):
                continue

            file_path = os.path.join(clarna_dir, filename)
            mol_name = filename.split('.')[0]

            residue_map, chain_num_map = parse_cif(mol_name, cif_dir)
            if not residue_map:
                print(f"Skipping {filename} - CIF parsing failed")
                continue

            for row in parse_clarna_output(file_path, residue_map, chain_num_map):
                outfile.write(row + '\n')

if __name__ == "__main__":
    main()
