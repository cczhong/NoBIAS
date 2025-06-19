import os
import csv
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

FR3D_DIR = "FR3D/FR3D"
CIF_DIR = "cif/cif"
OUTPUT_CSV = "output_fr3.csv"
ANNO_EXTENSION = ".anno"

RNA_BASES = {'A', 'U', 'C', 'G'}
INTERACTION_LABELS = {
    's35': 'up/down',
    's53': 'up/down',
    's55': 'outward',
    's33': 'inward',
}
CSV_HEADER = [
    'mol_name', 'chain1', 'chain2', 'chain_no', 'base/residue_name',
    'position1', 'position2', 'label'
]

def parse_cif(molecule_name, cif_dir):
    """
    Parses a .cif file to extract residue and chain information for the first model.
    """
    cif_path = os.path.join(cif_dir, f"{molecule_name}.cif")

    try:
        cif_dict = MMCIF2Dict(cif_path)

        # Extract mapping from author chain ID to entity ID
        chain_ids = cif_dict.get('_atom_site.auth_asym_id', [])
        entity_ids = cif_dict.get('_atom_site.label_entity_id', [])
        chain_num_map = {}
        for cid, eid in zip(chain_ids, entity_ids):
            if cid not in chain_num_map:
                chain_num_map[cid] = eid

        # Focus on the first model if multiple exist
        model_numbers = cif_dict.get('_atom_site.pdbx_PDB_model_num')
        if model_numbers:
            first_model_indices = [i for i, num in enumerate(model_numbers) if num == '1']
        else:
            first_model_indices = range(len(chain_ids))

        # Create a map of (chain, position) -> residue_name
        residue_map = {}
        seen_residues = set()
        for i in first_model_indices:
            try:
                chain = cif_dict['_atom_site.auth_asym_id'][i]
                pos = cif_dict['_atom_site.auth_seq_id'][i]
                if (chain, pos) in seen_residues:
                    continue

                res = cif_dict['_atom_site.label_comp_id'][i]
                residue_map[(chain, pos)] = res
                seen_residues.add((chain, pos))
            except IndexError:
                # Silently skip incomplete atom records
                continue

        return residue_map, chain_num_map

    except Exception as e:
        print(f"Error parsing {cif_path}: {e}")
        return None, None


def main():
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)

        for filename in os.listdir(FR3D_DIR):
            if not filename.endswith(ANNO_EXTENSION):
                continue

            mol_name = os.path.splitext(filename)[0]
            mcdata_path = os.path.join(FR3D_DIR, filename)

            residue_map, chain_num_map = parse_cif(mol_name, CIF_DIR)
            if not residue_map or not chain_num_map:
                print(f"Skipping {mol_name}: CIF data unavailable or failed to parse.")
                continue

            try:
                with open(mcdata_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 6:
                            continue

                        interaction = row[1].strip() or row[2].strip()
                        label = INTERACTION_LABELS.get(interaction)
                        if not label:
                            continue

                        res1_parts = row[0].split('|')
                        res2_parts = row[5].split('|')
                        chain1, pos1 = res1_parts[2], res1_parts[4]
                        chain2, pos2 = res2_parts[2], res2_parts[4]

                        res_name1 = residue_map.get((chain1, pos1), 'UNK')
                        res_name2 = residue_map.get((chain2, pos2), 'UNK')

                        base1 = res_name1[0].upper() if res_name1 != 'UNK' else 'X'
                        base2 = res_name2[0].upper() if res_name2 != 'UNK' else 'X'

                        if not {base1, base2}.issubset(RNA_BASES):
                            continue
                        
                        chain_no = chain_num_map.get(chain1, '?')

                        writer.writerow([
                            mol_name, chain1, chain2, chain_no,
                            f"{base1}-{base2}", pos1, pos2, label
                        ])

            except Exception as e:
                print(f"Error processing {mcdata_path}: {e}")

if __name__ == "__main__":
    main()