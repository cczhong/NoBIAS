import re
import os
import pandas as pd

def parse_stackings(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = os.listdir(input_folder)

    for filename in filenames:
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"{filename[:4]}.csv")

        if not os.path.isfile(input_file):
            continue

        pdb_id = filename[:4].lower()
        print(f"\n=== Processing PDB: {pdb_id} (file: {filename}) ===")

        with open(input_file, 'r') as file:
            lines = file.readlines()

        parsed_data = []
        parsing_type = None  # 'adjacent' or 'nonadjacent'

        for line in lines:
            line = line.strip()

            # Detect which stacking block we're in
            if line.startswith("Adjacent stackings"):
                parsing_type = 'adjacent'
                continue
            elif line.startswith("Non-Adjacent stackings"):
                parsing_type = 'nonadjacent'
                continue
            elif (line.startswith("Base-pairs") or
                  line.startswith("Residue conformations") or
                  '-'*10 in line):
                parsing_type = None
                continue

            # Only process stacking lines inside the right block
            if parsing_type and ':' in line:
                try:
                    bp_part, desc_part = line.split(":", 1)
                    bp_part = bp_part.strip()
                    # Try each format in order of complexity
                    match = (
                        re.match(r"^([A-Za-z]+)(-?\d+)-([A-Za-z]+)(-?\d+)$", bp_part) or          # A1-B16
                        re.match(r"^'(\d+)'\s*(\d+)-'(\d+)'\s*(\d+)$", bp_part) or                # '0'531-'0'532
                        re.match(r"^([A-Za-z])(\d+)\.([A-Za-z])\-([A-Za-z])(\d+)\.([A-Za-z])$", bp_part) or # A129.A-B131.E
                        re.match(r"^'(\d+)'\s*(\d+)\.([A-Za-z])-'(\d+)'\s*(\d+)\.([A-Za-z])$", bp_part) or  # '0'186.A-'0'191.G
                        re.match(r"^([A-Za-z]+)(-?\d+)\.([A-Za-z]+)-([A-Za-z]+)(-?\d+)$", bp_part) or       # A738.E-B10
                        re.match(r"^([A-Za-z]+)(-?\d+)-([A-Za-z]+)(-?\d+)\.([A-Za-z]+)$", bp_part) or       # B10-A738.E
                        re.match(r"^'(\d+)'\s*(\d+)\.([A-Za-z]+)-'(\d+)'\s*(\d+)$", bp_part) or             # '5'738.A-'5'918
                        re.match(r"^'(\d+)'\s*(\d+)-'(\d+)'\s*(\d+)\.([A-Za-z]+)$", bp_part) or             # '5'741-'5'922.A
                        re.match(r"^'(\d+)'\s*(-?\d+)-'(\d+)'\s*(-?\d+)$", bp_part)                         # '0'25-'1'-25
                    )

                    if match:
                        groups = match.groups()
                        # Parse each possible case, see which matched:
                        if len(groups) == 4:  # A1-B16, '0'531-'0'532, or similar
                            chain1, pos1, chain2, pos2 = groups
                        elif len(groups) == 6 and "." in bp_part:  # dot notation variants
                            # Try to be robust:
                            if "'" in bp_part:  # quote-dot e.g. '0'186.A
                                q1, p1, c1, q2, p2, c2 = groups
                                chain1 = f"{c1}{q1}"
                                pos1 = p1
                                chain2 = f"{c2}{q2}"
                                pos2 = p2
                            else:
                                # Standard: A129.A-B131.E
                                a1, p1, a2, a3, p2, a4 = groups
                                chain1 = f"{a2}{a1}"
                                pos1 = p1
                                chain2 = f"{a4}{a3}"
                                pos2 = p2
                        elif len(groups) == 5:  # Asymmetry, e.g. A738.E-B10 or B10-A738.E
                            if "." in bp_part.split("-")[0]:
                                # A738.E-B10
                                a1, p1, a2, a3, p2 = groups
                                chain1 = f"{a2}{a1}"
                                pos1 = p1
                                chain2 = a3
                                pos2 = p2
                            else:
                                # B10-A738.E
                                a1, p1, a2, p2, a3 = groups
                                chain1 = a1
                                pos1 = p1
                                chain2 = f"{a3}{a2}"
                                pos2 = p2
                        else:
                            print(f"Unparsed groups for: {bp_part}")
                            continue
                    else:
                        print(f"Unrecognized stacking line: {bp_part}")
                        continue

                    # Label is typically the last word, e.g. "adjacent_5p upward"
                    desc_parts = desc_part.strip().split()
                    if len(desc_parts) >= 2:
                        label = desc_parts[-1]
                    else:
                        label = desc_parts[0]
                    parsed_data.append([chain1, pos1, chain2, pos2, parsing_type, label])
                except Exception as e:
                    print(f"Failed to parse stacking line: {line}\nError: {e}")
                    continue

        if parsed_data:
            df = pd.DataFrame(parsed_data, columns=['chain1', 'pos1', 'chain2', 'pos2', 'type', 'label'])
            df.drop_duplicates(inplace=True)
            df.replace('', pd.NA, inplace=True)
            df.to_csv(output_file, index=False)
            print(f"Parsed stacking data saved to {output_file} successfully.")
        else:
            print(f"No stacking data found in file: {filename}")

# === Set your folders ===
input_folder = '/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/MC_Annotate_0419'
output_folder = '/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/Parsed_MC_stackings'
parse_stackings(input_folder, output_folder)
