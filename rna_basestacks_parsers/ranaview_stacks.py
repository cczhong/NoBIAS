import pandas as pd
import re
import os

def process_pdb_stackings(input_dir, output_dir):
    pdb_files = [file for file in os.listdir(input_dir) if file.endswith('.cif.out')]
    
    for pdb_file in pdb_files:
        print(f"Processing: {pdb_file}")
        with open(os.path.join(input_dir, pdb_file), 'r') as file:
            lines = file.readlines()

        data = []
        start_parsing = False

        for line in lines:
            line = line.strip()

            if line == 'BEGIN_base-pair':
                start_parsing = True
                continue
            if line == 'END_base-pair':
                break
            if not start_parsing or not line:
                continue

            # **Only parse lines with "stacked"**
            if 'stacked' not in line.lower():
                continue

            try:
                parts = re.split(r'\s+', line)
                # The columns are the same as in base-pair block
                chain1 = parts[1].split(':')[0] if ':' in parts[1] else parts[1]
                pos1 = int(parts[2])
                chain2 = parts[5].split(':')[0] if ':' in parts[5] else parts[5]
                pos2 = int(parts[4])

                nt_pair = parts[3]
                nt1 = nt_pair[0]
                nt2 = nt_pair[2]

                # Label as "stacked" (or you can add more columns if needed)
                label = "stacked"

                data.append([chain1, pos1, chain2, pos2, nt1, nt2, label])

            except (IndexError, ValueError):
                print(f"Skipping malformed line: {line}")
                continue

        df = pd.DataFrame(data, columns=['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'label'])

        output_filename = os.path.splitext(pdb_file)[0][:4] + '.csv'
        output_path = os.path.join(output_dir, output_filename)
        df.to_csv(output_path, index=False)

# Usage
input_directory = '/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/RNAV_0419'
output_directory = '/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/Parsed_RNAV_stackings'
os.makedirs(output_directory, exist_ok=True)

process_pdb_stackings(input_directory, output_directory)
