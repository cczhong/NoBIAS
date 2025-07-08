#!/usr/bin/env python3
# coding: utf-8

import os
import re
import argparse
import pandas as pd

def process_pdb_file(file_path):
    """
    Parse a single RNAVIEW .cif.out file and extract base-pair annotations.
    """
    data = []
    start_parsing = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        if line == 'BEGIN_base-pair':
            start_parsing = True
            continue
        if line == 'END_base-pair':
            break
        if not start_parsing or not line:
            continue
        if 'stacked' in line.lower():
            continue

        try:
            parts = re.split(r'\s+', line)

            chain1 = parts[1].split(':')[0]
            pos1 = int(parts[2])
            chain2 = parts[5].split(':')[0]
            pos2 = int(parts[4])

            nt1 = parts[3][0]
            nt2 = parts[3][2]

            bp_type = parts[6].strip()
            orientation = parts[7].strip().lower() if len(parts) > 7 else ''

            # Normalize edge
            if bp_type in ['+/+', '-/-']:
                edge = 'WW'
            elif '/' in bp_type and len(bp_type) == 3:
                edge = bp_type[0] + bp_type[2]
            else:
                edge = 'NA'

            # Normalize orientation
            if 'cis' in orientation:
                label = 'c' + edge
            elif 'trans' in orientation or 'tran' in orientation:
                label = 't' + edge
            else:
                label = 'NA'

            if edge != 'NA' and label != 'NA':
                data.append([chain1, pos1, chain2, pos2, nt1, nt2, label])

        except (IndexError, ValueError):
            print(f"Skipping malformed line: {line}")
            continue

    return pd.DataFrame(data, columns=['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'label'])

def main():
    parser = argparse.ArgumentParser(description="Parse RNAVIEW .cif.out base-pair annotation files.")
    parser.add_argument('--input_dir', '-i', required=True, help="Directory containing RNAVIEW .cif.out files")
    parser.add_argument('--output_dir', '-o', required=True, help="Directory to save parsed CSV files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for file_name in os.listdir(args.input_dir):
        if file_name.endswith('.cif.out'):
            print(f"Processing: {file_name}")
            file_path = os.path.join(args.input_dir, file_name)
            df = process_pdb_file(file_path)

            output_name = file_name[:4].lower() + '.csv'
            output_path = os.path.join(args.output_dir, output_name)
            df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
