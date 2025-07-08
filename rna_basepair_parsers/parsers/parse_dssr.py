#!/usr/bin/env python3
# coding: utf-8

import os
import re
import argparse
import pandas as pd

def parse_out_file(file_path):
    """
    Parse a single DSSR .out file and return a DataFrame of base-pair annotations.
    """
    # Match lines like: 1 A.DA12 A.DU24 cWW ...
    pattern = re.compile(r'^\s+\d+\s+[A-Za-z0-9]+\.[A-Z]{1,3}-?\d+\s+[A-Za-z0-9]+\.[A-Z]{1,3}-?\d+\s+[A-Z\+~]+-[A-Z\+~]+\s+')

    nt1_values = []
    nt2_values = []
    bp = []
    labels = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if pattern.search(line):
                    values = line.strip().split()
                    if len(values) < 7:
                        continue

                    nt1_parts = values[1].split('.')
                    nt2_parts = values[2].split('.')

                    if len(nt1_parts) != 2 or len(nt2_parts) != 2:
                        continue

                    chain1, res1 = nt1_parts
                    chain2, res2 = nt2_parts

                    match1 = re.search(r'([A-Z]+)(-?\d+)', res1)
                    match2 = re.search(r'([A-Z]+)(-?\d+)', res2)
                    if not match1 or not match2:
                        continue

                    nt1, pos1 = match1.groups()
                    nt2, pos2 = match2.groups()

                    bp1 = values[3].split('-')[0]
                    bp2 = values[3].split('-')[1]
                    lw_label = values[6]

                    nt1_values.append((chain1, pos1, nt1))
                    nt2_values.append((chain2, pos2, nt2))
                    bp.append((bp1, bp2))
                    labels.append(lw_label)

        if not nt1_values:
            return pd.DataFrame()

        df = pd.DataFrame({
            'chain1': [x[0] for x in nt1_values],
            'pos1': [x[1] for x in nt1_values],
            'nt1': [x[2] for x in nt1_values],
            'chain2': [x[0] for x in nt2_values],
            'pos2': [x[1] for x in nt2_values],
            'nt2': [x[2] for x in nt2_values],
            'bp1': [x[0] for x in bp],
            'bp2': [x[1] for x in bp],
            'label': labels
        })

        # Remove reverse duplicates based on chain+pos
        df['pair_key'] = df.apply(
            lambda row: tuple(sorted([
                (row['chain1'], int(row['pos1'])),
                (row['chain2'], int(row['pos2']))
            ])), axis=1
        )
        df = df.drop_duplicates(subset='pair_key').drop(columns='pair_key')

        return df

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Parse DSSR .out files for base-pair annotations.")
    parser.add_argument('--input_dir', '-i', required=True, help="Directory containing DSSR .out files")
    parser.add_argument('--output_dir', '-o', required=True, help="Directory to save parsed CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".out"):
            input_path = os.path.join(args.input_dir, filename)
            try:
                df = parse_out_file(input_path)
                if not df.empty:
                    output_path = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0][:4]}.csv")
                    df.to_csv(output_path, index=False)
                    print(f"Saved: {output_path}")
                else:
                    print(f"No base pairs found in: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
