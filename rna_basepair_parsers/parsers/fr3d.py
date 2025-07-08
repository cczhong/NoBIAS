#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd

def parse_out_file(file_path):
    """
    Parses a single FR3D .anno file and returns a DataFrame of base-pair annotations.
    """
    nt1_values, nt2_values, labels = [], [], []

    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) >= 6:
                label = values[1].strip('"')
                if label:  # Only retain if label is not empty
                    try:
                        nt1_values.append(values[0].strip('"'))
                        nt2_values.append(values[5].strip('"'))
                        labels.append(label)
                    except IndexError:
                        continue

    if not nt1_values:
        return pd.DataFrame()  # Return empty if no data

    df = pd.DataFrame({
        'chain1': [x.split('|')[2] for x in nt1_values],
        'pos1':   [x.split('|')[4] for x in nt1_values],
        'chain2': [x.split('|')[2] for x in nt2_values],
        'pos2':   [x.split('|')[4] for x in nt2_values],
        'nt1':    [x.split('|')[3] for x in nt1_values],
        'nt2':    [x.split('|')[3] for x in nt2_values],
        'label': labels
    })

    # Remove exact duplicates
    df = df.drop_duplicates(subset=['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'label'])

    # Remove mirror duplicates (pos1<->pos2 and nt1<->nt2)
    df['mirror_key'] = df.apply(
        lambda row: '|'.join(sorted([row['pos1'], row['pos2']])) + '|' +
                    '|'.join(sorted([row['nt1'], row['nt2']])), axis=1
    )
    df = df.drop_duplicates(subset='mirror_key', keep='first')
    df = df.drop(columns='mirror_key')

    return df

def main():
    parser = argparse.ArgumentParser(description="Parse FR3D .anno base-pair annotation files.")
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing FR3D .anno files")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory to save parsed CSV files")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.anno'):
            print(f"Processing file: {file_name}")
            try:
                input_file = os.path.join(input_dir, file_name)
                df = parse_out_file(input_file)
                if not df.empty:
                    output_file = os.path.join(output_dir, f"{file_name[:4].lower()}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"Saved to {output_file}")
                else:
                    print(f"No valid annotations found in {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    main()
