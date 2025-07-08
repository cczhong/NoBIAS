#!/usr/bin/env python3
# coding: utf-8

import re
import os
import pandas as pd
import argparse

def parse_base_pairs(input_folder, output_folder):
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
        start_parsing = False

        for line in lines:
            line = line.strip()
            if not start_parsing:
                if line.startswith("Base-pairs") and '-' * 10 in line:
                    start_parsing = True
                continue

            if line == '' or line.startswith("Residue conformations"):
                break

            if ':' not in line:
                continue

            try:
                bp_part, desc_part = line.split(":", 1)
                desc_parts = desc_part.strip().split()

                match_std = re.match(r"^([A-Za-z]+)(-?\d+)-([A-Za-z]+)(-?\d+)$", bp_part.strip())
                match_quote = re.match(r"^'(\d+)'\s*(\d+)-'(\d+)'\s*(\d+)$", bp_part.strip())
                match_dot = re.match(r"^([A-Za-z])(\d+)\.([A-Za-z])\-([A-Za-z])(\d+)\.([A-Za-z])$", bp_part.strip())
                match_mixed = re.match(r"^'(\d+)'\s*(\d+)-([A-Za-z]+)(-?\d+)$", bp_part.strip())
                match_mixed_rev = re.match(r"^([A-Za-z]+)(-?\d+)-'(\d+)'\s*(\d+)$", bp_part.strip())
                match_asym_dot = re.match(r"^([A-Za-z]+)(-?\d+)\.([A-Za-z]+)-([A-Za-z]+)(-?\d+)$", bp_part.strip())
                match_asym_dot_rev = re.match(r"^([A-Za-z]+)(-?\d+)-([A-Za-z]+)(-?\d+)\.([A-Za-z]+)$", bp_part.strip())
                match_quote_dot_left = re.match(r"^'(\d+)'\s*(\d+)\.([A-Za-z]+)-'(\d+)'\s*(\d+)$", bp_part.strip())
                match_quote_dot_right = re.match(r"^'(\d+)'\s*(\d+)-'(\d+)'\s*(\d+)\.([A-Za-z]+)$", bp_part.strip())
                match_quote_neg = re.match(r"^'(\d+)'\s*(-?\d+)-'(\d+)'\s*(-?\d+)$", bp_part.strip())
                match_quote_dot_both = re.match(r"^'(\d+)'\s*(\d+)\.([A-Za-z])-'(\d+)'\s*(\d+)\.([A-Za-z])$", bp_part.strip())

                if match_std:
                    chain1, pos1, chain2, pos2 = match_std.groups()
                elif match_quote:
                    chain1, pos1, chain2, pos2 = match_quote.groups()
                elif match_dot:
                    a1, p1, a2, a3, p2, a4 = match_dot.groups()
                    chain1 = a2 + a1
                    pos1 = p1
                    chain2 = a4 + a3
                    pos2 = p2
                elif match_mixed:
                    chain1, pos1, chain2, pos2 = match_mixed.groups()
                elif match_mixed_rev:
                    chain1, pos1, chain2, pos2 = match_mixed_rev.groups()
                elif match_asym_dot:
                    a1, p1, a2, a3, p2 = match_asym_dot.groups()
                    chain1 = a2 + a1
                    pos1 = p1
                    chain2 = a3
                    pos2 = p2
                elif match_asym_dot_rev:
                    a1, p1, a2, p2, a3 = match_asym_dot_rev.groups()
                    chain1 = a1
                    pos1 = p1
                    chain2 = a3 + a2
                    pos2 = p2
                elif match_quote_dot_left:
                    q1, p1, c1, q2, p2 = match_quote_dot_left.groups()
                    chain1 = c1 + q1
                    pos1 = p1
                    chain2 = q2
                    pos2 = p2
                elif match_quote_dot_right:
                    q1, p1, q2, p2, c2 = match_quote_dot_right.groups()
                    chain1 = q1
                    pos1 = p1
                    chain2 = c2 + q2
                    pos2 = p2
                elif match_quote_neg:
                    chain1, pos1, chain2, pos2 = match_quote_neg.groups()
                elif match_quote_dot_both:
                    q1, p1, c1, q2, p2, c2 = match_quote_dot_both.groups()
                    chain1 = c1 + q1
                    pos1 = p1
                    chain2 = c2 + q2
                    pos2 = p2
                else:
                    print(f"Unrecognized format: {bp_part}")
                    continue

                if '-' in desc_parts[0]:
                    nt1, nt2 = desc_parts[0].split('-')
                else:
                    nt1 = nt2 = '-'

                if 'cis' in desc_parts or 'trans' in desc_parts:
                    try:
                        if 'cis' in desc_parts:
                            canonical = desc_parts[1] if len(desc_parts) > 1 else 'Ww/Ww'
                            label = 'c' + canonical[0] + canonical[3]
                        else:
                            canonical = desc_parts[1] if len(desc_parts) > 1 else 'Ww/Ww'
                            label = 't' + canonical[0] + canonical[3]
                    except Exception:
                        label = '_'.join(desc_parts[1:])
                else:
                    label = '_'.join(desc_parts[1:]) if len(desc_parts) > 1 else desc_parts[0]

                parsed_data.append([chain1, pos1, chain2, pos2, nt1, nt2, label])

            except Exception as e:
                print(f"Failed to parse line: {line}\nError: {e}")
                continue

        if parsed_data:
            df = pd.DataFrame(parsed_data, columns=['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'label'])
            df.drop_duplicates(inplace=True)
            df.replace('', pd.NA, inplace=True)
            df.to_csv(output_file, index=False)
            print(f"Parsed data saved to {output_file} successfully.")
        else:
            print(f"No base-pair data found in file: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse MC-Annotate base pair annotations.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to input folder")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to output folder")
    args = parser.parse_args()

    parse_base_pairs(args.input_dir, args.output_dir)
