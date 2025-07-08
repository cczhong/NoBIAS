#!/usr/bin/env python3
# coding: utf-8

import re
import csv
import os
import argparse

def get_lines_after_keyword(file_path, keyword):
    """
    Extracts all lines in a file after a line containing the given keyword.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        found = False
        result_lines = []

        for line in lines:
            if found:
                result_lines.append(line.strip())
            elif keyword.lower() in line.lower():
                found = True

        return result_lines
    except FileNotFoundError:
        return []

def parse_line(line):
    """
    Parses a single line of ClaRNA output and returns base pair info.
    """
    if not line.strip():
        return (False, None)

    split_line = line.split()
    try:
        residue1 = split_line[1][0] + split_line[2]
        chain1 = residue1[0]
        pos1 = residue1[1:]

        residue2 = split_line[5][0] + split_line[4]
        chain2 = residue2[0]
        pos2 = residue2[1:]

        base1 = split_line[3][0]
        base2 = split_line[3][2]

        interaction = split_line[6]
        if "cis" in interaction:
            interaction = "c" + interaction.split("_")[0]
        elif "tran" in interaction:
            interaction = "t" + interaction.split("_")[0]
        else:
            return (False, None)

        score = split_line[-1].split("=")[1].split("/")[0]

        return (True, [chain1, pos1, chain2, pos2, base1, base2, interaction, score])

    except Exception as e:
        print(f"Error parsing line: {line}\n{e}")
        return (False, None)

def parse_clarna_annotation(file_path, output_dir):
    """
    Parses a ClaRNA .out file and saves results to a CSV.
    """
    lines = get_lines_after_keyword(file_path, "find contacts ..")
    pdb_filename = os.path.basename(file_path)
    pdb_id = os.path.splitext(pdb_filename)[0]
    output_csv = os.path.join(output_dir, f"{pdb_id}.csv")

    seen_pairs = set()
    results = []

    for line in lines:
        if line.strip().startswith("find contacts .. DONE"):
            break

        parsed = parse_line(line)
        if not parsed[0]:
            continue

        chain1, pos1, chain2, pos2, nt1, nt2, label, score = parsed[1]
        key = (chain1, pos1, chain2, pos2)
        reverse_key = (chain2, pos2, chain1, pos1)

        if key in seen_pairs or reverse_key in seen_pairs:
            continue  # Skip repeated pairs

        seen_pairs.add(key)
        results.append([chain1, pos1, chain2, pos2, nt1, nt2, label, score])

    # Write to CSV
    os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["chain1", "pos1", "chain2", "pos2", "nt1", "nt2", "label", "Score"])
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description="Parse ClaRNA base pair annotations.")
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory containing ClaRNA .out files")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory to save parsed CSV files")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and file.endswith(".out"):
            print(f"Parsing {file}...")
            parse_clarna_annotation(file_path, output_dir)

if __name__ == "__main__":
    main()
