import re
import csv
import os

# === Hardcoded input/output directories ===
in_directory = "/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/ClaRNA_out_1120"
out_directory = "/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/Parsed_CR_stackings"

def get_lines_after_keyword(file_path, keyword):
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

def parse_line_stack(line):
    # Only parse if there is '>' or '<' in the line
    if '>' not in line and '<' not in line:
        return (False, None)
    
    # Find the first group of only > and < (arrows, 1 to 3 in a row)
    arrow_match = re.search(r'([<>]{1,3})', line)
    label = arrow_match.group(1) if arrow_match else ''
    if not label:
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

        # Score: extract like [score=0.982/0.077,...]
        score_match = re.search(r'\[score=([0-9\.]+)/', line)
        score = score_match.group(1) if score_match else ''

        return (True, [chain1, pos1, chain2, pos2, base1, base2, label, score])
    except Exception as e:
        print(f"Error parsing line: {line}\n{e}")
        return (False, None)

def parse_clarna_stack(file_path, output_path):
    lines = get_lines_after_keyword(file_path, "find contacts ..")
    pdb_filename = os.path.basename(file_path)
    pdb_id = os.path.splitext(pdb_filename)[0]  # Remove .out extension
    output_csv = os.path.join(output_path, f"{pdb_id}.csv")

    seen_pairs = set()
    results = []

    for line in lines:
        if line.strip().startswith("find contacts .. DONE"):
            break

        # Only parse lines with stacking arrows
        result = parse_line_stack(line)
        if result is None or result[0] is False:
            continue

        chain1, pos1, chain2, pos2, nt1, nt2, label, score = result[1]
        key = (chain1, pos1, chain2, pos2)
        reverse_key = (chain2, pos2, chain1, pos1)

        if key in seen_pairs or reverse_key in seen_pairs:
            continue  # Skip repetitions

        seen_pairs.add(key)
        results.append([chain1, pos1, chain2, pos2, nt1, nt2, label, score])

    # Write to CSV
    os.makedirs(output_path, exist_ok=True)
    with open(output_csv, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(["chain1", "pos1", "chain2", "pos2", "nt1", "nt2", "label", "Score"])
        writer.writerows(results)

def main():
    files = os.listdir(in_directory)
    for file in files:
        if not file.endswith(".out"):
            continue
        parse_clarna_stack(os.path.join(in_directory, file), out_directory)

if __name__ == "__main__":
    main()
