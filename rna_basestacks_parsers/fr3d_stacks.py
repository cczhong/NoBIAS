import os
import pandas as pd

def parse_out_file(file_name):
    nt1_values = []
    nt2_values = []
    labels = []

    with open(file_name, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) >= 6:
                label = values[2].strip('"')  # Use the third column for stacking
                if label:  # Only retain if label is not empty
                    try:
                        nt1_values.append(values[0].strip('"'))
                        nt2_values.append(values[5].strip('"'))
                        labels.append(label)
                    except IndexError:
                        pass

    if not nt1_values:
        return pd.DataFrame()  # Return empty if no data

    df = pd.DataFrame({
        'chain1': [x.split('|')[2] for x in nt1_values],
        'pos1':   [x.split('|')[4] for x in nt1_values],
        'chain2': [x.split('|')[2] for x in nt2_values],
        'pos2':   [x.split('|')[4] for x in nt2_values],
        'nt1':    [x.split('|')[3] for x in nt1_values],
        'nt2':    [x.split('|')[3] for x in nt2_values],
        'label':  labels
    })

    # Drop exact duplicates first
    df = df.drop_duplicates(subset=['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'label'])

    # Create a mirror key ignoring direction and label
    df['mirror_key'] = df.apply(
        lambda row: '|'.join(sorted([row['pos1'], row['pos2']])) + '|' +
                    '|'.join(sorted([row['nt1'], row['nt2']])), axis=1
    )

    # Keep only the first occurrence of each mirror pair
    df = df.drop_duplicates(subset='mirror_key', keep='first')
    df = df.drop(columns='mirror_key')

    return df

# === Paths ===
input_folder = '/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/FR3D/'
output_folder = '/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw/Parsed_fr3d_stackings/'

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.anno'):
        print(f"Processing file: {file_name}")  # Print before parsing
        input_file = os.path.join(input_folder, file_name)
        try:
            df = parse_out_file(input_file)
            output_file = os.path.join(output_folder, f"{file_name[:4]}.csv")
            df.to_csv(output_file, index=False)
        except Exception as e:
            print(f" Error while processing {file_name}: {e}")

print("All files processed and saved.")
