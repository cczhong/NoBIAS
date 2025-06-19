# RNA Base Pair Coordinate Extractor

This tool extracts 3D atomic coordinates and RGB color features of RNA base pairs from mmCIF structures using Biopython.
## Features

- Downloads RNA structures from the PDB
- Parses base pair annotation CSVs (e.g., DSSR)
- Extracts atomic coordinates for paired residues
- Assigns RGB colors to atoms using Jmol standards
- Organizes output by base pair type

## File Structure

```
rna_coord_extractor/
├── src/
│   ├── main.py
│   ├── pdb_utils.py
│   ├── structure_utils.py
│   ├── io_utils.py
│   ├── swap_labels.py
│   └── color_code_atoms.py
├── data/
│   └── jmolcolors.csv
├── examples
├── requirements.txt
└── Readme.md
```

## Usage

From the project root:

```bash
python -m src.main \
  --input_csv /path/to/dssr_folder \
  --output_dir ./output \
  --pdb_dir ./pdbs
```

The `input_csv` folder should contain CSVs named like `1a1t_dssr.csv`, with columns:
- `chain1`, `pos1`, `chain2`, `pos2`, `nt1`, `nt2`, `bp1`, `bp2`, `label`

Output files will be saved to `output/<bp_type>/` as:

```
1a1t_A_12_B_22_G_C.txt
1a1t_A_12_B_22_G_C_label.txt
```

## Dependencies

- Python 3.8+
- Biopython
- Pandas
- NumPy