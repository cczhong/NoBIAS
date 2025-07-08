# RNA Base-Pair Annotation Parsers

This repository provides Python scripts to parse and standardize base-pair annotations from the outputs of five widely used RNA structural analysis tools:

- **ClaRNA**
- **DSSR**
- **FR3D**
- **MC-Annotate**
- **RNAVIEW**

---

## Usage

Each script is located in the `parsers/` directory and supports CLI arguments:

### ClaRNA

```bash
python parsers/clarna_parse.py -i example_data/clarna/ -o parsed_results/clarna/
```

### DSSR

```bash
python parsers/parse_dssr.py -i example_data/dssr/ -o parsed_results/dssr/
```

### FR3D

```bash
python parsers/fr3d.py -i example_data/fr3d/ -o parsed_results/fr3d/
```

### MC-Annotate

```bash
python parsers/mc_annotate.py -i example_data/mc_annotate/ -o parsed_results/mc_annotate/
```

### RNAVIEW

```bash
python parsers/rnaview.py -i example_data/rnaview/ -o parsed_results/rnaview/
```

---

## Requirements

- Python 3.7+
- `pandas`

---
