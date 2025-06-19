# RNA base stacking interactions

This code extracts annotations from mutiple annotation sources, conforms them to one format, generates imagery and trains a CNN on base-stack classification

## File structure

## Usage

#### Annotation downloads

All the annotations and the cifs are in the git LFS on ML platform [Huggingface](https://huggingface.co), all the files are avalible [here](https://huggingface.co/datasets/VatsaDev/BioD2/tree/main), but there is a download script

```bash
python download_anno.py
```

#### Parsing

There is currently one parsing file for each data source, they were built for diff formats and some of the functions should be the same, yet have variance, combining the all into one `parse.py` is a future step

```bash
python ClaRNAdata.py
python DSSRdata.py
python FR3Ddata.py
python MCAdata.py
```

# To be added from here on out (small bugfixes)

#### Grouping

takes the four CSVs as input outputs the following:

 - pass_all.csv: bases with same label across all files
 - pass_3.csv: bases with same label across at least 3 files
 - pass_2.csv: bases with same label across at least 2 files

Also outputs csvs comparing files in groups of 2 or 3, all to folder comparision_results 

```bash
python grouping.py
```

#### Image Gen

use `image_gen.py` with csv of choice

```bash
python image_gen.py --csv_file pass_all.csv 
```

#### Training

Quick step to link images to classes, use `python linker.py`

```bash
python linker.py 
```

Actual training colab (EDITABLE, CLONE IT) -> [link](https://colab.research.google.com/drive/1MwHMRlSXgSpXjup0j0BKFbSluFzhJPeU?usp=sharing)
