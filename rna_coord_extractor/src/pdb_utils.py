from Bio.PDB import PDBList
from Bio.PDB.MMCIFParser import MMCIFParser
import os

def retrieve_pdb(pdb_id, pdb_dir):
    """
    Downloads and parses the mmCIF structure for a given PDB ID.

    Args:
        pdb_id (str): The PDB ID.
        pdb_dir (str): Directory where PDB files are stored.

    Returns:
        structure (Bio.PDB.Structure.Structure): Parsed RNA structure.
    """
    os.makedirs(pdb_dir, exist_ok=True)
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='mmCif')

    cif_path = os.path.join(pdb_dir, f"{pdb_id.lower()}.cif")
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(pdb_id, cif_path)
