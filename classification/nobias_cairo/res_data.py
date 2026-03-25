import os
import csv
import random
import logging
import warnings
import traceback
import multiprocessing
import concurrent.futures

from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial import cKDTree

import cairo

ENABLE_LOGGING = True
TARGET_SAMPLES_PER_CLASS = 100000
NUM_WORKERS = os.cpu_count() or 18 # I have 20 and run other stuff so cores-2

IMG_HEIGHT = 128
IMG_WIDTH = 128 * 3
PANEL_WIDTH = 128

warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)
if ENABLE_LOGGING and not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hex_to_rgb_float(h): 
    return tuple(int(h.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))

JMOL_COLORS_HEX = {'C':"909090",'N':"3050F8",'O':"FF0D0D",'P':"FF8000",'S':"FFFF30",'DEFAULT':"808080"}
JMOL_COLORS_RGB_FLOAT = {k: hex_to_rgb_float(v) for k, v in JMOL_COLORS_HEX.items()}
DEFAULT_ATOM_COLOR_FLOAT = JMOL_COLORS_RGB_FLOAT['DEFAULT']

PDB_DATA_CACHE = {}

def init_worker(cache_data):
    """This function is called once per worker process to initialize its global cache."""
    global PDB_DATA_CACHE
    PDB_DATA_CACHE = cache_data

# help func

def get_resolution_from_mmcif(pdb_file):
    try:
        mmcif_dict = MMCIF2Dict(pdb_file)
        res_str = mmcif_dict.get('_refine.ls_d_res_high', [None])[0] or \
                  mmcif_dict.get('_em_3d_reconstruction.resolution', [None])[0]
        if res_str and res_str not in ['.', '?']: return float(res_str)
    except Exception:
        pass
    
    return None

def get_bonds(base_residue):
    """
    This function now correctly isolates only the atoms of the base's aromatic ring,
    excluding the sugar-phosphate backbone.
    """
    atoms = list(base_residue.get_atoms())
    
    ring_atoms = [a for a in atoms if a.element in ('C', 'N') and a.name[0] in ('C', 'N') and "'" not in a.name]

    if len(ring_atoms) < 2: return [], ring_atoms
    coords = np.array([a.coord for a in ring_atoms])
    kdtree = cKDTree(coords)
    bonds = []
    for i in range(len(ring_atoms)):
        for j_idx in kdtree.query_ball_point(coords[i], r=1.8):
            if i < j_idx and 1.1 < np.linalg.norm(coords[i] - coords[j_idx]) < 1.7:
                bonds.append((i, j_idx))
    return bonds, ring_atoms

def rotation_matrix_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a, (x, y, z) = np.cos(angle / 2.0), -axis * np.sin(angle / 2.0)
    return np.array([
        [a*a+x*x-y*y-z*z, 2*(x*y-a*z), 2*(x*z+a*y)],
        [2*(x*y+a*z), a*a+y*y-x*x-z*z, 2*(y*z-a*x)],
        [2*(x*z-a*y), 2*(y*z+a*x), a*a+z*z-x*x-y*y]
    ])

# STAGE 1: PRE-LOADING AND CACHING (Worker function)
def preload_pdb_data(pdb_id, pdb_files_dir):
    cif_file_path = os.path.join(pdb_files_dir, f"{pdb_id}.cif")
    if not os.path.exists(cif_file_path):
        return pdb_id, None
    try:
        structure = PDB.MMCIFParser(QUIET=True).get_structure(pdb_id, cif_file_path)
        resolution = get_resolution_from_mmcif(cif_file_path)
        residue_data = {}
        for residue in structure.get_residues():
            res_id = residue.get_id()
            full_id = (residue.get_parent().id, (res_id[0].strip(), res_id[1], res_id[2].strip()))
            bonds, ring_atoms = get_bonds(residue)
            if not ring_atoms: continue
            residue_data[full_id] = {
                'coords': np.array([a.coord for a in ring_atoms]),
                'elements': [a.element for a in ring_atoms], 'bonds': bonds,
            }
        return pdb_id, {'resolution': resolution, 'residues': residue_data}
    except Exception:
        return pdb_id, None

# ==============================================================================
# HIGH-PERFORMANCE IMAGE GENERATION (Function remains the same)
# ==============================================================================
def generate_image_pycairo_from_preprocessed(res1_data, res2_data, resolution, output_path):
    
    all_coords = np.vstack([res1_data['coords'], res2_data['coords']])
    all_elements = res1_data['elements'] + res2_data['elements']
    center_of_mass = np.mean(all_coords, axis=0)
    centered_coords = all_coords - center_of_mass
    
    R_view1 = rotation_matrix_from_axis_angle(np.array([0, 1, 0]), np.pi / 2)
    R_view2 = np.eye(3)
    R_view3 = rotation_matrix_from_axis_angle(np.array([1, 0, 0]), np.pi / 2)
    
    views = [centered_coords @ R.T for R in [R_view1, R_view2, R_view3]]
    
    all_projected_xy = np.vstack([v[:, :2] for v in views])
    min_coord, max_coord = np.min(all_projected_xy, axis=0), np.max(all_projected_xy, axis=0)
    
    span = np.max(max_coord - min_coord)
    scale = (PANEL_WIDTH - 12.0) / span if span > 1e-6 else 1.0
    min_lw, max_lw = 0.8, 2.2
    
    if resolution is None or resolution > 3.5: 
        bond_width = min_lw
    elif resolution < 1.0: 
        bond_width = max_lw
    else: 
        bond_width = min_lw + (3.5 - resolution) / 2.5 * (max_lw - min_lw)
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMG_WIDTH, IMG_HEIGHT)
    
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1); ctx.paint()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width(bond_width)
    
    offset1 = len(res1_data['elements'])
    
    bonds1 = res1_data['bonds']
    bonds2 = [(i + offset1, j + offset1) for i, j in res2_data['bonds']]
    
    for i, proj_coords in enumerate(views):
        ctx.save()
        
        offset_x = -min_coord[0] * scale + 6.0 + (i * PANEL_WIDTH)
        offset_y = -min_coord[1] * scale + 6.0
        
        pixel_coords = (proj_coords[:, :2] * scale) + [offset_x, offset_y]
        
        for idx1, idx2 in bonds1 + bonds2:
            x1, y1 = pixel_coords[idx1]; x2, y2 = pixel_coords[idx2]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            color1 = JMOL_COLORS_RGB_FLOAT.get(all_elements[idx1], DEFAULT_ATOM_COLOR_FLOAT)
            ctx.set_source_rgb(*color1); ctx.move_to(x1, y1); ctx.line_to(mid_x, mid_y); ctx.stroke()
            color2 = JMOL_COLORS_RGB_FLOAT.get(all_elements[idx2], DEFAULT_ATOM_COLOR_FLOAT)
            ctx.set_source_rgb(*color2); ctx.move_to(mid_x, mid_y); ctx.line_to(x2, y2); ctx.stroke()
        
        base_radius = 3.0
        
        for atom_idx, element in enumerate(all_elements):
            x, y = pixel_coords[atom_idx]
            color = JMOL_COLORS_RGB_FLOAT.get(element, DEFAULT_ATOM_COLOR_FLOAT)
            ctx.set_source_rgb(*color)
            radius = base_radius / 2.0 if element == 'C' else base_radius
            ctx.arc(x, y, radius, 0, 2 * np.pi); ctx.fill()
        
        ctx.restore()
    
    surface.write_to_png(output_path)
    
    return True

# ==============================================================================
# STAGE 2: WORKER FUNCTION (Now correctly accesses the initialized cache)
# ==============================================================================
def find_residue_in_cache(residue_cache, chain_id, res_num):
    common_key = (chain_id, ('', res_num, ''))
    
    if common_key in residue_cache: 
        return residue_cache[common_key]
    
    for key, residue_data in residue_cache.items():
        if key[0] == chain_id and key[1][1] == res_num: 
            return residue_data
    
    return None

def process_image_job(job_data):
    
    mol_name = job_data['mol_name']
    
    try:
        structure_data = PDB_DATA_CACHE.get(mol_name)
        if not structure_data:
            return f"FAIL: PDB data for '{mol_name}' not found in worker's cache."

        residue_cache = structure_data.get('residues')
        if not residue_cache:
             return f"FAIL: PDB '{mol_name}' is in cache, but has 0 processable residues."

        b1_id_tuple, b2_id_tuple = job_data['b1_id'], job_data['b2_id']
        res1_data = find_residue_in_cache(residue_cache, b1_id_tuple[0], b1_id_tuple[1])
        res2_data = find_residue_in_cache(residue_cache, b2_id_tuple[0], b2_id_tuple[1])

        if not res1_data or not res2_data:
            fail_msg = f"FAIL: Residue lookup failed for PDB '{mol_name}'.\n"
            available_chains = sorted(list(set(k[0] for k in residue_cache.keys())))
            fail_msg += f"    Available chain IDs in cache: {available_chains}\n"
            if not res1_data: fail_msg += f"    > COULD NOT FIND: {b1_id_tuple}.\n"
            if not res2_data: fail_msg += f"    > COULD NOT FIND: {b2_id_tuple}.\n"
            return fail_msg

        generate_image_pycairo_from_preprocessed(
            res1_data, res2_data, structure_data['resolution'], job_data['output_path']
        )
    
        return "SUCCESS"
    
    except Exception:
        return f"FAIL: UNEXPECTED EXCEPTION on PDB '{mol_name}'. Traceback: {traceback.format_exc()}"

def gather_data_from_csv(csv_files_list):
    
    all_data = []
    
    for csv_file in csv_files_list:
        
        if not os.path.exists(csv_file):
            logging.warning(f"Source file not found: {csv_file}"); continue
        
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mol_name = (row.get('mol_name') or row.get('name')).strip()
                    c1, p1s = row['chain1'].strip(), (row.get('position1') or row.get('pos1')).strip()
                    c2, p2s = row['chain2'].strip(), (row.get('position2') or row.get('pos2')).strip()
                    p1, p2 = int(float(p1s)), int(float(p2s))
                    all_data.append({'mol_name': mol_name, 'b1_id': (c1, p1), 'b2_id': (c2, p2)})
                except (KeyError, ValueError, TypeError): 
                    continue

    return all_data

def main():

    BATCH_SIZE = 100
    try: 
        script_dir_path = os.path.dirname(os.path.abspath(__file__))
    except NameError: 
        script_dir_path = os.getcwd()

    pdb_structure_files_dir = os.path.join(script_dir_path, "../../cif")
    base_pair_csv_files = [os.path.join(script_dir_path, f) for f in ["clarna.csv", "dssr.csv", "fr3d.csv"]]
    stack_csv_file = [os.path.join(script_dir_path, "consolidated.csv")]
    base_output_images_dir = os.path.join(script_dir_path, "final_imgs_send_smriti")
    pair_output_dir = os.path.join(base_output_images_dir, "pair")
    stack_output_dir = os.path.join(base_output_images_dir, "stack")
    
    all_pairs = gather_data_from_csv(base_pair_csv_files)
    all_stacks = gather_data_from_csv(stack_csv_file)
    
    random.shuffle(all_pairs); random.shuffle(all_stacks)
    pairs_to_process = all_pairs[:TARGET_SAMPLES_PER_CLASS]
    stacks_to_process = all_stacks[:TARGET_SAMPLES_PER_CLASS]
    logging.info(f"Gathered {len(pairs_to_process)} pairs and {len(stacks_to_process)} stacks.")
    
    all_jobs_for_generation = pairs_to_process + stacks_to_process
    jobs_by_pdb = defaultdict(list)
    
    for job in all_jobs_for_generation: 
        jobs_by_pdb[job['mol_name']].append(job)
    
    unique_pdb_ids = sorted(list(jobs_by_pdb.keys()))
    logging.info(f"Found {len(unique_pdb_ids)} unique PDB structures to process.")

    num_batches = (len(unique_pdb_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        pdb_id_batch = unique_pdb_ids[start_index:end_index]
        logging.info(f"\n--- PROCESSING BATCH {i+1}/{num_batches} ({len(pdb_id_batch)} PDBs) ---")
        
        # Step 1: Populate a temporary cache in the main process
        temp_cache = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(preload_pdb_data, pdb_id, pdb_structure_files_dir) for pdb_id in pdb_id_batch]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(pdb_id_batch), desc=f"Pre-loading Batch {i+1}"):
                pdb_id, data = future.result()
                if data: temp_cache[pdb_id] = data
        logging.info(f"Pre-loaded data for {len(temp_cache)} PDBs into main process memory.")

        # Step 2: Process images, explicitly initializing workers with the temporary cache
        for interaction_type, data_list, out_dir in [("PAIRS", pairs_to_process, pair_output_dir), ("STACKS", stacks_to_process, stack_output_dir)]:
            if not data_list: continue
            os.makedirs(out_dir, exist_ok=True)
            batch_job_set = set(pdb_id_batch)
            image_jobs_for_batch = []
            for row in data_list:
                if row['mol_name'] in batch_job_set:
                    b1_str, b2_str = f"{row['b1_id'][0]}{row['b1_id'][1]}", f"{row['b2_id'][0]}{row['b2_id'][1]}"
                    output_path = os.path.join(out_dir, f"{row['mol_name']}_{b1_str}_{b2_str}.png")
                    if not os.path.exists(output_path):
                        job_with_path = row.copy()
                        job_with_path['output_path'] = output_path
                        image_jobs_for_batch.append(job_with_path)

            if not image_jobs_for_batch: continue
            logging.info(f"Processing {len(image_jobs_for_batch)} potential images for {interaction_type}...")

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                initializer=init_worker,
                initargs=(temp_cache,)
            ) as executor:
                results = list(tqdm(executor.map(process_image_job, image_jobs_for_batch, chunksize=200), total=len(image_jobs_for_batch), desc=f"Images (Batch {i+1})"))
            
            successes = sum(1 for r in results if r == "SUCCESS")
            failures = [r for r in results if r != "SUCCESS"]
            
            logging.info(f"Batch section complete. Generated: {successes} images. Failed/Skipped: {len(failures)} jobs.")
            if failures:
                logging.warning("--- SAMPLE OF FAILURES FOR THIS BATCH ---")
                unique_failures = sorted(list(set(failures)))
                for fail_msg in unique_failures[:10]: print(fail_msg)
                if len(unique_failures) > 10: print(f"... and {len(unique_failures) - 10} other unique error types.")

    logging.info("\n=== ALL TASKS COMPLETED ===")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
