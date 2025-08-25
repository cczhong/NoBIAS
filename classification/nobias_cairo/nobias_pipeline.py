import os
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import logging
import argparse
import time
import io

# Parallel Processing Imports
import concurrent.futures
from tqdm import tqdm

# Image Generation & Classification Imports
import warnings
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
try:
    import cairo
    from scipy.spatial import cKDTree
except ImportError as e:
    print(f"CRITICAL ERROR: A required library is not installed: {e.name}")
    print("Please run: 'pip install pycairo scipy'")
    exit()

# --- Overall Configuration ---
XGB_MODEL_PATH = "binary_interaction_classifier_xgb_v2_high_recall.json" 
RESNET_MODEL_PATH = "resnet18_classifier_stable_high_recall.pth"
VALID_RESNAMES = ['A', 'C', 'G', 'U', 'I']
NUM_WORKERS = 18
NEIGHBOR_RADIUS = 20.0 # Pre-filtering cutoff in Angstroms maybe try pushing 25???

# --- Logging and Warning Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)
logging.getLogger('PIL').setLevel(logging.WARNING)

# --- Color and Style Constants for PyCairo ---
JMOL_COLORS_HEX = {'C':"909090",'N':"3050F8",'O':"FF0D0D",'P':"FF8000",'S':"FFFF30",'DEFAULT':"808080"}
def hex_to_rgb_float(h): return tuple(int(h.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))
JMOL_COLORS_RGB_FLOAT = {k: hex_to_rgb_float(v) for k, v in JMOL_COLORS_HEX.items()}
IMG_HEIGHT, IMG_WIDTH, PANEL_WIDTH = 128, 384, 128

# ==============================================================================
# WORKER INITIALIZATION AND GLOBAL STORAGE
# ==============================================================================
worker_data = {}

def init_worker(structure_path, resnet_model_path):
    global worker_data
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure("worker_structure", structure_path)
        resolution = MMCIF2Dict(structure_path).get('_refine.ls_d_res_high', [None])[0]
        structure.resolution = float(resolution) if resolution and resolution not in ['.', '?'] else None
        worker_data['structure'] = structure
    except Exception as e:
        logging.error(f"Worker failed to load structure '{structure_path}': {e}"); return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(resnet_model_path, map_location=device))
        model.to(device); model.eval()
        worker_data.update({'resnet_model': model, 'device': device})
    except Exception as e:
        logging.error(f"Worker failed to load ResNet '{resnet_model_path}': {e}"); return

# ==============================================================================
# CORE LOGIC (HELPER FUNCTIONS)
# ==============================================================================
def find_residue_in_worker(res_id_tuple):
    structure = worker_data.get('structure')
    if not structure: return None
    chain_id, res_num, icode = res_id_tuple[0], res_id_tuple[1], (res_id_tuple[2] or ' ').strip()
    try:
        chain = structure[0][chain_id]
        if (' ', res_num, icode) in chain: return chain[(' ', res_num, icode)]
        for res in chain:
            if res.id[1] == res_num and res.id[2].strip() == icode: return res
    except KeyError: pass
    return None

def get_base_plane(base_residue):
    coords = [a.coord for a in base_residue if a.element in ('C', 'N') and a.name[0] in ('C', 'N')]
    if len(coords) < 3: return None, None
    center = np.mean(coords, axis=0)
    try:
        _, _, vh = np.linalg.svd(np.array(coords) - center)
        return center, vh[2, :] / np.linalg.norm(vh[2, :])
    except: return None, None

def get_bonds(base_residue):
    atoms = list(base_residue.get_atoms())
    ring_atoms = [a for a in atoms if a.element in ('C', 'N') and a.name[0] in ('C', 'N')]
    if len(ring_atoms) < 2: return [], ring_atoms
    coords = np.array([a.coord for a in ring_atoms])
    bonds, kdtree = [], cKDTree(coords)
    for i, atom1 in enumerate(ring_atoms):
        for j_idx in kdtree.query_ball_point(coords[i], r=1.8):
            if i < j_idx and 1.1 < np.linalg.norm(atom1.coord - ring_atoms[j_idx].coord) < 1.7:
                bonds.append((atom1, ring_atoms[j_idx]))
    return bonds, ring_atoms

def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a, (x, y, z) = np.cos(angle / 2.0), -axis * np.sin(angle / 2.0)
    return np.array([[a*a+x*x-y*y-z*z,2*(x*y-a*z),2*(x*z+a*y)],[2*(x*y+a*z),a*a+y*y-x*x-z*z,2*(y*z-a*x)],[2*(x*z-a*y),2*(y*z+a*x),a*a+z*z-x*x-y*y]])

def extract_all_features(res1, res2):
    center1, normal1 = get_base_plane(res1)
    center2, normal2 = get_base_plane(res2)
    if any(v is None for v in [center1, center2, normal1, normal2]): return None
    anchor_dist = np.linalg.norm(center1 - center2)
    inter_vec = center2 - center1;
    if np.allclose(inter_vec, 0): return None
    dot_orient = np.clip(np.dot(normal1, inter_vec / np.linalg.norm(inter_vec)),-1.,1.)
    orientation_angle = min(np.degrees(np.arccos(dot_orient)), 180-np.degrees(np.arccos(dot_orient)))
    dot_normal = np.clip(np.dot(normal1, normal2),-1.,1.); normal_angle = np.degrees(np.arccos(dot_normal))
    is_sequential = 1 if (res1.parent.id == res2.parent.id and abs(res1.id[1]-res2.id[1])==1) else 0
    return [anchor_dist, orientation_angle, normal_angle, is_sequential]

def generate_base_image_pycairo(base1_res, base2_res, resolution):
    bonds1, atoms1 = get_bonds(base1_res); bonds2, atoms2 = get_bonds(base2_res)
    all_atoms = atoms1 + atoms2
    if not all_atoms: return None
    all_coords = np.array([a.coord for a in all_atoms]); centered_coords = all_coords - np.mean(all_coords, axis=0)
    views = [centered_coords @ R.T for R in [rotation_matrix(np.array([0,1,0]),np.pi/2),np.eye(3),rotation_matrix(np.array([1,0,0]),np.pi/2)]]
    all_xy = np.vstack([v[:,:2] for v in views]); min_c, max_c = np.min(all_xy, axis=0), np.max(all_xy, axis=0)
    scale = (PANEL_WIDTH-12.)/np.max(max_c-min_c) if np.max(max_c-min_c)>1e-6 else 1.
    min_lw, max_lw = 0.8, 2.2
    bond_w = min_lw if resolution is None or resolution>3.5 else max_lw if resolution<1.0 else min_lw+(3.5-resolution)/2.5*(max_lw-min_lw)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMG_WIDTH, IMG_HEIGHT)
    ctx = cairo.Context(surface); ctx.set_source_rgb(1,1,1); ctx.paint(); ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.set_line_width(bond_w)
    for i, proj_coords in enumerate(views):
        ctx.save(); px_coords = proj_coords[:,:2]*scale + (-min_c*scale+6.+[i*PANEL_WIDTH,0])
        for a1,a2 in bonds1+bonds2:
            (x1,y1), (x2,y2) = px_coords[all_atoms.index(a1)], px_coords[all_atoms.index(a2)]; (mx,my) = (x1+x2)/2,(y1+y2)/2
            ctx.set_source_rgb(*JMOL_COLORS_RGB_FLOAT.get(a1.element)); ctx.move_to(x1,y1); ctx.line_to(mx,my); ctx.stroke()
            ctx.set_source_rgb(*JMOL_COLORS_RGB_FLOAT.get(a2.element)); ctx.move_to(mx,my); ctx.line_to(x2,y2); ctx.stroke()
        for i_atom, atom in enumerate(all_atoms):
            x,y = px_coords[i_atom]; ctx.set_source_rgb(*JMOL_COLORS_RGB_FLOAT.get(atom.element))
            ctx.arc(x,y, 3. if atom.element!='C' else 1.5, 0, 2*np.pi); ctx.fill()
        ctx.restore()
    buf = io.BytesIO(); surface.write_to_png(buf); buf.seek(0)
    return Image.open(buf).convert("RGB")

def classify_image_from_memory(pil_image):
    model, device = worker_data['resnet_model'], worker_data['device']
    if not pil_image: return "IMAGE_FAILED"
    transform = transforms.Compose([transforms.Resize((128,384)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img_t = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad(): pred = torch.sigmoid(model(img_t)).item()
    return 'stack' if pred > 0.5 else 'pair'

# ==============================================================================
# WORKER TASK FUNCTIONS
# ==============================================================================
def process_feature_task(id_pair):
    res1, res2 = find_residue_in_worker(id_pair[0]), find_residue_in_worker(id_pair[1])
    if res1 and res2:
        features = extract_all_features(res1, res2)
        if features: return {'id1':id_pair[0], 'id2':id_pair[1], 'res1':res1.resname, 'res2':res2.resname, 'features':features}
    return None

def process_classify_task(id_pair):
    res1, res2 = find_residue_in_worker(id_pair[0]), find_residue_in_worker(id_pair[1])
    resolution = worker_data['structure'].resolution
    pil_image = generate_base_image_pycairo(res1, res2, resolution)
    return {'id1': id_pair[0], 'id2': id_pair[1], 'class': classify_image_from_memory(pil_image)}

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_pipeline(structure_path):
    start_time = time.time()
    mol_name = os.path.splitext(os.path.basename(structure_path))[0]
    abs_path = os.path.abspath(structure_path)
    logging.info(f"--- Starting FAST pipeline for {mol_name} with K-D Tree pre-filtering ---")

    # --- 1. K-D Tree pre-filtering (THE 1000x SPEEDUP) ---
    parser = PDB.MMCIFParser(QUIET=True); main_struct = parser.get_structure(mol_name, abs_path)
    
    # Get all valid RNA residues and their centers
    residues = [r for m in main_struct for c in m for r in c if r.id[0]==' ' and r.resname.strip() in VALID_RESNAMES]
    centers, valid_residues = [], []
    for r in residues:
        center, _ = get_base_plane(r)
        if center is not None:
            centers.append(center)
            valid_residues.append(r)
    
    if len(valid_residues) < 2:
        logging.warning("Fewer than 2 valid RNA residues found. Exiting.")
        return

    # Build the K-D tree and query for neighboring pairs
    kdtree = cKDTree(np.array(centers))
    candidate_indices = kdtree.query_pairs(r=NEIGHBOR_RADIUS)

    # Convert indices back to lightweight residue ID tuples
    id_pairs = set()
    for i, j in candidate_indices:
        res1, res2 = valid_residues[i], valid_residues[j]
        id1 = (res1.parent.id, res1.id[1], res1.id[2].strip())
        id2 = (res2.parent.id, res2.id[1], res2.id[2].strip())
        # Add as a sorted tuple to ensure uniqueness
        id_pairs.add(tuple(sorted((id1, id2))))
    
    id_pairs = list(id_pairs)
    logging.info(f"K-D Tree reduced {len(residues)*(len(residues)-1)//2} total possible pairs to {len(id_pairs)} candidates within {NEIGHBOR_RADIUS}Å.")
    
    # --- The rest of the pipeline now runs on the drastically smaller candidate list ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker, initargs=(abs_path, RESNET_MODEL_PATH)) as executor:
        logging.info("Step 1: Extracting features for candidate pairs...")
        all_pairs_data = list(tqdm(executor.map(process_feature_task, id_pairs, chunksize=500), total=len(id_pairs), desc="Extracting Features"))
        all_pairs_data = [r for r in all_pairs_data if r]
        
        logging.info("Step 2: Predicting interactions with XGBoost...")
        xgb_model = xgb.XGBClassifier(); xgb_model.load_model(XGB_MODEL_PATH)
        feature_vectors = np.array([d['features'] for d in all_pairs_data])
        predictions = xgb_model.predict(feature_vectors)
        
        interacting_id_pairs = [ (d['id1'], d['id2']) for i, d in enumerate(all_pairs_data) if predictions[i] == 0 ]
        logging.info(f"XGBoost found {len(interacting_id_pairs)} potential interactions.")
        if not interacting_id_pairs: logging.info("No interactions found."); return

        logging.info("Step 3: Generating images and classifying with ResNet...")
        class_results = list(tqdm(executor.map(process_classify_task, interacting_id_pairs, chunksize=100), total=len(interacting_id_pairs), desc="Classifying Pairs"))
    
    logging.info("Step 4: Assembling final report...")
    class_map = {tuple(sorted((res['id1'], res['id2']))): res['class'] for res in class_results}
    
    final_data = []
    for i, d in enumerate(all_pairs_data):
        if predictions[i] == 0:
            id_tup_sorted = tuple(sorted((d['id1'], d['id2'])))
            final_data.append({
                'mol_name': mol_name, 'chain1': d['id1'][0], 'chain2': d['id2'][0],
                'chain_no': 1 if d['id1'][0]==d['id2'][0] else 2,
                'base/residue_name': f"{d['res1'].strip()}-{d['res2'].strip()}",
                'position1': d['id1'][1], 'position2': d['id2'][1],
                'class': class_map.get(id_tup_sorted, 'UNKNOWN')
            })

    pd.DataFrame(final_data).to_csv(f"{mol_name}_no_bias_anno.csv", index=False)
    logging.info(f"--- Pipeline complete in {time.time()-start_time:.2f}s. Results in {mol_name}_no_bias_anno.csv ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="High-performance RNA interaction analysis pipeline.")
    parser.add_argument("input_file", type=str, help="Path to input PDB or CIF file.")
    args = parser.parse_args()
    if os.path.exists(args.input_file):
        run_pipeline(args.input_file)
    else:
        logging.error(f"Input file not found: {args.input_file}")