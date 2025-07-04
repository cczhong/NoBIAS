import os
import csv
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict # For reading resolution from MMCIF
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, ConvexHull
import multiprocessing
from functools import partial
import concurrent.futures
from tqdm import tqdm
import warnings
import logging
import time
import re
import gc
import argparse

# Configuration
ENABLE_LOGGING = True

warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)

if ENABLE_LOGGING and not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
elif not ENABLE_LOGGING:
    logging.getLogger().setLevel(logging.CRITICAL + 1)

# Updated Jmol Colors from HEX
JMOL_COLORS_HEX = {
    'H': "FFFFFF", 'He': "D9FFFF", 'Li': "CC80FF", 'Be': "C2FF00", 'B': "FFB5B5",
    'C': "909090", 'N': "3050F8", 'O': "FF0D0D", 'F': "90E050", 'Ne': "B3E3F5",
    'Na': "AB5CF2", 'Mg': "8AFF00", 'Al': "BFA6A6", 'Si': "F0C8A0", 'P': "FF8000",
    'S': "FFFF30", 'Cl': "1FF01F", 'Ar': "80D1E3", 'K': "8F40D4", 'Ca': "3DFF00",
    'Sc': "E6E6E6", 'Ti': "BFC2C7", 'V': "A6A6AB", 'Cr': "8A99C7", 'Mn': "9C7AC7",
    'Fe': "E06633", 'Co': "F090A0", 'Ni': "50D050", 'Cu': "C88033", 'Zn': "7D80B0",
    'Ga': "C28F8F", 'Ge': "668F8F", 'As': "BD80E3", 'Se': "FFA100", 'Br': "A62929",
    'Kr': "5CB8D1", 'Rb': "702EB0", 'Sr': "00FF00", 'Y': "94FFFF", 'Zr': "94E0E0",
    'Nb': "73C2C9", 'Mo': "54B5B5", 'Tc': "3B9E9E", 'Ru': "248F8F", 'Rh': "0A7D8C",
    'Pd': "006985", 'Ag': "C0C0C0", 'Cd': "FFD98F", 'In': "A67573", 'Sn': "668080",
    'Sb': "9E63B5", 'Te': "D47A00", 'I': "940094", 'Xe': "429EB0", 'Cs': "57178F",
    'Ba': "00C900", 'La': "70D4FF", 'Ce': "FFFFC7", 'Pr': "D9FFC7", 'Nd': "C7FFC7",
    'Pm': "A3FFC7", 'Sm': "8FFFC7", 'Eu': "61FFC7", 'Gd': "45FFC7", 'Tb': "30FFC7",
    'Dy': "1FFFC7", 'Ho': "00FF9C", 'Er': "00E675", 'Tm': "00D452", 'Yb': "00BF38",
    'Lu': "00AB24", 'Hf': "4DC2FF", 'Ta': "4DA6FF", 'W': "2194D6", 'Re': "267DAB",
    'Os': "266696", 'Ir': "175487", 'Pt': "D0D0E0", 'Au': "FFD123", 'Hg': "B8B8D0",
    'Tl': "A6544D", 'Pb': "575961", 'Bi': "9E4FB5", 'Po': "AB5C00", 'At': "754F45",
    'Rn': "428296", 'Fr': "420066", 'Ra': "007D00", 'Ac': "70ABFA", 'Th': "00BAFF",
    'Pa': "00A1FF", 'U': "008FFF", 'Np': "0080FF", 'Pu': "006BFF", 'Am': "545CF2",
    'Cm': "785CE3", 'Bk': "8A4FE3", 'Cf': "A136D4", 'Es': "B31FD4", 'Fm': "B31FBA",
    'Md': "B30DA6", 'No': "BD0D87", 'Lr': "C70066", 'Rf': "CC0059", 'Db': "D1004F",
    'Sg': "D90045", 'Bh': "E00038", 'Hs': "E6002E", 'Mt': "EB0026",
    'DEFAULT': "808080" # Default gray for unknown elements
}

def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

JMOL_COLORS_RGB = {k: hex_to_rgb(v) for k, v in JMOL_COLORS_HEX.items()}
DEFAULT_ATOM_COLOR_RGB = JMOL_COLORS_RGB['DEFAULT']

parser = argparse.ArgumentParser(description="Process a CSV file to generate molecular interaction images.")
parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file")
args = parser.parse_args()
csv_file = [args.csv_file]

# --- Structure Loading and Caching ---
def get_resolution_from_mmcif(pdb_file):
    try:
        mmcif_dict = MMCIF2Dict(pdb_file)
        if '_refine.ls_d_res_high' in mmcif_dict:
            res_str = mmcif_dict['_refine.ls_d_res_high']
            if isinstance(res_str, list): res_str = res_str[0]
            if res_str and res_str not in ['.', '?']: return float(res_str)
        if '_em_3d_reconstruction.resolution' in mmcif_dict:
            res_str = mmcif_dict['_em_3d_reconstruction.resolution']
            if isinstance(res_str, list): res_str = res_str[0]
            if res_str and res_str not in ['.', '?']: return float(res_str)
    except Exception as e:
        logging.debug(f"Could not parse resolution from {pdb_file}: {e}")
    return None

def load_structure(pdb_file):
    is_cif = pdb_file.lower().endswith(".cif")
    parser = PDB.MMCIFParser(QUIET=True) if is_cif else PDB.PDBParser(QUIET=True)
    structure_id = os.path.splitext(os.path.basename(pdb_file))[0]
    try:
        structure = parser.get_structure(structure_id, pdb_file)
        if structure:
            structure.resolution = get_resolution_from_mmcif(pdb_file) if is_cif else None
        return structure
    except FileNotFoundError: logging.debug(f"File not found: {pdb_file}"); return None
    except Exception as e: logging.error(f"Error loading {pdb_file}: {e}"); return None

_structure_cache, _cache_limit = {}, 20
def load_structure_cached(pdb_file):
    global _structure_cache, _cache_limit
    abs_path = os.path.abspath(pdb_file)
    if abs_path in _structure_cache:
        s = _structure_cache.pop(abs_path); _structure_cache[abs_path] = s; return s
    s = load_structure(abs_path)
    if s:
        if len(_structure_cache) >= _cache_limit:
            try: del _structure_cache[next(iter(_structure_cache))]
            except StopIteration: pass
        _structure_cache[abs_path] = s
    return s

# --- Geometric Functions ---
def rotation_matrix_from_axis_angle(axis, angle):
    if np.isclose(angle, 0): return np.eye(3)
    norm_axis = np.linalg.norm(axis)
    if np.isclose(norm_axis, 0): return np.eye(3)
    axis = axis / norm_axis
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def get_base_plane(base_residue):
    _, ring_atom_objects = get_bonds(base_residue)
    if len(ring_atom_objects) < 3: return None, None
    coords = np.array([atom.coord for atom in ring_atom_objects])
    center = np.mean(coords, axis=0)
    coords_centered = coords - center
    try:
        _, _, vh = np.linalg.svd(coords_centered); normal = vh[2, :]
        return center, normal
    except np.linalg.LinAlgError:
        logging.warning(f"SVD failed for plane calc of {base_residue.get_full_id()}."); return None, None

def get_bonds(base_residue):
    bonds_atom_pairs, atoms = [], list(base_residue.get_atoms())
    ring_atom_objects = [a for a in atoms if a.element in ('C', 'N') and a.name[0] in ('C', 'N')]
    if len(ring_atom_objects) < 2: return [], ring_atom_objects
    coords = np.array([a.coord for a in ring_atom_objects])
    if coords.shape[0] < 2: return [], ring_atom_objects
    kdtree, bonded_pairs_set = cKDTree(coords), set()
    for i, atom1_obj in enumerate(ring_atom_objects):
        for j_idx in kdtree.query_ball_point(coords[i], r=1.8):
            if i == j_idx: continue
            atom2_obj = ring_atom_objects[j_idx]
            pair_ids = tuple(sorted((id(atom1_obj), id(atom2_obj))))
            if pair_ids in bonded_pairs_set: continue
            if 1.1 < np.linalg.norm(atom1_obj.coord - atom2_obj.coord) < 1.7:
                bonds_atom_pairs.append((atom1_obj, atom2_obj)); bonded_pairs_set.add(pair_ids)
    return bonds_atom_pairs, ring_atom_objects

def get_anchor_atom_coord(residue):
    resname = residue.resname.strip().upper()
    anchor_atom = None
    if resname in ['A', 'G', 'I']: anchor_atom = next((a for a in residue.get_atoms() if a.name == 'N9'), None)
    elif resname in ['C', 'U', 'T']: anchor_atom = next((a for a in residue.get_atoms() if a.name == 'N1'), None)
    
    if anchor_atom is not None: 
        return anchor_atom.coord

    plane_center, _ = get_base_plane(residue)
    if plane_center is not None:
        logging.debug(f"N1/N9 not found for {residue.get_full_id()}, using ring plane center.")
        return plane_center
    
    _, ring_atoms_list = get_bonds(residue)
    if ring_atoms_list:
        logging.debug(f"Using geometric center of identified ring atoms for {residue.get_full_id()}.")
        return np.mean([a.coord for a in ring_atoms_list], axis=0)
    
    all_atoms = list(residue.get_atoms())
    if all_atoms:
        logging.warning(f"No ring atoms for {residue.get_full_id()}, using center of all its atoms.");
        return np.mean([a.coord for a in all_atoms], axis=0)
    logging.error(f"Cannot get anchor for {residue.get_full_id()} (no atoms?)."); return None


def get_ring_nitrogen_coords(base_residue):
    _, ring_atom_objects = get_bonds(base_residue)
    return [atom.coord for atom in ring_atom_objects if atom.element.upper() == 'N']

# --- Plotting Functions ---
def plot_projected_base(base_residue, ax, view_rotation_matrix, pre_rotation_offset_vector, structure_resolution):
    atoms_in_residue = list(base_residue.get_atoms())
    if not atoms_in_residue: return [], False
    bonds_atom_obj_pairs, ring_atoms_list = get_bonds(base_residue)
    if not ring_atoms_list: return [], True

    min_res_thin, max_res_thick, min_lw, max_lw = 3.5, 1.0, 0.6, 1.8
    bond_thickness = min_lw
    if structure_resolution is not None:
        if structure_resolution <= max_res_thick: bond_thickness = max_lw
        elif structure_resolution >= min_res_thin: bond_thickness = min_lw
        else:
            factor = (structure_resolution - min_res_thin) / (max_res_thick - min_res_thin)
            bond_thickness = min_lw + factor * (max_lw - min_lw)
            bond_thickness = np.clip(bond_thickness, min_lw, max_lw)

    resname = base_residue.resname.strip().upper()
    central_atom_obj = None
    if resname in ['A', 'G', 'I']: central_atom_obj = next((a for a in atoms_in_residue if a.name == 'N9'), None)
    elif resname in ['C', 'U', 'T']: central_atom_obj = next((a for a in atoms_in_residue if a.name == 'N1'), None)
    if not central_atom_obj:
        prefs = ['N9','N1','N7','N3','C8','C6','C5','C4','C2']
        for p_name in prefs:
            central_atom_obj = next((a for a in ring_atoms_list if a.name == p_name), None)
            if central_atom_obj: break
        if not central_atom_obj and ring_atoms_list: central_atom_obj = ring_atoms_list[0]

    def_s, N_s, cen_s = 15, 50, 40
    proj_coords_lims, ring_ids_set = [], {id(a) for a in ring_atoms_list}
    cen_atom_id = id(central_atom_obj) if central_atom_obj else None

    for atom in atoms_in_residue:
        if id(atom) not in ring_ids_set: continue
        transformed_3d_coord = atom.coord + pre_rotation_offset_vector
        rotated_3d_coord = np.dot(transformed_3d_coord, view_rotation_matrix.T)
        projected_2d_coord = rotated_3d_coord[:2]
        proj_coords_lims.append(projected_2d_coord)
        element_symbol = atom.element.upper()
        atom_color_rgb = JMOL_COLORS_RGB.get(element_symbol, DEFAULT_ATOM_COLOR_RGB)
        marker_shape, atom_plot_size, plot_z_order = 'o', def_s, 5
        if element_symbol == 'N': atom_plot_size = N_s
        if cen_atom_id and id(atom) == cen_atom_id:
            atom_plot_size, marker_shape, plot_z_order = cen_s, '*', 10
        ax.scatter(projected_2d_coord[0], projected_2d_coord[1], 
                   s=atom_plot_size, c=[atom_color_rgb], marker=marker_shape, 
                   zorder=plot_z_order, edgecolors='black', linewidths=0.3)

    for atom1_obj, atom2_obj in bonds_atom_obj_pairs:
        start_orig_3d, end_orig_3d = atom1_obj.coord, atom2_obj.coord
        color1 = JMOL_COLORS_RGB.get(atom1_obj.element.upper(), DEFAULT_ATOM_COLOR_RGB)
        color2 = JMOL_COLORS_RGB.get(atom2_obj.element.upper(), DEFAULT_ATOM_COLOR_RGB)
        start_transformed = start_orig_3d + pre_rotation_offset_vector
        end_transformed   = end_orig_3d   + pre_rotation_offset_vector
        start_rotated = np.dot(start_transformed, view_rotation_matrix.T)
        end_rotated   = np.dot(end_transformed,   view_rotation_matrix.T)
        start_2d, end_2d = start_rotated[:2], end_rotated[:2]
        mid_2d = (start_2d + end_2d) / 2.0
        ax.plot([start_2d[0], mid_2d[0]], [start_2d[1], mid_2d[1]], 
                color=color1, linewidth=bond_thickness, zorder=1, solid_capstyle='round')
        ax.plot([mid_2d[0], end_2d[0]], [mid_2d[1], end_2d[1]], 
                color=color2, linewidth=bond_thickness, zorder=1, solid_capstyle='round')
    return proj_coords_lims, True

def generate_base_image(structure, base1_id, base2_id, output_path):
    base1_res, base2_res = find_residue(structure, base1_id), find_residue(structure, base2_id)
    if not base1_res or not base2_res or base1_res.get_full_id() == base2_res.get_full_id():
        logging.warning(f"Invalid base pair for image: {base1_id}-{base2_id}"); return False
    anchor1 = get_anchor_atom_coord(base1_res)
    anchor2 = get_anchor_atom_coord(base2_res)
    if anchor1 is None or anchor2 is None:
        logging.error(f"Could not get anchors for {base1_id}-{base2_id}. Skipping."); return False
    struct_resolution = getattr(structure, 'resolution', None)

    best_azimuth_for_middle_view = 0; max_projected_N_area = -1.0
    elevation_top_down = 90.0
    offset_b1_for_opt, offset_b2_for_opt = -anchor1, -anchor2
    ring_N_coords_b1 = get_ring_nitrogen_coords(base1_res)
    ring_N_coords_b2 = get_ring_nitrogen_coords(base2_res)
    if ring_N_coords_b1 or ring_N_coords_b2:
        for test_az_deg in range(0, 360, 15):
            R_z_test = rotation_matrix_from_axis_angle(np.array([0,0,1]), np.deg2rad(test_az_deg))
            R_x_elev_fixed = rotation_matrix_from_axis_angle(np.array([1,0,0]), np.deg2rad(elevation_top_down))
            current_test_rot_matrix = R_x_elev_fixed @ R_z_test
            projected_N_points_for_hull = []
            for c3d in ring_N_coords_b1:
                projected_N_points_for_hull.append(np.dot(c3d + offset_b1_for_opt, current_test_rot_matrix.T)[:2])
            for c3d in ring_N_coords_b2:
                projected_N_points_for_hull.append(np.dot(c3d + offset_b2_for_opt, current_test_rot_matrix.T)[:2])
            if len(projected_N_points_for_hull) >= 3:
                try:
                    points_array = np.array(projected_N_points_for_hull)
                    if len(np.unique(points_array, axis=0)) >= 3:
                        hull = ConvexHull(points_array, qhull_options='QJ Pp')
                        area = hull.volume
                        if area > max_projected_N_area:
                            max_projected_N_area = area; best_azimuth_for_middle_view = test_az_deg
                except Exception as e_hull:
                    logging.debug(f"ConvexHull err optim {base1_id}-{base2_id}, az {test_az_deg}: {e_hull}")
    
    views_params_final_ordered = [
       (-90, 20), (best_azimuth_for_middle_view, 90.0), (90, 20)
    ]
    view_rotation_matrices_final = []
    for azim_final, elev_final in views_params_final_ordered:
        R_z = rotation_matrix_from_axis_angle(np.array([0,0,1]), np.deg2rad(azim_final))
        R_x = rotation_matrix_from_axis_angle(np.array([1,0,0]), np.deg2rad(elev_final))
        view_rotation_matrices_final.append(R_x @ R_z)

    Y_STACK_OFFSET_VAL = 3.5 
    HALF_Y_STACK_OFFSET_VEC = np.array([0, Y_STACK_OFFSET_VAL / 2.0, 0])
    offsets_b1_side_stack = HALF_Y_STACK_OFFSET_VEC - anchor1
    offsets_b2_side_stack = -HALF_Y_STACK_OFFSET_VEC - anchor2
    base_offsets_per_view_final = [
        (offsets_b1_side_stack, offsets_b2_side_stack),
        (offset_b1_for_opt,     offset_b2_for_opt),
        (offsets_b1_side_stack, offsets_b2_side_stack)
    ]

    target_subplot_px, num_subplots_val, plot_dpi_val = 128, 3, 25
    fig_h_inches = target_subplot_px / plot_dpi_val
    fig_w_inches = (target_subplot_px * num_subplots_val) / plot_dpi_val
    fig, axes_list = plt.subplots(1, num_subplots_val, figsize=(fig_w_inches, fig_h_inches))
    if num_subplots_val == 1: axes_list = [axes_list]

    any_atom_plotted_in_any_view = False
    for i, current_ax in enumerate(axes_list):
        current_rot_m = view_rotation_matrices_final[i]
        current_offset_b1, current_offset_b2 = base_offsets_per_view_final[i]
        coords_collected_for_this_view = []
        pcoords1, success_b1 = plot_projected_base(base1_res, current_ax, current_rot_m, current_offset_b1, struct_resolution)
        if success_b1 and pcoords1: coords_collected_for_this_view.extend(pcoords1); any_atom_plotted_in_any_view = True
        pcoords2, success_b2 = plot_projected_base(base2_res, current_ax, current_rot_m, current_offset_b2, struct_resolution)
        if success_b2 and pcoords2: coords_collected_for_this_view.extend(pcoords2); any_atom_plotted_in_any_view = True
        if not coords_collected_for_this_view:
            current_ax.axis('off'); continue
        coords_arr_curr_view = np.array(coords_collected_for_this_view)
        min_x,max_x = coords_arr_curr_view[:,0].min(), coords_arr_curr_view[:,0].max()
        min_y,max_y = coords_arr_curr_view[:,1].min(), coords_arr_curr_view[:,1].max()
        center_x,center_y = (min_x+max_x)/2, (min_y+max_y)/2
        half_span_plot = max((max_x-min_x)/2, (max_y-min_y)/2, 0.1) + 1.5
        current_ax.set_xlim(center_x-half_span_plot, center_x+half_span_plot)
        current_ax.set_ylim(center_y-half_span_plot, center_y+half_span_plot)
        current_ax.set_aspect('equal', adjustable='box'); current_ax.axis('off')

    if not any_atom_plotted_in_any_view:
        logging.warning(f"No atoms plotted for {base1_id}-{base2_id} in any view. Skip save."); plt.close(fig); return False
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    try: plt.savefig(output_path, dpi=plot_dpi_val, bbox_inches='tight', pad_inches=0)
    except Exception as e_save: logging.error(f"Save error {output_path}: {e_save}"); plt.close(fig); return False
    finally: plt.close(fig)
    return True

# --- Residue Finding ---
def find_residue(structure, residue_id_info):
    target_chain_id, target_res_num, target_res_icode = None, None, ' '
    if isinstance(residue_id_info, tuple) and len(residue_id_info) >= 2:
        target_chain_id, target_res_num = str(residue_id_info[0]), int(residue_id_info[1])
        if len(residue_id_info) > 2 and residue_id_info[2]: target_res_icode = str(residue_id_info[2]).strip() or ' '
    elif isinstance(residue_id_info, str):
        match = re.match(r"([A-Za-z0-9]+)(\d+)([A-Za-z]?)", residue_id_info)
        if match:
            target_chain_id, target_res_num = match.group(1).upper(), int(match.group(2))
            icode_match = match.group(3).upper(); target_res_icode = icode_match if icode_match else ' '
        else: logging.error(f"Could not parse ID string: {residue_id_info}"); return None
    else: logging.error(f"Unsupported ID format: {residue_id_info}"); return None
    if not target_chain_id or target_res_num is None: logging.error(f"Invalid ID components: {residue_id_info}"); return None
    try:
        if 0 not in structure: logging.error(f"No Model 0 in {structure.id}."); return None
        model = structure[0]; chain_obj = None
        if target_chain_id in model: chain_obj = model[target_chain_id]
        else: chain_obj = next((mc for mc in model if mc.id.upper() == target_chain_id.upper()), None)
        if not chain_obj: logging.warning(f"Chain '{target_chain_id}' not found in {structure.id}"); return None
        std_id_tuple = (' ', target_res_num, target_res_icode)
        if std_id_tuple in chain_obj: return chain_obj[std_id_tuple]
        common_het_flags = ['H_A','H_G','H_C','H_U','H_T','H_I','H_DA','H_DG','H_DC','H_DT','H_DI']
        for hf_val in common_het_flags:
            het_id_tuple = (hf_val, target_res_num, target_res_icode)
            if het_id_tuple in chain_obj: return chain_obj[het_id_tuple]
        for res_obj_iter in chain_obj:
            res_iter_id = res_obj_iter.id
            if res_iter_id[1] == target_res_num and \
               (res_iter_id[2].strip() == target_res_icode.strip() or \
                (not res_iter_id[2].strip() and not target_res_icode.strip())):
                return res_obj_iter
        logging.warning(f"Residue ({target_res_num}, icode='{target_res_icode}') not found in Chain '{chain_obj.id}' of {structure.id}."); return None
    except Exception as e_find: logging.error(f"Find residue error for {residue_id_info} in {structure.id}: {e_find}", exc_info=False); return None

# --- CSV and File Processing Logic ---
def get_identifiers_from_row(row_dict, is_new_format_csv):
    mol_name = row_dict.get('mol_name'); base1_id_val, base2_id_val, base1_str_val, base2_str_val = None,None,"NA","NA"
    if not mol_name: return None,None,None,None,None
    try:
        if is_new_format_csv:
            c1,p1s,c2,p2s = row_dict.get('chain1'),row_dict.get('position1'),row_dict.get('chain2'),row_dict.get('position2')
            if not all([c1,p1s,c2,p2s]): return mol_name,None,None,None,None
            p1_int,p2_int = int(float(p1s)), int(float(p2s))
            base1_id_val,base2_id_val = (c1,p1_int,' '),(c2,p2_int,' '); base1_str_val,base2_str_val = f"{c1}{p1_int}",f"{c2}{p2_int}"
        else:
            b1_raw,b2_raw = row_dict.get('b1'),row_dict.get('b2')
            if not b1_raw or not b2_raw: return mol_name,None,None,None,None
            base1_id_val,base2_id_val = b1_raw.replace("'","").strip(),b2_raw.replace("'","").strip(); base1_str_val,base2_str_val = base1_id_val,base2_id_val
        return mol_name,base1_id_val,base2_id_val,base1_str_val,base2_str_val
    except (ValueError, TypeError) as e_id: logging.warning(f"ID conversion error for {mol_name}: {e_id}"); return mol_name,None,None,None,None

def process_row(row_data_dict, pdb_files_dir, output_images_dir, is_new_format_flag):
    molname,b1_identifier,b2_identifier,b1_string,b2_string = get_identifiers_from_row(row_data_dict,is_new_format_flag)
    if b1_identifier is None or b2_identifier is None: return False
    try:
        cif_file_path = os.path.join(pdb_files_dir,f"{molname}.cif")
        pdb_file_path = os.path.join(pdb_files_dir,f"{molname}.pdb")
        file_to_load_path = cif_file_path if os.path.exists(cif_file_path) else (pdb_file_path if os.path.exists(pdb_file_path) else None)
        if not file_to_load_path: logging.debug(f"No structure file for {molname}."); return False
        struct_obj = load_structure_cached(file_to_load_path)
        if not struct_obj: return False
        output_image_path = os.path.join(output_images_dir, f"{molname}_{b1_string}_{b2_string}.png")
        return generate_base_image(struct_obj,b1_identifier,b2_identifier,output_image_path)
    except Exception as e_proc_row: logging.exception(f"Unhandled error in process_row for {molname} ({b1_string}-{b2_string}): {e_proc_row}"); return False

def get_existing_image_keys(image_output_dir):
    existing_keys_set=set();
    if not os.path.isdir(image_output_dir): logging.warning(f"Scan directory not found: {image_output_dir}"); return existing_keys_set
    logging.info(f"Scanning {image_output_dir} for existing images...")
    for filename_str in os.listdir(image_output_dir):
        if filename_str.lower().endswith(".png"):
            filename_parts = filename_str[:-4].split('_');
            if len(filename_parts)>=3:
                mol_part = "_".join(filename_parts[:-2]); b1_part = filename_parts[-2]; b2_part = filename_parts[-1]
                if mol_part and b1_part and b2_part: existing_keys_set.add((mol_part,b1_part,b2_part))
    logging.info(f"Found {len(existing_keys_set)} existing image keys."); return existing_keys_set

def process_csv_parallel(csv_file_path_str, pdb_dir_str, out_dir_str, num_workers_val, chunk_size_val=500):
    
    os.makedirs(out_dir_str, exist_ok=True)
    pdb_abs_dir=os.path.abspath(pdb_dir_str)
    existing_keys_collection=get_existing_image_keys(out_dir_str)
    
    is_new_csv = False
    rows_read_count = 0
    skipped_count = 0
    submitted_count = 0
    success_count = 0
    failure_count = 0 
    
    time_start_total=time.time()
    
    try:
        with open(csv_file_path_str,'r',encoding='utf-8') as csv_input_file:
            csv_reader_obj=csv.DictReader(csv_input_file)
            
            if not csv_reader_obj.fieldnames: 
                logging.error(f"CSV {csv_file_path_str} empty/no header")
                return 0,0,0
            
            if all(k in csv_reader_obj.fieldnames for k in ['mol_name','chain1','position1','chain2','position2']): 
                is_new_csv=True
            elif not all(k in csv_reader_obj.fieldnames for k in ['mol_name','b1','b2']):
                csv_reader_obj.fieldnames = [field.strip() for field in csv_reader_obj.fieldnames]
                if all(k in csv_reader_obj.fieldnames for k in ['mol_name', 'b1', 'b2']):
                    pass
                elif all(k in csv_reader_obj.fieldnames for k in ['mol_name','chain1','position1','chain2','position2']):
                    is_new_csv=True
                else:
                    logging.error(f"CSV {csv_file_path_str} bad header")
                    return 0,0,0
            
            logging.info(f"CSV: {csv_file_path_str} Format: {'new' if is_new_csv else 'old'}")
            
            process_row_func=partial(process_row,
                                     pdb_files_dir=pdb_abs_dir,
                                     output_images_dir=out_dir_str,
                                     is_new_format_flag=is_new_csv)
            chunk_num_counter=0
            while True:
                chunk_time_start_sec=time.time()
                rows_to_process_list = []
                rows_read_this_chunk = 0
                
                try:
                    for _ in range(chunk_size_val):
                        row_item=next(csv_reader_obj) 
                        rows_read_count+=1
                        rows_read_this_chunk+=1
                        
                        mol_n,_,_,b1_s,b2_s=get_identifiers_from_row(row_item,is_new_csv)
                        if mol_n and b1_s and b2_s:
                            if (mol_n,b1_s,b2_s) in existing_keys_collection:
                                skipped_count+=1
                                continue
                            rows_to_process_list.append(row_item)
                            
                except StopIteration: 
                    pass
                if not rows_to_process_list and rows_read_this_chunk==0: 
                    break
                if not rows_to_process_list:
                    chunk_num_counter+=1;
                    logging.info(f"Chunk {chunk_num_counter}: All {rows_read_this_chunk} rows skipped/invalid.")
                    continue
                
                chunk_num_counter+=1
                submitted_count+=len(rows_to_process_list)
                logging.info(f"Chunk {chunk_num_counter} ({len(rows_to_process_list)} rows). TotalRead:{rows_read_count} Skipped:{skipped_count}")
                chunk_success_val,chunk_failure_val=0,0
                
                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers_val) as executor_pool:
                        futures_list_obj=[executor_pool.submit(process_row_func,r_dict) for r_dict in rows_to_process_list]
                        for future_obj_completed in tqdm(concurrent.futures.as_completed(futures_list_obj),total=len(futures_list_obj),desc=f"C{chunk_num_counter}",unit="img",leave=False):
                            try:
                                if future_obj_completed.result(): chunk_success_val+=1
                                else: chunk_failure_val+=1
                            except Exception as e_task: 
                                logging.error(f"Task exception in C{chunk_num_counter}: {e_task}"); chunk_failure_val+=1
                except Exception as e_pool: 
                    logging.error(f"Pool error in C{chunk_num_counter}: {e_pool}")
                    chunk_failure_val=len(rows_to_process_list)
                    
                success_count+=chunk_success_val
                failure_count+=chunk_failure_val
                gc.collect()
                current_chunk_rate = len(rows_to_process_list)/(time.time()-chunk_time_start_sec) if (time.time()-chunk_time_start_sec)>0 else 0
                logging.info(f"C{chunk_num_counter} end. S:{chunk_success_val} F:{chunk_failure_val}. Rate: {current_chunk_rate:.1f} img/s")
                
    except FileNotFoundError: 
        logging.error(f"CSV not found: {csv_file_path_str}")
        return 0,0,0
    except Exception as e_csv_global: 
        logging.error(f"Global CSV processing error {csv_file_path_str}: {e_csv_global}",exc_info=True)
    total_processing_duration = time.time()-time_start_total
    logging.info(f"END {csv_file_path_str} in {total_processing_duration:.1f}s. Read:{rows_read_count} Submit:{submitted_count} Succ:{success_count} Fail:{failure_count} Skip:{skipped_count}")
    return success_count,failure_count,skipped_count

def process_multiple_csvs(list_of_csv_paths, pdb_directory_main, base_output_dir_main, num_processes_val):
    
    total_s_overall = 0
    total_f_overall = 0
    total_sk_overall = 0
    
    logging.info(f"Multi-CSV processing: {len(list_of_csv_paths)} files. PDBs: {pdb_directory_main}, OutBase: {base_output_dir_main}, Procs: {num_processes_val}")
    
    for csv_path_item_str in list_of_csv_paths:
        if not os.path.exists(csv_path_item_str): 
            logging.warning(f"Skipping missing CSV: {csv_path_item_str}") 
            continue
        
        output_subdir_path = os.path.join(base_output_dir_main, os.path.splitext(os.path.basename(csv_path_item_str))[0])
        logging.info(f"\n-> Processing CSV: {csv_path_item_str} -> Outputting to: {output_subdir_path}")
        s_val,f_val,sk_val = process_csv_parallel(csv_path_item_str,pdb_directory_main,output_subdir_path,num_processes_val)
        total_s_overall+=s_val
        total_f_overall+=f_val 
        total_sk_overall+=sk_val

    logging.info(f"\n=== ALL CSVs PROCESSED ===\nTotal Success: {total_s_overall}, Total Failed: {total_f_overall}, Total Skipped Existing: {total_sk_overall}\n============================")

def main():
    valid_csv_paths_list = []
    csv_files_to_process = csv_file
    
    try: 
        script_dir_path = os.path.dirname(os.path.abspath(__file__))
    except NameError: 
        script_dir_path = os.getcwd()
        logging.info("Running in interactive mode: using CWD as script_dir.")
        
    pdb_structure_files_dir = "cif"
    base_output_images_dir = os.path.join(script_dir_path, "output_images")

    if not os.path.isdir(pdb_structure_files_dir):
        logging.error(f"PDB/CIF directory NOT FOUND: {pdb_structure_files_dir}")
        return

    for f_csv_filename_str in csv_files_to_process:
        
        path_in_script_dir_str = os.path.join(script_dir_path, f_csv_filename_str)
        
        if os.path.exists(f_csv_filename_str) and os.path.isabs(f_csv_filename_str): 
            valid_csv_paths_list.append(f_csv_filename_str)
        elif os.path.exists(path_in_script_dir_str): 
            valid_csv_paths_list.append(path_in_script_dir_str)
        elif os.path.exists(os.path.join(os.getcwd(), f_csv_filename_str)):
            valid_csv_paths_list.append(os.path.join(os.getcwd(), f_csv_filename_str))
            logging.warning(f"CSV '{f_csv_filename_str}' in CWD.")
        else: 
            logging.error(f"CSV file '{f_csv_filename_str}' not found. Skipping.")
            
    if not valid_csv_paths_list: 
        logging.error("No valid CSV files found. Exiting.")
        return
    
    num_workers = 18
    process_multiple_csvs(valid_csv_paths_list, pdb_structure_files_dir, base_output_images_dir, num_workers)

if __name__ == "__main__":
    main()