import os
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# === CONFIG ===
parsed_folders = {
    # "dssr": "Parsed_dssr_stackings",  # DSSR not present
    "fr3d": "Parsed_fr3d_stackings",
    "mc": "Parsed_MC_stackings",
    "rnaview": "Parsed_RNAV_stackings",
    "clarna": "Parsed_CR_stackings"
}
base_path = "/home/s081p868/scratch/RNA_annotations/bp_annotations/annotations_sw"
output_dir = os.path.join(base_path, "Parsed_union_stackings")
os.makedirs(output_dir, exist_ok=True)

expected_cols = [
    'chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2',
    'stack_type_fr3d', 'stack_type_mc', 'stack_type_rnaview', 'stack_type_clarna',
    'clarna_score'
]

def load_all_software():
    sw_pdb_data = defaultdict(dict)
    for sw, folder in parsed_folders.items():
        full_folder = os.path.join(base_path, folder)
        csv_files = [f for f in os.listdir(full_folder) if f.endswith(".csv")]
        for f in csv_files:
            try:
                pdb_id = os.path.splitext(f)[0].lower()
                df = pd.read_csv(os.path.join(full_folder, f))
                if sw == "clarna":
                    if 'Score' in df.columns:
                        df = df.rename(columns={'label': 'stack_type_clarna', 'Score': 'clarna_score'})
                        df = df[['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'stack_type_clarna', 'clarna_score']]
                    else:
                        df = df.rename(columns={'label': 'stack_type_clarna'})
                        df['clarna_score'] = pd.NA
                        df = df[['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'stack_type_clarna', 'clarna_score']]
                elif sw == "mc":
                    df = df.rename(columns={'label': 'stack_type_mc'})
                    for col in ['nt1', 'nt2']:
                        if col not in df.columns:
                            df[col] = pd.NA
                    df = df[['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', 'stack_type_mc']]
                else:
                    df = df.rename(columns={'label': f"stack_type_{sw}"})
                    for col in ['nt1', 'nt2']:
                        if col not in df.columns:
                            df[col] = pd.NA
                    df = df[['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2', f'stack_type_{sw}']]
                df['pos1'] = df['pos1'].astype(str)
                df['pos2'] = df['pos2'].astype(str)
                sw_pdb_data[sw][pdb_id] = df
            except Exception as e:
                print(f"Error processing {f} in {sw}: {e}")
    return sw_pdb_data

def fill_mc_nts(mc_df, fr3d_df, clarna_df, rnaview_df):
    # Build lookup for each s/w in priority order
    nt_lookup = {}
    # fr3d first
    if fr3d_df is not None and not fr3d_df.empty:
        for _, row in fr3d_df.iterrows():
            key = (row['chain1'], str(row['pos1']), row['chain2'], str(row['pos2']))
            nt_lookup[key] = (row['nt1'], row['nt2'])
    # clarna next
    if clarna_df is not None and not clarna_df.empty:
        for _, row in clarna_df.iterrows():
            key = (row['chain1'], str(row['pos1']), row['chain2'], str(row['pos2']))
            if key not in nt_lookup or any(pd.isna(v) for v in nt_lookup[key]):
                nt_lookup[key] = (row['nt1'], row['nt2'])
    # rnaview last
    if rnaview_df is not None and not rnaview_df.empty:
        for _, row in rnaview_df.iterrows():
            key = (row['chain1'], str(row['pos1']), row['chain2'], str(row['pos2']))
            if key not in nt_lookup or any(pd.isna(v) for v in nt_lookup[key]):
                nt_lookup[key] = (row['nt1'], row['nt2'])
    # Fill MC-Annotate DataFrame
    for idx, row in mc_df.iterrows():
        key = (row['chain1'], str(row['pos1']), row['chain2'], str(row['pos2']))
        nt1, nt2 = nt_lookup.get(key, (pd.NA, pd.NA))
        mc_df.at[idx, 'nt1'] = nt1
        mc_df.at[idx, 'nt2'] = nt2
    return mc_df

def merge_and_write(pdb_id, sw_pdb_data):
    dfs = []
    # For MC-Annotate, fill nt1/nt2 in order: fr3d, clarna, rnaview
    mc_df = sw_pdb_data['mc'].get(pdb_id, pd.DataFrame())
    fr3d_df = sw_pdb_data['fr3d'].get(pdb_id, pd.DataFrame())
    clarna_df = sw_pdb_data['clarna'].get(pdb_id, pd.DataFrame())
    rnaview_df = sw_pdb_data['rnaview'].get(pdb_id, pd.DataFrame())
    if not mc_df.empty:
        mc_df = fill_mc_nts(mc_df, fr3d_df, clarna_df, rnaview_df)
    for sw in parsed_folders.keys():
        if sw == "mc" and not mc_df.empty:
            dfs.append(mc_df)
        elif sw != "mc" and pdb_id in sw_pdb_data[sw]:
            dfs.append(sw_pdb_data[sw][pdb_id])
    if not dfs:
        return
    merged = dfs[0]
    for df in dfs[1:]:
        merge_cols = ['chain1', 'pos1', 'chain2', 'pos2', 'nt1', 'nt2']
        for col in merge_cols:
            merged[col] = merged[col].astype(str)
            df[col] = df[col].astype(str)
        merged = pd.merge(merged, df, on=merge_cols, how='outer')
    merged = merged[[col for col in merged.columns if col not in ['label', 'Score', 'type']]]
    for col in expected_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[expected_cols]
    merged = merged.drop_duplicates()
    out_path = os.path.join(output_dir, f"{pdb_id}.csv")
    merged.to_csv(out_path, index=False)
    print(f"Written stacking union for {pdb_id}: {out_path}")

def main():
    sw_pdb_data = load_all_software()
    all_pdb_ids = set()
    for d in sw_pdb_data.values():
        all_pdb_ids.update(d.keys())
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(merge_and_write, pdb_id, sw_pdb_data) for pdb_id in sorted(all_pdb_ids)]
        for future in as_completed(futures):
            future.result()  # Will raise exceptions if any

if __name__ == '__main__':
    main()
