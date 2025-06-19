#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""import numpy as np

def count_nan_values_in_dict(d):
    nan_count = 0
    for value in d.values():
        if isinstance(value, float) and "-":
            nan_count += 1
    return nan_count

# Example dictionary
#my_dict = {'a': 1, 'b': float('nan'), 'c': 3, 'd': float('nan')}

nan_count = count_nan_values_in_dict(pdb_res_dict)
print("Number of NaN values in the dictionary:", nan_count)
"""


# In[1]:


import pandas as pd
import numpy as np

def extract_pdb(file_path):
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, delimiter="\t")
    #print(df.columns)


    # Convert "Resolution" column to numeric
    #df["Resolution"] = pd.to_numeric(df["Resolution"], errors='coerce')
    for i in range(len(df['Resolution'])):
        if ',' in df['Resolution'].iloc[i]:
            df.loc[i, 'Resolution'] = df['Resolution'].iloc[i].split(',')[0]
    df["Resolution"] = pd.to_numeric(df["Resolution"],errors='coerce')
    # Filter the DataFrame based on the conditions
    #x_ray = df[df["Experimental_Method"] == "X-RAY_DIFFRACTION" ] 
    #filtered_df_hres = df[(df["Resolution"] <= 3.0) & (df["Experimental_Method"] == "X-RAY_DIFFRACTION") ]#& (df['Chain_Length'] >= 500)]
    #filtered_df_rem = df[(df["Resolution"] > 5.0) & (df["Experimental_Method"] == "X-RAY_DIFFRACTION") ]
    #pdb_ids =  filtered_df_rem["PDB_ID"].tolist()
    pdb_ids = df["PDB_ID"].tolist()
    #resolution_values =  filtered_df_rem["Resolution"].tolist()
    resolution_values =  df["Resolution"].tolist()
    #print(resolution_values)
    #df_1 = df[["PDB_ID","Resolution"]]
    #print(df[:14466])

    # Create a dictionary using a dictionary comprehension
    pdb_resolution_dict = {pdb_id: resolution for pdb_id, resolution in zip(pdb_ids, resolution_values)}

    # If you want to remove duplicate PDB IDs and only keep unique ones with their respective resolutions:
    # pdb_resolution_dict = {pdb_id: resolution for pdb_id, resolution in zip(set(pdb_ids), resolution_values)}

    #print(len(pdb_resolution_dict) )
    
    # Extract the PDB IDs from the filtered DataFrame
    #pdb_ids_hres = filtered_df_hres["PDB_ID"].tolist()
    #pdb_ids_rem = filtered_df_rem["PDB_ID"].tolist() 
    #pdb_xray = x_ray["PDB_ID"].tolist()
    #print(len(list(set(pdb_xray))))
    # Return the extracted PDB IDs
    return list(set(pdb_ids)) , pdb_resolution_dict
"""

file_path = '/home/s081p868/RNA_annotations/bp_annotations/RNA_chain_list_final'
extracted_pdb_ids , pdb_res_dict= extract_pdb(file_path)


#print(len(extracted_pdb_ids))
#lst = [x.lower() for x in extracted_pdb_ids]
#df = pd.DataFrame(lst)
#df.to_csv("list_pdbIds_from_RNA_chain_list",index=None,header=False)


with open("/home/s081p868/RNA_annotations/bp_annotations/Data_coords_all/CG/list", "r") as file:
    file_data = [line.strip().upper() for line in file]
#file_data = file_data.upper()
file_set = set(file_data)
#print(file)
difference = set(extracted_pdb_ids) - file_set

# Print the difference
print("Difference:", len(difference))
# Print the extracted PDB IDs
#print(len(extracted_pdb_ids))
"""

