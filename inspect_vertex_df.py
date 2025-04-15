from MIND import compute_MIND
from get_vertex_df import get_vertex_df
import os
import pandas as pd
import networkx as nx
import nibabel.freesurfer.io as fsio
import numpy as np

# surf_dir = r'\\wsl$\Ubuntu-22.04\home\efekan\adni'
surf_dir = r'C:\Users\efeka\Desktop\TubitakData\2025.01.03_mri_patients_features\tubitak_fs'
features = ['CT', 'Vol', 'SA', 'MC', 'SD']
parcellation = 'aparc'

# Get all subject directories, excluding 'fsaverage' and any shell scripts
subject_ptids = os.listdir(surf_dir)
subject_ptids = [i for i in subject_ptids if (i != 'fsaverage' and '.sh' not in i)]

processed_subjects = 0
for id in subject_ptids:
    print(f"Processing {id}...")
    subject_dir = os.path.join(surf_dir, id)
    surf_files_dir = os.path.join(subject_dir, 'surf')
    # Skip if surf_files_dir doesn't exist or is empty
    if not os.path.isdir(surf_files_dir) or not os.listdir(surf_files_dir):
        continue
    if not os.path.isdir(os.path.join(subject_dir,'label')) or not os.listdir(os.path.join(subject_dir,'label')):
        continue
    if not os.path.isdir(os.path.join(subject_dir,'surf')) or not os.listdir(os.path.join(subject_dir,'surf')):
        continue
    vertex_data, regions, features_used  = get_vertex_df(subject_dir, features, parcellation)
    processed_subjects += 1
    # Check whether region number is 68, else warning
    if len(np.unique(regions)) != 68:
        print(f"Warning: Region number is not 68 for subject {id}.")
        break
        continue
    # Check if vertex_data contains 68 unique regions
    if len(np.unique(vertex_data['Label'].values)) != 68:
        print(f"Warning: Vertex data does not contain 68 unique regions for subject {id}.")
        continue

print(f"Processed {processed_subjects} subjects.")