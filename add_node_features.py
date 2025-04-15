import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import nibabel.freesurfer.io as fsio

def aggregate_regionwise(
    annot_file,
    thickness_file,
    volume_file,
    area_file,
    curv_file,
    sulc_file,
    hemisphere_prefix="lh"
):
    """
    Reads the annotation file and the morphometry data files for one hemisphere (lh or rh).
    Aggregates vertex-level data into region-level means (or sums).
    
    Returns a dict of the form:
      {
         "<hemisphere_prefix>_<region_name>": {
            "thickness_mean": float,
            "volume_sum": float,
            "area_mean": float,
            "curvature_mean": float,
            "sulc_mean": float
         },
         ...
      }
    }
    """
    # If any file doesn’t exist, return empty dict
    for fpath in [annot_file, thickness_file, volume_file, area_file, curv_file, sulc_file]:
        if not os.path.isfile(fpath):
            print(f"WARNING: Missing file {fpath}. Skipping.")
            return {}

    # 1) Load annotation => per-vertex region labels
    labels, ctab, region_names = fsio.read_annot(annot_file)
    
    # 2) Load morphometry data => each is an array of shape (n_vertices,)
    thickness = fsio.read_morph_data(thickness_file)
    volume   = fsio.read_morph_data(volume_file)
    area     = fsio.read_morph_data(area_file)
    curv     = fsio.read_morph_data(curv_file)
    sulc     = fsio.read_morph_data(sulc_file)

    # 3) Prepare output dictionary
    region_dict = {}

    # region_names is something like [b'unknown', b'bankssts', b'caudalanteriorcingulate', ... ]
    for idx, rn in enumerate(region_names):
        region_name = rn.decode("utf-8")  # convert from bytes to str

        # Skip 'unknown' or other placeholders
        if region_name in ["unknown", "???"]:
            continue

        # Create a mask for vertices that belong to this region index.
        # Usually, read_annot sets labels == idx for that region.  If there's a mismatch,
        # you may need to map actual label values to idx, but in standard Freesurfer usage,
        # this direct equality should work.
        region_mask = (labels == idx)

        if not np.any(region_mask):
            continue

        # Example approach: thickness => mean, std, min, max; area => same; volume => sum, std, min, max; etc.
        # Adjust to your preferences.
        thickness_values = thickness[region_mask]
        volume_values    = volume[region_mask]
        area_values      = area[region_mask]
        curv_values      = curv[region_mask]
        sulc_values      = sulc[region_mask]

        thickness_mean = float(np.mean(thickness_values))
        thickness_std  = float(np.std(thickness_values))
        thickness_min  = float(np.min(thickness_values))
        thickness_max  = float(np.max(thickness_values))

        volume_sum     = float(np.sum(volume_values))
        volume_std     = float(np.std(volume_values))
        volume_min     = float(np.min(volume_values))
        volume_max     = float(np.max(volume_values))

        area_mean      = float(np.mean(area_values))
        area_std       = float(np.std(area_values))
        area_min       = float(np.min(area_values))
        area_max       = float(np.max(area_values))

        curvature_mean = float(np.mean(curv_values))
        curvature_std  = float(np.std(curv_values))
        curvature_min  = float(np.min(curv_values))
        curvature_max  = float(np.max(curv_values))

        sulc_mean      = float(np.mean(sulc_values))
        sulc_std       = float(np.std(sulc_values))
        sulc_min       = float(np.min(sulc_values))
        sulc_max       = float(np.max(sulc_values))

        # Construct the dictionary key: e.g. "lh_bankssts"
        key_name = f"{hemisphere_prefix}_{region_name}"

        region_dict[key_name] = {
            "thickness_mean": thickness_mean,
            "thickness_std": thickness_std,
            "thickness_min": thickness_min,
            "thickness_max": thickness_max,

            "volume_sum": volume_sum,
            "volume_std": volume_std,
            "volume_min": volume_min,
            "volume_max": volume_max,

            "area_mean": area_mean,
            "area_std": area_std,
            "area_min": area_min,
            "area_max": area_max,

            "curvature_mean": curvature_mean,
            "curvature_std": curvature_std,
            "curvature_min": curvature_min,
            "curvature_max": curvature_max,

            "sulc_mean": sulc_mean,
            "sulc_std": sulc_std,
            "sulc_min": sulc_min,
            "sulc_max": sulc_max,
        }


    return region_dict

def attach_fs_features_to_graph(
    G: nx.Graph,
    subject_surf_dir: str,
    aparc_name="aparc"
):
    """
    Given a NetworkX graph G with nodes named like "lh_bankssts", "rh_precentral", etc.,
    read vertex-wise files from <subject_surf_dir>:
      surf/lh.{thickness,volume,area,curv,sulc}
      surf/rh.{thickness,volume,area,curv,sulc}
      label/lh.aparc.annot, label/rh.aparc.annot
    Aggregate them to ROI-level and attach as node attributes.
    """
    surf_dir  = os.path.join(subject_surf_dir, "surf")
    label_dir = os.path.join(subject_surf_dir, "label")

    # We'll collect region features from both hemispheres in one big dict
    region_features = {}

    for hemi in ["lh", "rh"]:
        # Build file paths
        annot_file     = os.path.join(label_dir,  f"{hemi}.{aparc_name}.annot")
        thickness_file = os.path.join(surf_dir,   f"{hemi}.thickness")
        volume_file    = os.path.join(surf_dir,   f"{hemi}.volume")
        area_file      = os.path.join(surf_dir,   f"{hemi}.area")
        curv_file      = os.path.join(surf_dir,   f"{hemi}.curv")
        sulc_file      = os.path.join(surf_dir,   f"{hemi}.sulc")

        # Aggregate ROI-level metrics
        hemi_dict = aggregate_regionwise(
            annot_file,
            thickness_file,
            volume_file,
            area_file,
            curv_file,
            sulc_file,
            hemisphere_prefix=hemi
        )
        # Merge with the global dict
        region_features.update(hemi_dict)

    # Now attach these region_features to each node if it exists
    for node in G.nodes:
        if node in region_features:
            for feature_key, feature_val in region_features[node].items():
                G.nodes[node][feature_key] = feature_val
        else:
            # Possibly the node is something else that doesn't match "lh_*" or "rh_*" naming
            pass

def main():
    """
    Example script that:
      - Reads adjacency CSVs from a folder.
      - Builds a NetworkX graph from each adjacency matrix.
      - Attaches ROI-level features from surf/* and label/*.annot files.
      - Saves the graph as a .pickle file using standard Python 'pickle'.
    """

    # Directory containing your adjacency CSVs
    csv_dir = r'C:\Users\efeka\Documents\GitHub\MIND\mind_adni1_bl\MIND_filtered_vertices'
    csv_list = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    # Root directory with your FreeSurfer subjects
    # (e.g., 'C:\\Users\\...\\2025.01.03_mri_patients_features\\tubitak_fs')
    fs_subjects_dir = r'\\wsl.localhost\Ubuntu-22.04\home\efekan\adni1_bl_surfaces\ADNI'

    for csv_file in csv_list:
        csv_path = os.path.join(csv_dir, csv_file)

        # 1) Read adjacency matrix -> DataFrame
        #    If the CSV has the first column as ROI names and first row as ROI names,
        #    you can do index_col=0. Make sure your CSV is NxN with a header row.
        adj_df = pd.read_csv(csv_path)
        adj_df.index = adj_df.columns

        print(adj_df)
        # 2) Convert to a graph
        G = nx.from_pandas_adjacency(adj_df)

        # 3) Identify the subject ID from the CSV filename
        #    e.g., if csv_file is 'sub-123.csv', then subject_id='sub-123'
        subject_id = csv_file.replace('.csv', '')

        # 4) Build path to subject's FreeSurfer folder
        subject_dir = os.path.join(fs_subjects_dir, subject_id)

        if not os.path.isdir(subject_dir):
            print(f"Subject folder not found for {subject_id}, skipping feature attachment.")
        else:
            # Attach FreeSurfer ROI-based features
            attach_fs_features_to_graph(G, subject_dir, aparc_name="aparc")

        # 5) Save the graph as a .pickle
        output_pickle_path = os.path.join(
            csv_dir,
            "node_feature_graphs_with_min_max",
            csv_file.replace('.csv', '.pickle')
        )
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(G, f)

        print(f"Processed {csv_file}, saved pickle -> {output_pickle_path}")
    print("All done.")

if __name__ == "__main__":
    main()
