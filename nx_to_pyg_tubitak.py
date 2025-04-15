import os
import pickle
import pandas as pd
import networkx as nx
import torch
import numpy as np

from torch_geometric.data import Data

def remove_low_weight_edges(G, threshold=0.05):
    """
    Example function to remove edges from G whose 'weight' is below threshold.
    Assumes undirected graph. Modify as needed.
    """
    edges_to_remove = []
    for u, v, attrs in G.edges(data=True):
        w = attrs.get('weight', 1.0)
        if w < threshold:
            edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)
    return G

def nx_to_pyg_data(G: nx.Graph, label: int):
    """
    Converts a single NetworkX graph G to a PyG Data object.
    Assumes G is undirected.

    Node features:
      - We gather 'thickness_mean', 'thickness_std',
                 'volume_sum', 'volume_std',
                 'area_mean', 'area_std',
                 'curvature_mean', 'curvature_std',
                 'sulc_mean', 'sulc_std'
        from each node attribute, or 0 if missing.

    Edge attributes:
      - We use the 'weight' attribute if it exists; otherwise 1.0.

    label: integer or tensor for classification/regression tasks.
    """
    # Map node -> index so we can build edge_index arrays
    node2idx = {node: i for i, node in enumerate(G.nodes())}

    # Build node feature matrix x
    x_list = []
    for node in G.nodes():
        node_data = G.nodes[node]
        feats = [
            node_data.get('thickness_mean', 0.0),
            node_data.get('thickness_std',  0.0),
            node_data.get('thickness_min',  0.0),
            node_data.get('thickness_max',  0.0),

            node_data.get('volume_sum',     0.0),
            node_data.get('volume_std',     0.0),
            node_data.get('volume_min',     0.0),
            node_data.get('volume_max',     0.0),

            node_data.get('area_mean',      0.0),
            node_data.get('area_std',       0.0),
            node_data.get('area_min',       0.0),
            node_data.get('area_max',       0.0),

            node_data.get('curvature_mean', 0.0),
            node_data.get('curvature_std',  0.0),
            node_data.get('curvature_min',  0.0),
            node_data.get('curvature_max',  0.0),

            node_data.get('sulc_mean',      0.0),
            node_data.get('sulc_std',       0.0),
            node_data.get('sulc_min',       0.0),
            node_data.get('sulc_max',       0.0),
        ]
        x_list.append(feats)


    x_tensor = torch.tensor(x_list, dtype=torch.float)  # shape [num_nodes, 10]

    # Build edge index and (optional) edge_attr
    edge_index = []
    edge_attr  = []
    for u, v, attrs in G.edges(data=True):
        i = node2idx[u]
        j = node2idx[v]
        edge_index.append([i, j])
        edge_index.append([j, i])

        w = attrs.get('weight', 1.0)
        edge_attr.append([w])
        edge_attr.append([w])

    if len(edge_index) == 0:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor  = torch.empty((0, 1),  dtype=torch.float)
    else:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr_tensor  = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(
        x         = x_tensor,
        edge_index= edge_index_tensor,
        edge_attr = edge_attr_tensor,
        y         = torch.tensor([label], dtype=torch.long)
    )
    return data


def main():
    """
    Example script:
      - Loads a list of patient labels from a DataFrame 'patients'
      - Reads Nx graphs from a directory
      - Thresholds edges (optional)
      - Converts each Nx graph to PyG Data
      - Collects them in data_list, label_list
      - Saves them in a single pickle file
    """

    patients = pd.read_excel(r'C:\Users\efeka\Desktop\TubitakData\2025.01.03_mri_patients_features\combined_24_with_external_features_6months.xlsx')
    patients["Diagnosis"] = patients["Diagnosis"].str.strip() # remove leading and trailing whitespaces
    patients = patients[patients["MRI Machine"] != "Error"]
    # patients = patients[patients["MRI Machine"] == "EG ARS"]
    # patients = patients[patients["MRI Machine"] == "EAH"]

    label_0_dxs = ["MCI", "MCI-AD", "MCI-VD"]
    label_1_dxs = ["AD"]

    def assign_label(diagnosis):
        if diagnosis in label_0_dxs:
            return 0
        elif diagnosis in label_1_dxs:
            return 1
        else:
            return None  # Use None for diagnoses not in either list (optional)

    patients['Label'] = patients['Diagnosis'].apply(assign_label)

    # Directory where you have Nx .pickle graphs
    graph_dir = "./tubitak_graphs/node_feature_graphs_with_min_max"

    # If you want thresholding:
    apply_thresholding = True
    threshold = 0.05

    data_list  = []
    label_list = []

    # Collect list of .pickle files
    for fname in os.listdir(graph_dir):
        if fname.endswith('.pickle'):
            subject_name = fname.replace('.pickle', '').upper()

            # Check if subject_name is in the DataFrame
            if subject_name in patients['Subject_Name'].values:
                with open(os.path.join(graph_dir, fname), 'rb') as f:
                    G = pickle.load(f)

                if apply_thresholding:
                    G = remove_low_weight_edges(G, threshold)

                # Get label from DataFrame
                label = patients[patients['Subject_Name'] == subject_name]['Label'].values[0]
                
                # Skip if label is None or NaN
                if label is None or (isinstance(label, float) and np.isnan(label)):
                    print(f"Skipping graph for {subject_name} due to NaN or undefined label.")
                    continue

                
                if G.number_of_nodes() != 68:
                    print(f"Graph for {subject_name} contains missing nodes, skipping.")
                    continue
                
                print(label, type(label))
                # Convert to PyG Data
                pyg_data = nx_to_pyg_data(G, label)
                data_list.append(pyg_data)
                label_list.append(label)

    # Save them in a single pickle (or however you'd like)
    out_path = "tubitak_graphs/pyg_dataset_with_stds_min_max.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump((data_list, label_list), f)

    print(f"Saved {len(data_list)} graphs + labels to {out_path}")

if __name__ == "__main__":
    main()
