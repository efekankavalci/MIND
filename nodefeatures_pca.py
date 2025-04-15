import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
with open("tubitak_graphs/pyg_dataset_with_stds_min_max.pkl", 'rb') as f:
    data_list, label_list = pickle.load(f)

print(f"Loaded {len(data_list)} PyG graphs.")

# Check number of nodes per graph
expected_nodes = 68
concatenated_features = []
labels = []

for i, data in enumerate(data_list):
    x = data.x  # shape: [num_nodes, num_features]
    
    if x.size(0) != expected_nodes:
        print(f"Skipping graph {i} due to unexpected node count: {x.size(0)}")
        continue

    x_flat = x.view(-1).numpy()  # flatten node features into a single vector
    concatenated_features.append(x_flat)
    labels.append(data.y.item())  # assuming y is a tensor with shape [1]

X = np.vstack(concatenated_features)
y = np.array(labels)

print(f"Feature matrix shape: {X.shape}, Label vector shape: {y.shape}")

# Standardize before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Label {label}", alpha=0.7)

plt.title("PCA of Concatenated Node Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
