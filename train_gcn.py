import os
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import KFold
import numpy as np

# scikit-learn metrics for multi-class evaluation
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    classification_report
)

# matplotlib for plotting
import matplotlib.pyplot as plt


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_pooled = global_mean_pool(x, batch)
        x = self.lin(x_pooled)
        return x


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch.y = batch.y.long()  # Ensure integer (long) labels

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, return_preds=False):
    """
    Returns (accuracy, precision, recall) using macro averaging for multi-class problems.
    If `return_preds=True`, also returns (all_preds, all_labels).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.y = batch.y.long()

            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=-1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')

    if return_preds:
        return acc, prec, rec, all_preds, all_labels
    else:
        return acc, prec, rec


def main():
    # Load your dataset
    with open("./mind_adni1_bl/MIND_filtered_vertices/node_feature_graphs_with_min_max/pyg_dataset_with_stds_min_max.pkl", 'rb') as f:
        data_list, _ = pickle.load(f)

    print(f"Loaded {len(data_list)} PyG graphs.")

    k_folds = 5
    epochs =50
    batch_size = 32
    hidden_dim = 256
    num_classes = 3  # Adjust if you have more/less classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = data_list[0].x.shape[1]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=52)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []

    # For storing losses over epochs in each fold:
    all_fold_losses = []

    # For overall classification report across folds:
    # We will collect predictions and labels from *each* fold's test set
    all_preds_cv = []
    all_labels_cv = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
        print(f"\n=== Fold {fold+1}/{k_folds} ===")

        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = SimpleGCN(in_channels, hidden_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()

        fold_losses = []

        for epoch in range(1, epochs + 1):
            epoch_loss = train(model, train_loader, optimizer, criterion, device)
            fold_losses.append(epoch_loss)

            # acc, prec, rec = evaluate(model, test_loader, device)
            # print(
            #     f"Epoch {epoch:02d}, "
            #     f"Loss: {epoch_loss:.4f}, "
            #     f"Accuracy: {acc:.4f}, "
            #     f"Precision(macro): {prec:.4f}, "
            #     f"Recall(macro): {rec:.4f}"
            # )

        # Store per-epoch losses for this fold
        all_fold_losses.append(fold_losses)

        # Final evaluation for this fold, also retrieve predictions for overall classification report
        final_acc, final_prec, final_rec, fold_preds, fold_labels = evaluate(
            model, test_loader, device, return_preds=True
        )
        fold_accuracies.append(final_acc)
        fold_precisions.append(final_prec)
        fold_recalls.append(final_rec)

        # Collect predictions and labels in global lists
        all_preds_cv.extend(fold_preds)
        all_labels_cv.extend(fold_labels)

        print(
            f"Final Accuracy for Fold {fold+1}: {final_acc:.4f}, "
            f"Precision: {final_prec:.4f}, Recall: {final_rec:.4f}"
        )

    # Print average metrics (mean of each fold)
    avg_acc = np.mean(fold_accuracies)
    avg_prec = np.mean(fold_precisions)
    avg_rec = np.mean(fold_recalls)

    print("\n=== Cross-validation complete ===")
    print(f"Average accuracy over {k_folds} folds: {avg_acc:.4f}")
    print(f"Average precision (macro) over {k_folds} folds: {avg_prec:.4f}")
    print(f"Average recall (macro) over {k_folds} folds: {avg_rec:.4f}")

    # --- Overall classification report across all test folds ---
    print("\nOverall Classification Report (all folds combined):")
    print(classification_report(all_labels_cv, all_preds_cv))

    # --- Plot each fold's loss in its own subplot ---
    fig, axes = plt.subplots(1, k_folds, figsize=(15, 3), sharey=True)
    for fold_idx, fold_losses in enumerate(all_fold_losses):
        axes[fold_idx].plot(range(1, epochs + 1), fold_losses)
        axes[fold_idx].set_title(f'Fold {fold_idx+1} Loss')
        axes[fold_idx].set_xlabel('Epoch')
        axes[fold_idx].set_ylim(0, 10)  # limit y-axis to [0, 10]

        if fold_idx == 0:
            axes[fold_idx].set_ylabel('Training Loss')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# class SimpleGCN(nn.Module):
#     def __init__(self, in_channels, hidden_dim, out_channels, num_nodes=68):
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.conv1 = GCNConv(in_channels, hidden_dim)
#         # self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.lin = nn.Linear(hidden_dim * num_nodes, out_channels)

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         # x = self.conv2(x, edge_index)
#         # x = F.relu(x)

#         batch_size = batch.max().item() + 1
#         x_concat = torch.zeros(batch_size, self.num_nodes * x.size(1), device=x.device)

#         for i in range(batch_size):
#             xi = x[batch == i]
#             if xi.size(0) != self.num_nodes:
#                 raise ValueError(f"Graph {i} has {xi.size(0)} nodes, expected {self.num_nodes}")
#             x_concat[i] = xi.view(-1)

#         x = self.lin(x_concat)
#         return x
    