import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load PyG dataset
with open("tubitak_graphs/pyg_dataset_with_stds_min_max.pkl", 'rb') as f:
# with open("./mind_adni1_bl/MIND_filtered_vertices/node_feature_graphs_with_min_max/pyg_dataset_with_stds_min_max.pkl", 'rb') as f:
    data_list, _ = pickle.load(f)

print(f"Loaded {len(data_list)} graphs.")

# Convert graphs to flat node feature vectors
expected_nodes = 68
X = []
y = []

for data in data_list:
    if data.x.size(0) != expected_nodes:
        continue
    X.append(data.x.view(-1).numpy())
    y.append(data.y.item())

X = np.array(X)
y = np.array(y)

print(f"Unique labels: {np.unique(y)}")
print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear'),
    "Random Forest": RandomForestClassifier(n_estimators=500, class_weight='balanced', max_depth=10, n_jobs=-1),
    "SVM (RBF Kernel)": SVC(kernel='rbf', C=0.5, gamma='scale',class_weight='balanced'),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10),
}

# 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=55)

print("\n=== Cross-validation Classification Reports ===")
for name, clf in models.items():
    print(f"\n{name}")
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    report = classification_report(y, y_pred, digits=4)
    print(report)
