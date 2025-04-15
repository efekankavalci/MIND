import os
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

import networkx as nx

def remove_low_weight_edges(graph, threshold, weight_attr='weight'):
    graph_copy = graph.copy()
    edges_to_remove = [(u, v) for u, v, d in graph_copy.edges(data=True)
                       if d.get(weight_attr, 0) < threshold]
    graph_copy.remove_edges_from(edges_to_remove)
    return graph_copy

# --- Weisfeiler-Lehman feature extraction ---
def weisfeiler_lehman_step(graph, labels):
    new_labels = {}
    for node in graph.nodes():
        neighbors = sorted([labels[neighbor] for neighbor in graph.neighbors(node)])
        label_string = str(labels[node]) + "_" + "_".join(map(str, neighbors))
        new_labels[node] = hash(label_string)
    return new_labels

def extract_wl_features(graph, h=2):
    labels = {n: str(n) for n in graph.nodes()}
    doc = []

    for _ in range(h):
        labels = weisfeiler_lehman_step(graph, labels)
        doc.extend([str(v) for v in labels.values()])

    return doc
