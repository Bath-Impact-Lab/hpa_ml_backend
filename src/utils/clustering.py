import math

import numpy as np
import os

from sklearn.decomposition import PCA
from torch.nn.functional import threshold

os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def perform_clustering_and_build_graph(image_features, num_clusters=10, top_n=2, image_url='', image_positions=[],
                                       protein='', tissue_type='', threshold=0):

    pca = PCA(n_components=2, random_state=42)
    try:
        reduced_features = pca.fit_transform(image_features)
        pca_success = True
    except ValueError as e:
        # In case n_components > number of samples
        pca_success = False
        reduced_features = None

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_features)

    # Compute similarity matrix
    similarity = cosine_similarity(image_features)

    num_images = image_features.shape[0]
    print(image_features.shape, reduced_features.shape)
    cluster_spacing = 50

    # Create nodes list
    nodes = [{'id': i, 'cluster': int(labels[i]), 'x': int(labels[i]) % 3 * cluster_spacing,
              'y': math.floor(int(labels[i]) / 3) * cluster_spacing, 'image_url': image_url,
              'protein': protein, 'tissue_type': tissue_type, 'image_x': image_positions[i][0],
              'image_y': image_positions[i][1], 'vector': image_features[i].tolist(), 'reduced_vector': reduced_features[i].tolist()} for i in range(num_images)]

    # Create edges list
    edges = []
    for i in range(num_images):
        # Get indices of top_n similar images excluding itself
        similar_indices = np.argsort(-similarity[i])[:top_n + 1]
        similar_indices = similar_indices[similar_indices != i]
        for j in similar_indices[:top_n]:
            weight = float(similarity[i][j])
            if weight > threshold:
                edge = {
                    'source': i,
                    'target': int(j),
                    'weight': weight
                }
                edges.append(edge)

    # Combine nodes and edges into a graph dictionary
    graph_dict = {
        'nodes': nodes,
        'edges': edges
    }

    return graph_dict
