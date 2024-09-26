import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity



def perform_clustering_and_build_graph(image_features, num_clusters=10, top_n=5):
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(image_features)

    # Compute similarity matrix
    similarity = cosine_similarity(image_features)

    num_images = image_features.shape[0]

    # Create nodes list
    nodes = [{'id': i, 'cluster': int(labels[i])} for i in range(num_images)]

    # Create edges list
    edges = []
    for i in range(num_images):
        # Get indices of top_n similar images excluding itself
        similar_indices = np.argsort(-similarity[i])[:top_n + 1]
        similar_indices = similar_indices[similar_indices != i]
        for j in similar_indices[:top_n]:
            if labels[i] == labels[j]:
                edge = {
                    'source': i,
                    'target': int(j),
                    'weight': float(similarity[i][j])
                }
                edges.append(edge)

    # Combine nodes and edges into a graph dictionary
    graph_dict = {
        'nodes': nodes,
        'edges': edges
    }

    return graph_dict

