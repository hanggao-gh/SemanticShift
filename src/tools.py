from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import re


def analyze_embeddings_geometry(embeddings):
    """
    Analyze embedding distribution geometry:
    - mean similarity, std similarity, RSV
    - mean distance,  std distance, RDV

    Args:
        embeddings: numpy array of shape (N, D)

    Returns:
        A dictionary containing:
            {
                "mean_similarity": float,
                "std_similarity": float,
                "RSV": float,
                "mean_distance": float,
                "std_distance": float,
                "RDV": float
            }
    """
    # 1. Cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # 2. Extract upper triangular (no diag, no duplicates)
    i_upper = np.triu_indices(sim_matrix.shape[0], k=1)
    sims = sim_matrix[i_upper]  # pairwise similarities

    # 3. Compute mean and std for similarity
    mean_sim = np.mean(sims)
    std_sim = np.std(sims)
    RSV = std_sim / mean_sim if mean_sim != 0 else np.nan

    # 4. Compute cosine distance = 1 - similarity
    dists = 1.0 - sims

    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    RDV = std_dist / mean_dist if mean_dist != 0 else np.nan

    # 5. Examine smoothing, row and column
    row_means = np.mean(embeddings, axis=1)
    row_stds = np.std(embeddings, axis=1)
    mean_of_row_means = np.mean(row_means)
    mean_of_row_stds = np.mean(row_stds)

    col_means = np.mean(embeddings, axis=0)
    col_stds = np.std(embeddings, axis=0)
    mean_of_col_means = np.mean(col_means)
    mean_of_col_stds = np.mean(col_stds)

    return {
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "RSV": RSV,
        "mean_distance": mean_dist,
        "std_distance": std_dist,
        "RDV": RDV,
        "mean_of_row_means": mean_of_row_means,
        "mean_of_row_stds": mean_of_row_stds,
        "mean_of_col_means": mean_of_col_means,
        "mean_of_col_stds": mean_of_col_stds
    }


