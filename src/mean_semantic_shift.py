import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

# Hugging Face Hub Mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def semantic_shift(embs: np.ndarray) -> float:
    """
    Compute semantic shift for a window of embeddings.

      - dist_matrix = cosine_distances(embs)
      - mpd = mean of upper-triangle distances (excluding diagonal)
      - cumulate_shift = sum of adjacent distances
      - semantic_shift_value = cumulate_shift * mpd

    Args:
        embs: np.ndarray of shape (k, d)

    Returns:
        float: semantic shift value
    """
    if len(embs) <= 1:
        return 0.0

    dist_matrix = cosine_distances(embs)  # k x k

    k = len(embs)
    upper_tri_sub_matrix = dist_matrix[np.triu_indices(k, 1)]
    mpd = float(np.mean(upper_tri_sub_matrix))  # mean pairwise distance

    adjacent_values = dist_matrix[np.arange(k - 1), np.arange(1, k)]  # distances of adjacent vectors
    cumulate_shift = float(np.sum(adjacent_values))  # cumulated distance of adjacent vectors

    semantic_shift_value = cumulate_shift * mpd
    return semantic_shift_value


def mean_semantic_shift_across_windows(embeddings: np.ndarray):
    """
    Args:
        embeddings: np.ndarray of shape (n, d)

    Returns:
        A list of 9 values: the mean semantic shift over consecutive windows that include
        the current sentence plus the next w sentences, for w in [1..9].
    """
    embs = embeddings
    n = len(embs)

    # Nine lists for window sizes 1..9
    mpd_lists = [[] for _ in range(9)]

    # Iterate over each starting sentence i
    for i in range(n):
        for w in range(1, 10):  # window sizes 1..9
            end = i + w + 1     # include current + next w sentences
            if end > n:
                break
            window_embs = embs[i:end]
            mpd = semantic_shift(window_embs)
            mpd_lists[w - 1].append(mpd)

    # Compute final average semantic shift for each window size
    avg_mpds = [np.mean(lst) if len(lst) > 0 else 0.0 for lst in mpd_lists]
    return avg_mpds


def mean_semantic_shift_across_random_windows(embeddings: np.ndarray, seed: int = 42):
    """
    For each embedding i:
      - randomly choose 9 distinct embeddings (indices)
      - build a candidate window that includes i + those 9
      - for each window size w in [1..9], compute semantic_shift on the first (w+1) embeddings
    Finally, return the mean semantic shift for each window size.

    Args:
        embeddings: np.ndarray of shape (n, d)
        seed: random seed

    Returns:
        A list of 9 values: mean semantic shift for random windows of size (w+1), w in [1..9].
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(embeddings))

    n = len(embeddings)

    # Nine lists for window sizes 1..9
    mpd_lists = [[] for _ in range(9)]

    # For each sentence i
    for i in range(n):
        chosen = rng.choice(indices, size=9, replace=False)
        candidate_embs = embeddings[[i] + chosen.tolist()]

        for w in range(1, 10):  # window sizes 1..9
            end = w + 1         # include current + next w sentences (within candidate_embs)
            window_embs = candidate_embs[0:end]
            mpd = semantic_shift(window_embs)
            mpd_lists[w - 1].append(mpd)

    # Compute final mean semantic shift for each window size
    avg_mpds = [np.mean(lst) if len(lst) > 0 else 0.0 for lst in mpd_lists]
    return avg_mpds


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader(clean=True, split_sentence=True)

    # sentences = loader.load_txt_and_split_sentences("../data/Alice.txt")
    # sentences = loader.load_txt_and_split_sentences("../data/Pride.txt")
    # sentences = loader.load_arxiv_streaming(max_papers=5000)
    # sentences = loader.load_essays()
    sentences = loader.load_wikipedia(max_articles=5000)

    # model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    # model = SentenceTransformer("intfloat/e5-large-v2")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    embeddings = model.encode(sentences, normalize_embeddings=True)
    avg_mpds = mean_semantic_shift_across_windows(embeddings)
    avg_mpds_rand = mean_semantic_shift_across_random_windows(embeddings)

    save_dict = {
        "mean_semantic_shift_consecutive": avg_mpds,
        "mean_semantic_shift_random": avg_mpds_rand,
    }

    json_path = "./result/semantic_shift.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)

