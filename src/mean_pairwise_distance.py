from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from src.data_loader import DataLoader
import json
from openai import OpenAI
import math


# ------------------------------------------------------------------
def _to_serializable(obj):
    """Convert numpy/scalar containers to JSON-serializable Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


# ------------------------------------------------------------------
def _save_results(save_filename, result_dict, result_dir="./result/"):
    """Save result_dict to a JSON file under result_dir."""
    safe_res = _to_serializable(result_dict)
    json_path = f"{result_dir}{save_filename}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_res, f, indent=2, ensure_ascii=False)

    print(f"Saved: {json_path}")


def embed_sentences(sentences, model_name="text-embedding-3-large", batch_size=100):
    """
    Embed a large number of sentences in batches (e.g., 2000 sentences).
    Returns a numpy array with shape (num_sentences, embedding_dim).
    """
    client = OpenAI()
    all_embeddings = []
    num_batches = math.ceil(len(sentences) / batch_size)

    for i in range(num_batches):
        batch = sentences[i * batch_size : (i + 1) * batch_size]

        resp = client.embeddings.create(
            model=model_name,
            input=batch,
        )

        # Extract embedding vectors
        batch_emb = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_emb)

        print(f"Processed batch {i + 1}/{num_batches}")

    return np.array(all_embeddings)


def mean_pairwise_and_adjacent_distance(sentences, model_name):
    if model_name in ["text-embedding-3-large", "text-embedding-3-small"]:
        embeddings = embed_sentences(sentences, model_name=model_name)
    elif model_name in [
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "thenlper/gte-large",
        "sentence-transformers/all-mpnet-base-v2",
    ]:
        embedding_model = SentenceTransformer(model_name)
        embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
    else:
        raise ValueError(f"Input model name '{model_name}' is not valid.")

    m = len(embeddings)
    if m < 2:
        return [0.0], [0.0]

    # Compute the full pairwise distance matrix (m x m)
    dist_matrix = cosine_distances(embeddings)

    mean_pairwise_list = [0.0]
    mean_adjacent_list = [0.0]

    for size in range(2, len(sentences) + 1):
        sub_matrix = dist_matrix[0:size, 0:size]

        # === 1. Global MPD ===
        upper = sub_matrix[np.triu_indices(size, k=1)]
        mean_pairwise = upper.mean()

        # === 2. Mean distance of adjacent vectors (i, i+1), taken from the k=1 diagonal ===
        # Example: [dist(0,1), dist(1,2), dist(2,3), ...]
        adjacent_values = sub_matrix[np.arange(size - 1), np.arange(1, size)]
        mean_adjacent = adjacent_values.mean()

        mean_pairwise_list.append(mean_pairwise)
        mean_adjacent_list.append(mean_adjacent)

    res = {
        "MPD": mean_pairwise_list,  # Mean Pairwise Distance
        "Mean_Adjacent_Distance": mean_adjacent_list,
    }
    _save_results(model_name[0:8], res, result_dir="./result/")

    return mean_pairwise_list, mean_adjacent_list


if __name__ == "__main__":
    loader = DataLoader(clean=True, split_sentence=True)
    #sentences = loader.load_txt_and_split_sentences("../data/Alice.txt")
    # sentences = loader.load_txt_and_split_sentences("../data/Pride.txt")
    # sentences = loader.load_essays()
    # sentences = loader.load_wikipedia(max_articles=5000)
    sentences = loader.load_arxiv_streaming(max_papers=5000)
    # print(f"Detected {len(sentences)} sentences.")

    # "sentence-transformers/all-mpnet-base-v2",
    # "BAAI/bge-large-en-v1.5",
    # "BAAI/bge-base-en-v1.5",
    # "BAAI/bge-small-en-v1.5"
    # "intfloat/e5-large-v2", dim=1024,
    # "BAAI/bge-m3",
    # "thenlper/gte-large",  # base, small
    # "sentence-transformers/all-MiniLM-L6-v2",
    # "text-embedding-3-large", OpenAI model

    model_name = "sentence-transformers/all-mpnet-base-v2"
    mean_pairwise_and_adjacent_distance(sentences, model_name)


