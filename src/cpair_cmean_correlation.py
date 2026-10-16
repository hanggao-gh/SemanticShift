import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from scipy.stats import spearmanr, kendalltau
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from data_loader import DataLoader

# Ensure punkt tokenizer is available
nltk.download("punkt", quiet=True)

EMBEDDING_BACKEND = "hf"  # "hf" or "openai"
HF_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # "intfloat/e5-large-v2" / "thenlper/gte-large" / "BAAI/bge-large-en-v1.5"
OPENAI_MODEL = "text-embedding-3-large"

M = 200          # Number of sampled groups per mode
K = 10           # Number of sentences per group
MIN_SENT_LEN = 5 # Filter out very short sentences (by character length)
SEED = 42
GAP = 3          # Minimum index gap between sentences in "medium" mode

random.seed(SEED)
np.random.seed(SEED)

if EMBEDDING_BACKEND == "hf":
    hf_model = SentenceTransformer(HF_MODEL_NAME)
elif EMBEDDING_BACKEND == "openai":
    client = OpenAI()


def encode_texts(texts):
    """
    Args:
        texts: List[str]
    Returns:
        np.ndarray of shape (n, d), L2-normalized embeddings.
    """
    if EMBEDDING_BACKEND == "hf":
        return np.array(hf_model.encode(texts, normalize_embeddings=True))

    elif EMBEDDING_BACKEND == "openai":
        resp = client.embeddings.create(
            model=OPENAI_MODEL,
            input=texts
        )
        embs = [d.embedding for d in resp.data]
        return np.array(embs)  # already normalized

    else:
        raise ValueError("Unknown embedding backend")


def compute_C_pair(embs):
    """Compute the mean pairwise cosine distance within a group of sentence embeddings."""
    k = embs.shape[0]
    sims = np.dot(embs, embs.T)  # cosine similarity
    idx = np.triu_indices(k, k=1)
    cos_sims = sims[idx]
    cos_dists = 1.0 - cos_sims
    return cos_dists.mean()


def compute_C_mean(sent_embs, text_emb):
    """Compute the mean cosine distance between the text embedding and each sentence embedding."""
    sims = np.dot(sent_embs, text_emb)
    cos_dists = 1.0 - sims
    return cos_dists.mean()


def sample_groups_local(total_sentences, M, K):
    """
    local: Sample K consecutive sentences starting from a random index.
    """
    groups = []
    max_start = total_sentences - K
    for _ in range(M):
        start = random.randint(0, max_start)
        group = list(range(start, start + K))
        groups.append(group)
    return groups


def sample_groups_medium(total_sentences, M, K, gap=3):
    """
    medium: Sample K sentences starting from a random index, taking one sentence every `gap` steps.
    """
    groups = []
    max_start = total_sentences - (K - 1) * gap - 1
    for _ in range(M):
        start = random.randint(0, max_start)
        group = [start + i * gap for i in range(K)]
        groups.append(group)
    return groups


def sample_groups_high(total_sentences, M, K):
    """
    high: Randomly sample K sentences across the entire corpus.
    """
    groups = []
    all_indices = list(range(total_sentences))
    for _ in range(M):
        group = sorted(random.sample(all_indices, K))
        groups.append(group)
    return groups


def main():
    print("=== Loading and preparing MINE dataset ===")
    dl = DataLoader(clean=True, split_sentence=True)

    global_sentences = dl.load_txt_and_split_sentences("../data/Alice.txt")
    # global_sentences = dl.load_txt_and_split_sentences("../data/Pride.txt")
    # global_sentences = dl.load_essays()[0:2600]
    # global_sentences = dl.load_wikipedia(max_articles=5000)
    # global_sentences = dl.load_arxiv_streaming(max_papers=5000)

    total_sentences = len(global_sentences)

    print("=== Computing sentence embeddings (global) ===")
    all_embs = encode_texts(global_sentences)  # Unified embedding entry

    print("=== Sampling sentence groups ===")
    groups_local = sample_groups_local(total_sentences, M, K)
    groups_medium = sample_groups_medium(total_sentences, M, K, gap=GAP)
    groups_high = sample_groups_high(total_sentences, M, K)

    print(
        f"Sampled groups: local={len(groups_local)}, "
        f"medium={len(groups_medium)}, high={len(groups_high)}"
    )

    results = []

    def process_mode(mode_name, groups):
        for group_idxs in tqdm(groups, desc=f"Processing {mode_name}"):
            sent_list = [global_sentences[i] for i in group_idxs]
            sent_embs = all_embs[group_idxs]

            # C_pair
            C_pair = compute_C_pair(sent_embs)

            # Text embedding (concatenate sentences)
            concat_text = " ".join(sent_list)
            text_emb = encode_texts([concat_text])[0]

            # C_mean
            C_mean = compute_C_mean(sent_embs, text_emb)

            results.append({
                "mode": mode_name,
                "C_pair": C_pair,
                "C_mean": C_mean
            })

    process_mode("local", groups_local)
    process_mode("medium", groups_medium)
    process_mode("high", groups_high)

    df = pd.DataFrame(results)
    out_csv = "./result/cpair_cmean_correlation.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Statistical tests
    lines = []
    lines.append("\n=== Correlation analysis (C_pair vs C_mean) ===\n")

    overall_spearman = spearmanr(df["C_pair"], df["C_mean"])
    overall_kendall = kendalltau(df["C_pair"], df["C_mean"])

    lines.append(
        f"[Overall] Spearman: rho={overall_spearman.correlation:.4f}, "
        f"p={overall_spearman.pvalue:.2e}\n"
    )
    lines.append(
        f"[Overall] Kendall : tau={overall_kendall.correlation:.4f}, "
        f"p={overall_kendall.pvalue:.2e}\n"
    )

    for mode in ["local", "medium", "high"]:
        sub = df[df["mode"] == mode]
        if len(sub) < 5:
            continue
        sp = spearmanr(sub["C_pair"], sub["C_mean"])
        kt = kendalltau(sub["C_pair"], sub["C_mean"])
        lines.append(f"[{mode}] Spearman: rho={sp.correlation:.4f}, p={sp.pvalue:.2e}\n")
        lines.append(f"[{mode}] Kendall : tau={kt.correlation:.4f}, p={kt.pvalue:.2e}\n")

    # Write to file
    out_txt = "./result/spearman_kendall_txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Saved correlation analysis to: {out_txt}")


if __name__ == "__main__":
    main()

