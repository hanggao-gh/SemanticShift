import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import DataLoader
from qdrant_store import QdrantSentenceStore

# Hugging Face Mirror
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ------------------------------------------------------------
# 1. Generate all sentence variants
# ------------------------------------------------------------
def build_repeat_variants(sentences):
    S = sentences
    S2 = [s + " " + s for s in sentences]
    S5 = [" ".join([s] * 5) for s in sentences]
    S10 = [" ".join([s] * 10) for s in sentences]
    return {"S": S, "S2": S2, "S5": S5, "S10": S10}


def build_consecutive_variants(sentences):
    S = sentences
    S2 = [
        sentences[i] + " " + sentences[i + 1]
        if i < len(sentences) - 1 else sentences[i]
        for i in range(len(sentences))
    ]
    S5 = [
        sentences[i] + " " + " ".join(sentences[i+1:i+5])
        for i in range(len(sentences))
    ]
    S10 = [
        sentences[i] + " " + " ".join(sentences[i+1:i+10])
        for i in range(len(sentences))
    ]
    return {"S": S, "S2": S2, "S5": S5, "S10": S10}


def build_random_variants(sentences, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sentences))

    S = sentences
    S2 = [
        s + " " + sentences[rng.choice(indices)]
        for s in sentences
    ]
    S5 = [
        s + " " + " ".join(
            sentences[j] for j in rng.choice(indices, size=4, replace=False)
        )
        for s in sentences
    ]
    S10 = [
        s + " " + " ".join(
            sentences[j] for j in rng.choice(indices, size=9, replace=False)
        )
        for s in sentences
    ]
    return {"S": S, "S2": S2, "S5": S5, "S10": S10}


# ------------------------------------------------------------
# 2. Process function: create collection + embed + store
# ------------------------------------------------------------
def create_collection_for_variant(
    variant_name,
    model,
    dataset_name,
    split_name,
    sentences_variant,
    model_name
):
    collection_name = f"{dataset_name}_{variant_name}_{model_name.replace('/', '_')}"
    print(f"\n=== Creating collection: {collection_name} ===")

    store = QdrantSentenceStore(collection_name)

    # Compute embeddings
    embeddings = model.encode(sentences_variant, batch_size=32, show_progress_bar=True)
    dim = embeddings.shape[1]

    # Create or recreate the collection
    store.create_or_recreate_collection(dim, distance="cosine", use_hnsw=False)

    # Upsert
    store.upsert_sentences(
        sentences=sentences_variant,
        embeddings=embeddings,
        base_idx=0,
        dataset=dataset_name,
        split=split_name,
        model_name=model_name,
        batch_size=200,
    )


# ------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["alice", "pride", "mine", "wikipedia", "arxiv"])
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model

    # Load data
    loader = DataLoader(clean=True, split_sentence=True)
    if dataset_name == "alice":
        sentences = loader.load_txt_and_split_sentences("../data/Alice.txt")
        split = "full"
    elif dataset_name == "pride":
        sentences = loader.load_txt_and_split_sentences("../data/Pride.txt")
        split = "full"
    elif dataset_name == "wikipedia":
        sentences = loader.load_wikipedia(max_articles=5000)
        split = "full"
    elif dataset_name == "arxiv":
        sentences = loader.load_arxiv_streaming(max_papers=5000)
        split = "full"
    else:
        sentences = loader.load_essays()
        split = "train"

    print(f"Loaded {len(sentences)} sentences\n")

    # Load embedding model once
    model = SentenceTransformer(model_name)

    # 1) repeat variants
    repeat_variants = build_repeat_variants(sentences)

    # 2) consecutive variants
    consecutive_variants = build_consecutive_variants(sentences)

    # 3) random variants
    random_variants = build_random_variants(sentences, seed=42)

    # ------------------------------------------------------------
    # Total 10 collections
    # repeat: S, S2, S5, S10
    # consecutive: S2, S5, S10
    # random: S2, S5, S10
    # ------------------------------------------------------------

    # ---- repeat ----
    for name in ["S", "S2", "S5", "S10"]:
        create_collection_for_variant(
            f"repeat_{name}",
            model,
            dataset_name,
            split,
            repeat_variants[name],
            model_name
        )

    # ---- consecutive ----
    for name in ["S2", "S5", "S10"]:
        create_collection_for_variant(
            f"consecutive_{name}",
            model,
            dataset_name,
            split,
            consecutive_variants[name],
            model_name
        )

    # ---- random ----
    for name in ["S2", "S5", "S10"]:
        create_collection_for_variant(
            f"random_{name}",
            model,
            dataset_name,
            split,
            random_variants[name],
            model_name
        )

    print("\nAll 10 collections created successfully.")


if __name__ == "__main__":
    main()
