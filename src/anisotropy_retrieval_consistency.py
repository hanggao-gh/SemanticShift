import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tools import analyze_embeddings_geometry
from qdrant_store import QdrantSentenceStore


class AnisotropyRetrievalConsistency:
    def __init__(
        self,
        variant2collection: dict,  # Mapping from each variant (S/S2/S5/S10) to a Qdrant collection name
        seed: int = 42,
    ):
        """
        Args:
            variant2collection: A dict mapping variant names to Qdrant collection names, e.g.
                {
                    "repeat_S": "mine_repeat_S_sentence-transformers_all-mpnet-base-v2",
                    "repeat_S2": "...",
                    "consecutive_S2": "...",
                    ...
                }
            seed: Random seed.
        """
        self.variant2collection = variant2collection
        self.rng = np.random.default_rng(seed)

    def _get_embeddings_from_qdrant(self, variant_name: str) -> np.ndarray:
        """Fetch all embeddings from the Qdrant collection of the given variant and return them sorted by ID."""
        collection = self.variant2collection[variant_name]
        store = QdrantSentenceStore(collection)

        # 1) Check collection existence
        try:
            _ = store.client.get_collection(collection)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Qdrant collection '{collection}' does not exist.\n"
                f"Did you forget to run preprocess_to_qdrant_multi.py ?\n"
                f"Original error: {e}"
            )

        # 2) Fetch records
        ids, texts, embeddings = store.fetch_all(with_vectors=True)
        # print(ids[1000:1010])  # Sanity check whether fetched records are ordered

        if len(embeddings) == 0:
            raise RuntimeError(
                f"[ERROR] Qdrant collection '{collection}' is EMPTY.\n"
                f"Something went wrong in preprocess_to_qdrant_multi.py."
            )

        order = np.argsort(ids)
        return embeddings[order]

    def _to_serializable(self, obj):
        """Convert numpy/scalar containers to JSON-serializable Python types."""
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        return obj

    def _save_results(self, name: str, result_dict: dict, result_dir: str = "./result/"):
        """Save results to both JSON and CSV under result_dir."""
        safe_res = self._to_serializable(result_dict)

        json_path = f"{result_dir}{name}.json"
        csv_path = f"{result_dir}{name}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(safe_res, f, indent=2, ensure_ascii=False)

        pd.DataFrame([safe_res["metrics"]]).to_csv(csv_path, index=False)
        print(f"Saved: {json_path}, {csv_path}")

    def _run(self, variant_name: str) -> dict:
        """Unified pipeline: pull embeddings from Qdrant -> sort -> evaluate geometry metrics."""
        print(f"\n=== Running {variant_name} ===")

        embeddings = self._get_embeddings_from_qdrant(variant_name)
        metrics = analyze_embeddings_geometry(embeddings)

        res = {
            "variant_name": variant_name,
            "metrics": metrics,
        }

        self._save_results(variant_name, res)
        return res

    def retrieval_consistency(self, query_num: int = 100, top_k: int = 10, concat_model: str = "repeat"):
        """
        Evaluate consistency of nearest-neighbor retrieval across S, S2, S5, S10 embeddings.

        The test compares the top-k retrieval results of S against {S2, S5, S10}.
        """
        print(f"\n=== Running Consistency Test: model={concat_model}, query={query_num}, top_k={top_k} ===")

        # ------------------------------------------------------------
        # 1) Load four embedding sets
        # ------------------------------------------------------------
        variants = ["S", "S2", "S5", "S10"]
        embeddings_dict = {}

        for v in variants:
            key = f"{concat_model}_{v}"
            embeddings_dict[v] = self._get_embeddings_from_qdrant(key)
            print(f"Loaded {key} → shape={embeddings_dict[v].shape}")

        base_embeds = embeddings_dict["S"]
        rng = np.random.default_rng(self.rng)
        sample_size = min(len(base_embeds), query_num)
        query_indices = rng.choice(len(base_embeds), size=sample_size, replace=False)

        # ------------------------------------------------------------
        # Utility: batch cosine similarity + top-k retrieval
        # ------------------------------------------------------------
        def _sim_query_batch(query_vecs: np.ndarray, tgt_emb: np.ndarray, k: int):
            sims = cosine_similarity(query_vecs, tgt_emb)  # shape (Q, N)
            top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]  # fast top-k
            # Sort within the top-k by similarity (descending)
            sorted_top = np.take_along_axis(
                top_idx,
                np.argsort(np.take_along_axis(sims, top_idx, axis=1), axis=1)[:, ::-1],
                axis=1,
            )
            return sorted_top, sims

        # ------------------------------------------------------------
        # 2) Retrieve top_k for each query across all four sets
        # ------------------------------------------------------------
        result_top = {}

        for v in variants:
            top_indices, _ = _sim_query_batch(
                base_embeds[query_indices], embeddings_dict[v], top_k
            )
            result_top[v] = top_indices  # shape = (Q, K)

        # ------------------------------------------------------------
        # 3) Compute top-k overlap consistency — top1, top3, top5, top10
        # ------------------------------------------------------------
        def compute_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
            cnt = 0
            for i in range(len(a)):
                overlap_num = len(set(a[i][:k]) & set(b[i][:k]))
                cnt += overlap_num
            return cnt / (len(a) * k)

        ks = [1, 3, 5, 10]
        metrics = {f"top{k}_overlap": {} for k in ks}

        for k in ks:
            for v in ["S2", "S5", "S10"]:
                metrics[f"top{k}_overlap"][v] = compute_overlap(result_top["S"], result_top[v], k)

        # ------------------------------------------------------------
        # 4) Save result object
        # ------------------------------------------------------------
        save_dict = {
            "concat_model": concat_model,
            "query_num": query_num,
            "top_k": top_k,
            "metrics": metrics,
        }
        json_path = f"./result/{concat_model}_retrieval.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)

        return save_dict


    # ------------------------------------------------------------------
    #   Repeat experiment (S, S2, S5, S10)
    # ------------------------------------------------------------------
    def run_repeat_experiment(self):
        results = {
            name: self._run(f"repeat_{name}")
            for name in ["S", "S2", "S5", "S10"]
        }
        return results

    # ------------------------------------------------------------------
    def run_consecutive_experiment(self):
        results = {
            name: self._run(f"consecutive_{name}")
            for name in ["S", "S2", "S5", "S10"]
        }
        return results

    # ------------------------------------------------------------------
    def run_random_experiment(self):
        results = {
            name: self._run(f"random_{name}")
            for name in ["S", "S2", "S5", "S10"]
        }
        return results


# =====================================================================
# Example usage
# =====================================================================
if __name__ == "__main__":
    model_name = "BAAI/bge-large-en-v1.5".replace("/", "_")
    # model_name = "intfloat/e5-large-v2".replace("/", "_")
    # model_name = "sentence-transformers/all-mpnet-base-v2"

    dataset = "alice"
    # dataset = "pride"
    # dataset = "wikipedia"
    # dataset = "mine"
    # dataset = "arxiv"

    # ---- All 10 collection names ----
    variant2collection = {
        "repeat_S":   f"{dataset}_repeat_S_{model_name}",
        "repeat_S2":  f"{dataset}_repeat_S2_{model_name}",
        "repeat_S5":  f"{dataset}_repeat_S5_{model_name}",
        "repeat_S10": f"{dataset}_repeat_S10_{model_name}",

        "consecutive_S":   f"{dataset}_repeat_S_{model_name}",
        "consecutive_S2":  f"{dataset}_consecutive_S2_{model_name}",
        "consecutive_S5":  f"{dataset}_consecutive_S5_{model_name}",
        "consecutive_S10": f"{dataset}_consecutive_S10_{model_name}",

        "random_S":   f"{dataset}_repeat_S_{model_name}",
        "random_S2":  f"{dataset}_random_S2_{model_name}",
        "random_S5":  f"{dataset}_random_S5_{model_name}",
        "random_S10": f"{dataset}_random_S10_{model_name}",
    }

    exp = AnisotropyRetrievalConsistency(variant2collection, seed=42)

    exp.run_repeat_experiment()
    exp.run_consecutive_experiment()
    exp.run_random_experiment()

    for pattern in ["repeat", "consecutive", "random"]:
        exp.retrieval_consistency(query_num=1000, concat_model=pattern)
