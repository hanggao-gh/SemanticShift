# qdrant_store.py
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from qdrant_client import QdrantClient, models


class QdrantSentenceStore:
    """
    Helper for storing / retrieving sentence embeddings in Qdrant.

    Unified payload schema:
      - dataset: str
      - split: str
      - model_name: str
      - text: str
      - idx: int
      - (optional) doc_id, source, etc.
    """

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)

    # ------------------------ collection ------------------------
    def create_or_recreate_collection(
        self,
        vector_size: int,
        distance: str = "cosine",
        use_hnsw: bool = True,
    ) -> None:
        """
        Create or recreate a collection with given vector size and distance.
        Optionally specify HNSW config to speed up ANN search.
        """
        if distance.lower() == "cosine":
            dist = models.Distance.COSINE
        elif distance.lower() == "euclid":
            dist = models.Distance.EUCLID
        else:
            dist = models.Distance.DOT

        hnsw_config = None
        if use_hnsw:
            hnsw_config = models.HnswConfig(
                m=16,          # graph degree
                ef_construct=128,
                full_scan_threshold=10_000,
            )

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=dist,
            ),
            hnsw_config=hnsw_config,
        )
        print(
            f"[Qdrant] Collection '{self.collection_name}' "
            f"created with dim={vector_size}, distance={dist}, hnsw={use_hnsw}."
        )

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def count_points(self, exact: bool = True) -> int:
        """
        Count points in the collection.
        """
        res = self.client.count(
            collection_name=self.collection_name,
            exact=exact,
        )
        return res.count

    # ------------------------ insert / upsert ------------------------
    def upsert_sentences(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        base_idx: int = 0,
        dataset: str = "",
        split: str = "",
        model_name: str = "",
        extra_payload: Optional[Dict[str, Any]] = None,
        batch_size: int = 200,
    ) -> None:
        """
        Upsert sentences into Qdrant in small batches to avoid 32MB payload limit.
        """

        total = len(sentences)
        print(f"[Qdrant] Begin upsert {total} sentences (batch={batch_size})")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_sent = sentences[start:end]
            batch_emb  = embeddings[start:end]

            ids = []
            payloads = []

            for i, text in enumerate(batch_sent):
                pid = base_idx + start + i
                ids.append(pid)

                payload = {
                    "dataset": dataset,
                    "split": split,
                    "model_name": model_name,
                    "text": text,
                    "idx": pid,
                }
                if extra_payload:
                    payload.update(extra_payload)

                payloads.append(payload)

            vectors = batch_emb.tolist()

            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )
            print(f"  - Upserted batch {start} → {end}")

        print(f"[Qdrant] Finished upserting {total} sentences.")

    # ------------------------ basic retrieve / search ------------------------
    def get_point(self, point_id: int, with_vector: bool = True):
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=with_vector,
        )
        return res[0] if res else None

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        query_filter: Optional[models.Filter] = None,
    ):
        if isinstance(query_vector, np.ndarray):
            qv = query_vector.tolist()
        else:
            qv = list(query_vector)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=qv,
            limit=top_k,
            query_filter=query_filter,
        )
        return hits

    def fetch_all(
        self,
        batch_size: int = 1024,
        with_vectors: bool = True,
    ) -> Tuple[List[int], List[str], np.ndarray]:
        """
        Scroll through the whole collection and return:
        - ids: list[int]
        - texts: list[str]
        - embeddings: np.ndarray, shape (N, dim)
        """
        all_ids = []
        all_texts = []
        all_vectors = []

        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=with_vectors,
                with_payload=True,
            )
            if not points:
                break

            for p in points:
                all_ids.append(p.id)
                text = p.payload.get("text", "")
                all_texts.append(text)
                if with_vectors and p.vector is not None:
                    all_vectors.append(p.vector)

            if offset is None:
                break

        if all_vectors:
            vecs = np.array(all_vectors, dtype=np.float32)
        else:
            vecs = np.empty((0, 0), dtype=np.float32)

        print(
            f"[Qdrant] Fetched {len(all_ids)} points from '{self.collection_name}'."
        )
        return all_ids, all_texts, vecs

    # ------------------------ backup (snapshot) ------------------------
    def create_snapshot(self):
        snapshot_info = self.client.create_snapshot(self.collection_name)
        print(f"[Qdrant] Snapshot created: {snapshot_info.name}")
        return snapshot_info

    def list_snapshots(self):
        snapshots = self.client.list_snapshots(self.collection_name)
        for s in snapshots:
            print(f"- {s.name} (size={s.size} bytes)")
        return snapshots

    def recover_from_snapshot(self, snapshot_location: str):
        res = self.client.recover_snapshot(
            self.collection_name,
            snapshot_location,
        )
        print(f"[Qdrant] Recover snapshot from {snapshot_location}: {res.status}")
        return res
