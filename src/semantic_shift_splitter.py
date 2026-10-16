# -*- coding: utf-8 -*-
"""
Evaluate 3 splitters on multiple data sources (arXiv JSON paragraphs / txt).

Splitters:
  1) FixedSentenceGroupNodeParser: fixed N sentences per chunk (sentence-boundary aligned)
  2) SemanticSplitterNodeParser (LlamaIndex): cosine dissimilarity percentile breakpoints
  3) SemanticShiftDriftSplitterNodeParser: drift accumulated by Shift=Local*Disp + max_tokens

Metrics:
  - Boundary Precision / Recall / F1
  - Pk
  - WindowDiff
  - Chunk statistics: avg_sents/chunk, within-doc var(sents/chunk)

Install (suggested):
  pip install llama-index==0.14.10
  pip install llama-index-embeddings-huggingface
  pip install sentence-transformers torch
  pip install numpy tqdm requests nltk

Examples:
  # 1) arxiv.json (list of paragraphs)
  python semantic_shift_eval_multi.py --dataset arxiv_json --arxiv_json_path arxiv.json \
    --fixed_sents_per_chunk 5 --semantic_buffer 1 --semantic_percentile 95 \
    --shift_buffer 2 --shift_percentile 95 --shift_max_tokens 256 --shift_min_chunk_sents 1

  # 2) txt
  python semantic_shift_eval_multi.py --dataset mine --txt_path mine.txt \
    --fixed_sents_per_chunk 5 --semantic_buffer 1 --semantic_percentile 95 \
    --shift_buffer 2 --shift_percentile 95 --shift_max_tokens 256 --shift_min_chunk_sents 1
"""

from __future__ import annotations

import argparse
import os
import re
import tarfile
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from tqdm import tqdm

# ----------------------------
# LlamaIndex imports (0.14.10)
# ----------------------------
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, SerializeAsAny, WithJsonSchema
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits, default_id_func
from llama_index.core.schema import BaseNode, Document

# Semantic splitter
from llama_index.core.node_parser.text.semantic_splitter import SemanticSplitterNodeParser

# HF embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ----------------------------
# NLTK sentence tokenize
# ----------------------------
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ============================================================
# 1) Shared helpers: sentence tokenization + paragraph splitting
# ============================================================

def _nltk_sent_tokenize(text: str) -> List[str]:
    sents = nltk.sent_tokenize(text)
    out = []
    for s in sents:
        s = s.strip()
        if s:
            out.append(s)
    return out


def _split_paragraphs(text: str) -> List[str]:
    """
    Split by blank lines into paragraphs.
    Keeps paragraph content, strips extra whitespace.
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    paras = [p.strip() for p in t.split("\n\n")]
    paras = [p for p in paras if p]
    return paras


# ============================================================
# 2) Dataset loader B: arxiv.json (list of paragraphs -> ground truth)
# ============================================================

def load_paragraph_list_json(json_path: str, doc_id: str = "arxiv") -> List[Dict[str, Any]]:
    """
    json file: a list[str], each str is a paragraph/segment.
    Ground truth segmentation = paragraph boundaries.
    Returns a list with one document: [{"id", "sentences", "labels"}]
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Expected JSON format: a list of strings (paragraphs).")

    paragraphs = [p.strip() for p in data]
    paragraphs = [p for p in paragraphs if p]

    sentences: List[str] = []
    labels: List[int] = []

    for p in paragraphs:
        # normalize internal whitespace
        p = p.replace("\r", " ").replace("\n", " ")
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue

        sents = _nltk_sent_tokenize(p)
        if not sents:
            continue
        for i, s in enumerate(sents):
            s = s.replace("\n", " ").replace("\r", " ").strip()
            if not s:
                continue
            sentences.append(s)
            labels.append(1 if i == len(sents) - 1 else 0)

    if len(sentences) <= 2:
        raise RuntimeError("Too few sentences after parsing arxiv.json.")

    return [{"id": doc_id, "sentences": sentences, "labels": labels}]


# ============================================================
# 3) Dataset loader C: txt -> ground truth by natural paragraphs
# ============================================================

def _gutenberg_strip_header_footer(raw: str) -> str:
    """
    Best-effort: strip Project Gutenberg header/footer if markers exist.
    If markers don't exist, return original.
    """
    t = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Common Gutenberg markers (varies by edition)
    start_pat = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    end_pat = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

    m1 = start_pat.search(t)
    m2 = end_pat.search(t)

    if m1 and m2 and m2.start() > m1.end():
        t = t[m1.end(): m2.start()]

    return t.strip()


def _clean_text(raw: str) -> str:
    """
    Minimal cleaning to preserve paragraph boundaries:
      - strip Gutenberg header/footer
      - normalize line endings
      - collapse extra blank lines
      - remove decorative separator-only paragraphs like '* * * *'
    """
    t = _gutenberg_strip_header_footer(raw)

    # normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Remove lines that are just asterisks/spaces (decorative)
    # e.g. "*      *      *" or "***"
    lines = t.split("\n")
    kept_lines: List[str] = []
    for ln in lines:
        if re.fullmatch(r"[\s\*]+", ln or ""):
            # keep a blank line to preserve paragraph split signal
            kept_lines.append("")
        else:
            kept_lines.append(ln.rstrip())

    t = "\n".join(kept_lines)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def load_txt(txt_path: str, doc_id: str = "mine") -> List[Dict[str, Any]]:
    """
    Read pg11.txt, clean, split into paragraphs, use paragraph boundary as ground truth.
    Returns a list with one document: [{"id", "sentences", "labels"}]
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"txt path not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    cleaned = _clean_text(raw)
    paragraphs = _split_paragraphs(cleaned)

    sentences: List[str] = []
    labels: List[int] = []

    for p in paragraphs:
        # remove excessive spaces but keep punctuation
        p = p.replace("\n", " ")
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue

        sents = _nltk_sent_tokenize(p)
        if not sents:
            continue

        for i, s in enumerate(sents):
            s = s.replace("\n", " ").replace("\r", " ").strip()
            if not s:
                continue
            sentences.append(s)
            labels.append(1 if i == len(sents) - 1 else 0)

    if len(sentences) <= 2:
        raise RuntimeError("Too few sentences after parsing Alice txt.")

    return [{"id": doc_id, "sentences": sentences, "labels": labels}]


# ============================================================
# 4) Sentence-IO utilities (line-based)
# ============================================================

def build_text_from_sentences(sentences: List[str]) -> str:
    # Exactly one sentence per line; NO internal newline in sentence
    return "\n".join([s.strip() for s in sentences])


def newline_sentence_splitter(text: str) -> List[str]:
    """
    Split by line, keep line-break as delimiter so "".join keeps sentence boundaries.
    """
    lines = text.split("\n")
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        return []
    out = []
    for i, ln in enumerate(lines):
        ln = ln.strip()
        if i < len(lines) - 1:
            out.append(ln + "\n")
        else:
            out.append(ln)
    return out


def _normalize_chunk_text_for_counting(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text.strip())
    return text


def count_lines_as_sentences(chunk_text: str) -> int:
    t = _normalize_chunk_text_for_counting(chunk_text)
    if not t:
        return 0
    return len([x for x in t.split("\n") if x.strip() != ""])


def get_node_text(node) -> str:
    try:
        return node.get_content(metadata_mode="none")
    except TypeError:
        return node.get_content()


# ============================================================
# 5) Gold/pred boundaries + metrics
# ============================================================

def labels_to_gold_boundaries(labels: List[int]) -> np.ndarray:
    """
    labels[i] == 1 means boundary AFTER sentence i (end-of-segment).
    Boundaries length = n-1 => use labels[0:n-1].
    """
    n = len(labels)
    if n <= 1:
        return np.zeros((0,), dtype=np.int32)
    return np.array([int(x) for x in labels[:-1]], dtype=np.int32)


def chunks_to_pred_boundaries(chunks_texts: List[str], n_sentences: int) -> np.ndarray:
    """
    Convert chunk texts to boundary vector length n-1.
    Assumption: each sentence is one line in doc text, and splitters preserve '\n' between sentences.
    """
    if n_sentences <= 1:
        return np.zeros((0,), dtype=np.int32)

    counts = [count_lines_as_sentences(c) for c in chunks_texts]
    total = sum(counts)

    if total != n_sentences:
        raise ValueError(
            f"Sentence count mismatch: chunks sum={total}, gold={n_sentences}. "
            f"counts(head)={counts[:5]} ... last={counts[-1] if counts else None}"
        )

    y = np.zeros((n_sentences - 1,), dtype=np.int32)
    end = 0
    for k in counts[:-1]:
        end += k
        y[end - 1] = 1
    return y


def boundary_prf(y_pred: np.ndarray, y_gold: np.ndarray) -> Tuple[float, float, float]:
    tp = int(np.sum((y_pred == 1) & (y_gold == 1)))
    fp = int(np.sum((y_pred == 1) & (y_gold == 0)))
    fn = int(np.sum((y_pred == 0) & (y_gold == 1)))
    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)
    return p, r, f1


def boundaries_to_seg_ids(boundaries: np.ndarray) -> np.ndarray:
    n = len(boundaries) + 1
    seg = np.zeros((n,), dtype=np.int32)
    cur = 0
    for i in range(1, n):
        if boundaries[i - 1] == 1:
            cur += 1
        seg[i] = cur
    return seg


def pk_metric(y_pred: np.ndarray, y_gold: np.ndarray, window: Optional[int] = None) -> float:
    gold_seg = boundaries_to_seg_ids(y_gold)
    pred_seg = boundaries_to_seg_ids(y_pred)
    n = len(gold_seg)

    if window is None:
        num_segs = int(gold_seg[-1]) + 1
        avg_len = n / max(num_segs, 1)
        window = int(round(avg_len / 2))
        window = max(window, 1)

    errors = 0
    total = 0
    for i in range(n - window):
        j = i + window
        same_gold = (gold_seg[i] == gold_seg[j])
        same_pred = (pred_seg[i] == pred_seg[j])
        if same_gold != same_pred:
            errors += 1
        total += 1
    return errors / max(total, 1)


def windowdiff_metric(y_pred: np.ndarray, y_gold: np.ndarray, window: Optional[int] = None) -> float:
    n = len(y_gold) + 1

    if window is None:
        num_segs = int(boundaries_to_seg_ids(y_gold)[-1]) + 1
        avg_len = n / max(num_segs, 1)
        window = int(round(avg_len / 2))
        window = max(window, 1)

    errors = 0
    total = 0
    for i in range(0, (n - window)):
        gold_cnt = int(np.sum(y_gold[i : i + window]))
        pred_cnt = int(np.sum(y_pred[i : i + window]))
        if gold_cnt != pred_cnt:
            errors += 1
        total += 1
    return errors / max(total, 1)


# ============================================================
# 6) Fixed sentence-group splitter (sentence boundary aligned)
# ============================================================

SentenceSplitterCallable = WithJsonSchema({"type": "string"}, mode="serialization")


class FixedSentenceGroupNodeParser(NodeParser):
    sentences_per_chunk: int = Field(default=6, description="Fixed number of sentences per chunk.")
    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=lambda: newline_sentence_splitter, exclude=True
    )

    @classmethod
    def class_name(cls) -> str:
        return "FixedSentenceGroupNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        out: List[BaseNode] = []
        for node in nodes:
            doc = node  # Document is also BaseNode
            sents = self.sentence_splitter(doc.text)
            chunks: List[str] = []
            k = max(int(self.sentences_per_chunk), 1)
            for i in range(0, len(sents), k):
                chunks.append("".join(sents[i : i + k]))
            out.extend(build_nodes_from_splits(chunks, doc, id_func=self.id_func))
        return out


# ============================================================
# 7) Semantic Shift splitter
# ============================================================

TokenCounterCallable = Callable[[str], int]


def _default_token_counter(text: str) -> int:
    return len(text.split())


class SemanticShiftSplitterNodeParser(NodeParser):
    """
    Drift-style semantic shift splitter.

    Build chunks left-to-right; keep adding sentences until:
      (1) Shift(segment) would exceed threshold -> cut BEFORE adding (drift accumulated)
      (2) token budget would exceed max_tokens  -> cut (hard cap)

    Shift(k) = Local(k) * Disp(k)
      Local(k) = sum_{i=1..k-1} (1 - cos(e_i, e_{i+1}))
      Disp(k)  = mean_{i<j} (1 - cos(e_i, e_j))
    """

    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=lambda: newline_sentence_splitter, exclude=True
    )

    embed_model: SerializeAsAny[BaseEmbedding] = Field(description="Embedding model for semantic computations.")

    buffer_size: int = Field(default=2, description="Window radius for threshold estimation; window=2*buffer+1.")
    breakpoint_percentile_threshold: int = Field(default=95, description="Percentile for drift threshold estimation.")

    shift_threshold: Optional[float] = Field(default=None, description="If None, estimate per doc from window shifts.")

    max_tokens: int = Field(default=256, description="Hard cap per chunk (approx by token_counter).")
    token_counter: TokenCounterCallable = Field(default=_default_token_counter, exclude=True)

    min_chunk_sentences: int = Field(default=1, description="Minimum sentences per chunk before allowing a cut.")

    @classmethod
    def class_name(cls) -> str:
        return "SemanticShiftDriftSplitterNodeParser"

    @classmethod
    def from_defaults(
        cls,
        embed_model: BaseEmbedding,
        buffer_size: int = 2,
        breakpoint_percentile_threshold: int = 95,
        shift_threshold: Optional[float] = None,
        max_tokens: int = 256,
        token_counter: Optional[TokenCounterCallable] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        min_chunk_sentences: int = 1,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "SemanticShiftSplitterNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func
        token_counter = token_counter or _default_token_counter
        sentence_splitter = sentence_splitter or newline_sentence_splitter
        return cls(
            embed_model=embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            shift_threshold=shift_threshold,
            max_tokens=max_tokens,
            token_counter=token_counter,
            sentence_splitter=sentence_splitter,
            min_chunk_sentences=min_chunk_sentences,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        out: List[BaseNode] = []
        for node in nodes:
            doc = node
            out.extend(self._build_from_document(doc, show_progress=show_progress))
        return out

    def _build_from_document(self, doc: Document, show_progress: bool = False) -> List[BaseNode]:
        sents_with_delim = self.sentence_splitter(doc.text)
        if not sents_with_delim:
            return []

        sents_clean = [s.replace("\n", " ").strip() for s in sents_with_delim]
        sent_embs = self.embed_model.get_text_embedding_batch(sents_clean, show_progress=show_progress)

        tau = self.shift_threshold
        if tau is None:
            tau = self._estimate_shift_threshold(sent_embs)

        chunks = self._segment_by_drift_and_tokens(sents_with_delim, sents_clean, sent_embs, tau)
        return build_nodes_from_splits(chunks, doc, id_func=self.id_func)

    def _estimate_shift_threshold(self, sent_embs: List[List[float]]) -> float:
        n = len(sent_embs)
        if n <= 1:
            return 0.0
        b = max(int(self.buffer_size), 0)

        window_shifts: List[float] = []
        for i in range(n):
            left = max(0, i - b)
            right = min(n, i + b + 1)
            window_shifts.append(self._shift_of_ordered_embeddings(sent_embs[left:right]))

        return float(np.percentile(window_shifts, self.breakpoint_percentile_threshold))

    def _shift_of_ordered_embeddings(self, embs: List[List[float]]) -> float:
        k = len(embs)
        if k <= 1:
            return 0.0

        local = 0.0
        for i in range(k - 1):
            sim = float(self.embed_model.similarity(embs[i], embs[i + 1]))
            local += (1.0 - sim)

        pair_sum = 0.0
        cnt = 0
        for i in range(k):
            for j in range(i + 1, k):
                sim = float(self.embed_model.similarity(embs[i], embs[j]))
                pair_sum += (1.0 - sim)
                cnt += 1
        disp = pair_sum / cnt if cnt > 0 else 0.0

        return local * disp

    def _segment_by_drift_and_tokens(
        self,
        sents_with_delim: List[str],
        sents_clean: List[str],
        sent_embs: List[List[float]],
        tau: float,
    ) -> List[str]:
        n = len(sents_with_delim)
        if n == 1:
            return [sents_with_delim[0]]

        chunks: List[str] = []
        cur_start = 0
        cur_text = ""
        cur_tokens = 0

        local_sum = 0.0
        pair_sum = 0.0
        k = 0

        def reset_state():
            nonlocal cur_text, cur_tokens, local_sum, pair_sum, k
            cur_text = ""
            cur_tokens = 0
            local_sum = 0.0
            pair_sum = 0.0
            k = 0

        reset_state()

        for i in range(n):
            sent_text = sents_with_delim[i]
            sent_tok = int(self.token_counter(sents_clean[i]))
            emb = sent_embs[i]

            # hard cap by tokens
            if k > 0 and (cur_tokens + sent_tok) > self.max_tokens:
                chunks.append(cur_text)
                cur_start = i
                reset_state()

            # check drift threshold if we add current sentence
            if k >= 1:
                last_emb = sent_embs[cur_start + (k - 1)]
                sim_adj = float(self.embed_model.similarity(last_emb, emb))
                local_add = (1.0 - sim_adj)

                pair_add = 0.0
                for j in range(cur_start, cur_start + k):
                    sim = float(self.embed_model.similarity(sent_embs[j], emb))
                    pair_add += (1.0 - sim)

                k_new = k + 1
                local_new = local_sum + local_add
                pair_new = pair_sum + pair_add

                cnt_pairs_new = k_new * (k_new - 1) // 2
                disp_new = pair_new / float(cnt_pairs_new) if cnt_pairs_new > 0 else 0.0
                shift_new = local_new * disp_new

                if (k >= self.min_chunk_sentences) and (shift_new > tau):
                    chunks.append(cur_text)
                    cur_start = i
                    reset_state()

            # actually append
            if k == 0:
                cur_text = sent_text
                cur_tokens = sent_tok
                k = 1
            else:
                last_emb = sent_embs[cur_start + (k - 1)]
                sim_adj = float(self.embed_model.similarity(last_emb, emb))
                local_sum += (1.0 - sim_adj)

                for j in range(cur_start, cur_start + k):
                    sim = float(self.embed_model.similarity(sent_embs[j], emb))
                    pair_sum += (1.0 - sim)

                cur_text += sent_text
                cur_tokens += sent_tok
                k += 1

        if cur_text:
            chunks.append(cur_text)

        return chunks


# ============================================================
# 8) Build splitters
# ============================================================

def build_fixed_splitter(sentences_per_chunk: int) -> FixedSentenceGroupNodeParser:
    return FixedSentenceGroupNodeParser(
        sentences_per_chunk=sentences_per_chunk,
        sentence_splitter=newline_sentence_splitter,
    )


def build_semantic_splitter(embed_model: BaseEmbedding, buffer_size: int, percentile: int) -> SemanticSplitterNodeParser:
    return SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=percentile,
        sentence_splitter=newline_sentence_splitter,
    )


def build_shift_splitter(
    embed_model: BaseEmbedding,
    buffer_size: int,
    percentile: int,
    max_tokens: int,
    min_chunk_sents: int,
) -> SemanticShiftSplitterNodeParser:
    return SemanticShiftSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=percentile,
        shift_threshold=None,  # auto estimate per doc
        max_tokens=max_tokens,
        sentence_splitter=newline_sentence_splitter,
        min_chunk_sentences=min_chunk_sents,
        token_counter=_default_token_counter,
    )



# ============================================================
# 9) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        type=str,
        default="arxiv_json",
        choices=["mine", "arxiv_json"],
        help="Data source to evaluate.",
    )

    # arxiv json path
    ap.add_argument("--arxiv_json_path", type=str, default="arxiv.json")

    # txt path
    ap.add_argument("--txt_path", type=str, default="mine.txt")

    # Embedding model
    ap.add_argument("--hf_embed", type=str, default="BAAI/bge-base-en-v1.5")

    # Fixed
    ap.add_argument("--fixed_sents_per_chunk", type=int, default=6)

    # Semantic
    ap.add_argument("--semantic_buffer", type=int, default=1)
    ap.add_argument("--semantic_percentile", type=int, default=95)

    # Shift
    ap.add_argument("--shift_buffer", type=int, default=2)
    ap.add_argument("--shift_percentile", type=int, default=95)
    ap.add_argument("--shift_max_tokens", type=int, default=256)
    ap.add_argument("--shift_min_chunk_sents", type=int, default=1)

    ap.add_argument("--cache_dir", type=str, default=os.path.join(os.path.expanduser("~"), ".cache", "semantic_shift"))
    args = ap.parse_args()

    # ==========================
    # output JSON to ./result
    # ==========================
    result_dir = os.path.join(".", "result")
    os.makedirs(result_dir, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(result_dir, f"{args.dataset}_{run_id}.json")

    # ---- load docs ----
    if args.dataset == "arxiv_json":
        docs_raw = load_paragraph_list_json(args.arxiv_json_path, doc_id="arxiv_json")
    else:  # mine
        docs_raw = load_txt(args.txt_path, doc_id="mine")

    # ---- build splitters ----
    embed_model = HuggingFaceEmbedding(model_name=args.hf_embed)

    fixed_splitter = build_fixed_splitter(args.fixed_sents_per_chunk)
    semantic_splitter = build_semantic_splitter(embed_model, args.semantic_buffer, args.semantic_percentile)
    shift_splitter = build_shift_splitter(
        embed_model,
        buffer_size=args.shift_buffer,
        percentile=args.shift_percentile,
        max_tokens=args.shift_max_tokens,
        min_chunk_sents=args.shift_min_chunk_sents,
    )

    splitters = {
        "fixed": fixed_splitter,
        "semantic": semantic_splitter,
        "shift": shift_splitter,
    }

    # ---- segmentation metrics sums ----
    sums = {k: {"p": 0.0, "r": 0.0, "f1": 0.0, "pk": 0.0, "wd": 0.0} for k in splitters}

    # ---- chunk statistics containers ----
    chunk_counts_per_doc: Dict[str, List[int]] = {k: [] for k in splitters}
    sent_counts_all_chunks: Dict[str, List[int]] = {k: [] for k in splitters}
    within_doc_sent_var: Dict[str, List[float]] = {k: [] for k in splitters}

    # NEW: warnings/errors collected (instead of print)
    warnings: List[Dict[str, Any]] = []

    cnt = 0

    for ex in tqdm(docs_raw, desc=f"Evaluating ({args.dataset})"):
        sentences = ex["sentences"]
        labels = ex["labels"]
        n = len(sentences)
        if n <= 2:
            warnings.append(
                {"doc_id": ex.get("id", None), "splitter": None, "type": "skip_short_doc", "n_sentences": n}
            )
            continue

        text = build_text_from_sentences(sentences)
        doc = Document(text=text, metadata={"doc_id": ex["id"]})

        y_gold = labels_to_gold_boundaries(labels)

        ok = True
        for name, sp in splitters.items():
            try:
                nodes = sp.get_nodes_from_documents([doc], show_progress=False)
                chunks = [get_node_text(nd) for nd in nodes]

                # chunk stats
                counts = [count_lines_as_sentences(c) for c in chunks]
                total = sum(counts)
                if total != n:
                    raise ValueError(f"[{name}] Sentence count mismatch: chunks sum={total}, gold={n}")

                chunk_counts_per_doc[name].append(len(counts))
                sent_counts_all_chunks[name].extend(counts)

                if len(counts) >= 2:
                    within_doc_sent_var[name].append(float(np.var(np.array(counts, dtype=np.float64))))
                else:
                    within_doc_sent_var[name].append(0.0)

                # metrics
                y_pred = chunks_to_pred_boundaries(chunks, n)
                p, r, f1 = boundary_prf(y_pred, y_gold)
                pk = pk_metric(y_pred, y_gold)
                wd = windowdiff_metric(y_pred, y_gold)

                sums[name]["p"] += p
                sums[name]["r"] += r
                sums[name]["f1"] += f1
                sums[name]["pk"] += pk
                sums[name]["wd"] += wd
            except Exception as e:
                ok = False
                warnings.append(
                    {
                        "doc_id": ex.get("id", None),
                        "splitter": name,
                        "type": "splitter_failed",
                        "error": repr(e),
                    }
                )
                break

        if ok:
            cnt += 1

    # ==========================
    # NEW: compute final results + write JSON
    # ==========================
    output: Dict[str, Any] = {
        "run_id": run_id,
        "dataset": args.dataset,
        "args": vars(args),
        "docs_evaluated": cnt,
        "warnings": warnings,
        "results": {},
    }

    if cnt == 0:
        output["note"] = "No documents evaluated successfully. Please check dataset path / parsing."
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return

    # segmentation metrics
    for name in splitters:
        output["results"][name] = {
            "metrics": {
                "P": sums[name]["p"] / cnt,
                "R": sums[name]["r"] / cnt,
                "F1": sums[name]["f1"] / cnt,
                "Pk": sums[name]["pk"] / cnt,
                "WD": sums[name]["wd"] / cnt,
            },
            "chunk_stats": {},
        }

    # chunk statistics
    for name in splitters:
        per_doc_chunks = np.array(chunk_counts_per_doc[name], dtype=np.float64)
        all_chunk_sents = np.array(sent_counts_all_chunks[name], dtype=np.float64)
        within_doc_vars = np.array(within_doc_sent_var[name], dtype=np.float64)

        avg_chunks = float(per_doc_chunks.mean()) if len(per_doc_chunks) else 0.0
        var_chunks = float(per_doc_chunks.var()) if len(per_doc_chunks) else 0.0

        avg_sents = float(all_chunk_sents.mean()) if len(all_chunk_sents) else 0.0
        var_sents = float(all_chunk_sents.var()) if len(all_chunk_sents) else 0.0

        avg_within_var = float(within_doc_vars.mean()) if len(within_doc_vars) else 0.0
        var_within_var = float(within_doc_vars.var()) if len(within_doc_vars) else 0.0

        output["results"][name]["chunk_stats"] = {
            "avg_chunks_per_doc": avg_chunks,
            "var_chunks_per_doc": var_chunks,
            "avg_sents_per_chunk": avg_sents,
            "var_sents_per_chunk": var_sents,
            "avg_within_doc_var_sents_per_chunk": avg_within_var,
            "var_of_within_doc_var": var_within_var,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()