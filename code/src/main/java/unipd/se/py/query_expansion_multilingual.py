"""Build multilingual query expansions for sparse, dense, and ColBERT retrieval.

The script reads one or more raw query files, expands each query into three
retrieval-oriented views, and merges the results across languages into a single
JSON array. The canonical output contains:

    - original: merged original query text
    - sparse: lexical / BM25-oriented expansion
    - embedding: dense embedding-oriented expansion
    - colbert: late-interaction / ColBERT-oriented expansion

For backward compatibility the script also writes `expanded` as an alias of
`sparse`, so legacy consumers keep working while the new fields are adopted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import torch

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:  # pragma: no cover - handled at runtime
    BGEM3FlagModel = None


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]
REPO_ROOT = SCRIPT_PATH.parents[7]

DEFAULT_INPUT_GLOB = "*_train.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT = CODE_ROOT / "data" / "expanded_queries_multilingual_merged.json"
DEFAULT_COMPAT_OUTPUT = CODE_ROOT / "data" / "expanded_queries_bge_large.json"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 1024
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_COLBERT_SCORE_BATCH_SIZE = 256
DEFAULT_COLBERT_CORPUS_CACHE_SIZE = 0
DEFAULT_COLBERT_CORPUS_ENCODE_BATCH_SIZE = 8

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+(?:['\-][A-Za-zÀ-ÿ0-9]+)*")

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "so",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "we",
    "you",
    "your",
    "our",
    "they",
    "he",
    "she",
    "them",
    "his",
    "her",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
}


@dataclass(frozen=True)
class QuerySource:
    language: str
    path: Path
    index: str
    pubkey: str
    original: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate merged multilingual query expansions for sparse, dense, and ColBERT retrieval."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Raw query files to expand. If omitted, the script auto-discovers "
            f"`{DEFAULT_INPUT_GLOB}` files under `code/data/`."
        ),
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compat-output", type=Path, default=DEFAULT_COMPAT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-multiplier", type=int, default=DEFAULT_CANDIDATE_MULTIPLIER)
    parser.add_argument(
        "--colbert-score-batch-size",
        type=int,
        default=DEFAULT_COLBERT_SCORE_BATCH_SIZE,
    )
    parser.add_argument(
        "--colbert-corpus-cache-size",
        type=int,
        default=DEFAULT_COLBERT_CORPUS_CACHE_SIZE,
        help=(
            "Maximum number of corpus ColBERT vectors kept in RAM for reuse. "
            "Set to 0 to disable caching (lowest memory, slower runtime)."
        ),
    )
    parser.add_argument(
        "--colbert-corpus-encode-batch-size",
        type=int,
        default=DEFAULT_COLBERT_CORPUS_ENCODE_BATCH_SIZE,
        help="Batch size used when encoding on-demand corpus ColBERT candidates.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional debug limit; if set, only the first N queries per input file are processed.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def maybe_int(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def infer_language_from_path(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0].lower()
    return stem.lower()


def load_stopwords(language: str) -> set[str]:
    stopwords = set(COMMON_STOPWORDS)
    data_dir = CODE_ROOT / "data"
    candidates = sorted(data_dir.glob(f"stoplist_{language}_*.txt"))
    if not candidates and language == "en":
        candidates = sorted(data_dir.glob("stoplist_en_*.txt"))

    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as handle:
                stopwords.update(
                    normalize_text(line.strip().lower())
                    for line in handle
                    if line.strip()
                )
        except FileNotFoundError:
            continue

    return {token for token in stopwords if token}


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_RE.findall(text):
        token = normalize_text(raw_token.lower())
        token = token.lstrip("#@")
        if token.endswith("'s"):
            token = token[:-2]
        if len(token) < 3 or token in stopwords:
            continue
        tokens.append(token)
    return tokens


def dedupe_terms(terms: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for term in terms:
        normalized = normalize_text(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def merge_texts(parts: Iterable[str]) -> str:
    return normalize_text(" ".join(dedupe_terms(part for part in parts if part)))


def build_document_text(item: dict[str, Any]) -> str:
    title = normalize_text(str(item.get("title", "")).strip())
    abstract = normalize_text(str(item.get("abstract", "")).strip())
    return merge_texts((title, abstract))


def load_corpus(path: Path) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    raw_corpus = load_json(path)
    if not isinstance(raw_corpus, list):
        raise ValueError("Corpus JSON must be a list of documents.")

    pubkeys: list[str] = []
    texts: list[str] = []
    records: list[dict[str, Any]] = []
    seen_pubkeys: set[str] = set()

    for item in raw_corpus:
        if not isinstance(item, dict):
            continue

        pubkey = item.get("pubkey", item.get("id"))
        if pubkey is None:
            continue

        pubkey_str = str(pubkey)
        if pubkey_str in seen_pubkeys:
            continue

        text = build_document_text(item)
        if not text:
            continue

        seen_pubkeys.add(pubkey_str)
        pubkeys.append(pubkey_str)
        texts.append(text)
        records.append(
            {
                "pubkey": pubkey_str,
                "title": normalize_text(str(item.get("title", "")).strip()),
                "abstract": normalize_text(str(item.get("abstract", "")).strip()),
                "text": text,
            }
        )

    if not pubkeys:
        raise ValueError("No valid corpus documents were loaded.")

    return pubkeys, texts, records


def iter_query_records(raw_queries: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(raw_queries, list):
        for position, item in enumerate(raw_queries):
            if isinstance(item, dict):
                qid = str(item.get("index", item.get("qid", item.get("id", position))))
                yield qid, item
    elif isinstance(raw_queries, dict):
        for qid, item in raw_queries.items():
            if isinstance(item, dict):
                yield str(qid), item
            elif isinstance(item, str):
                yield str(qid), {"text": item}


def pick_source_text(item: dict[str, Any]) -> str:
    for field in ("original", "text", "query", "expanded", "sparse", "embedding", "colbert"):
        value = item.get(field, "")
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def load_query_sources(paths: list[Path], max_queries: int | None) -> list[QuerySource]:
    sources: list[QuerySource] = []
    for path in paths:
        raw = load_json(path)
        language = infer_language_from_path(path)
        skipped = 0
        for qid, item in iter_query_records(raw):
            if not isinstance(item, dict):
                skipped += 1
                continue

            text = pick_source_text(item)
            pubkey = item.get("pubkey")
            if not text or pubkey is None:
                skipped += 1
                continue

            source_language = str(item.get("language") or item.get("lang") or language).strip() or language
            sources.append(
                QuerySource(
                    language=source_language,
                    path=path,
                    index=str(qid),
                    pubkey=str(pubkey),
                    original=text,
                )
            )

            if max_queries is not None and len([src for src in sources if src.path == path]) >= max_queries:
                break

        print(f"Loaded {len([src for src in sources if src.path == path])} queries from {path}")
        if skipped:
            print(f"Skipped {skipped} records from {path}")

    if not sources:
        raise ValueError("No valid query sources were loaded.")

    return sources


def build_bge_m3_model(model_name: str) -> Any:
    if BGEM3FlagModel is None:
        raise ImportError(
            "FlagEmbedding is required for multilingual query expansion. "
            "Install it with: pip install -U FlagEmbedding"
        )

    use_fp16 = torch.cuda.is_available()
    try:
        return BGEM3FlagModel(model_name, use_fp16=use_fp16)
    except TypeError:
        return BGEM3FlagModel(model_name)


def encode_bge_m3(
    model: Any,
    texts: list[str],
    batch_size: int,
    label: str,
    max_length: int,
    return_dense: bool,
    return_sparse: bool,
    return_colbert_vecs: bool,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if max_length <= 0:
        raise ValueError("--max-length must be greater than 0.")

    current_batch_size = batch_size
    while True:
        try:
            total_items = len(texts)
            if total_items == 0:
                empty: dict[str, Any] = {}
                if return_dense:
                    empty["dense_vecs"] = np.empty((0, 0), dtype=np.float32)
                if return_sparse:
                    empty["lexical_weights"] = []
                if return_colbert_vecs:
                    empty["colbert_vecs"] = []
                return empty

            dense_chunks: list[np.ndarray] = []
            sparse_chunks: list[Any] = []
            colbert_chunks: list[Any] = []
            for start in range(0, total_items, current_batch_size):
                end = min(total_items, start + current_batch_size)
                chunk_output = model.encode(
                    texts[start:end],
                    batch_size=current_batch_size,
                    max_length=max_length,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert_vecs,
                )
                if return_dense:
                    dense = np.asarray(chunk_output["dense_vecs"], dtype=np.float32)
                    faiss.normalize_L2(dense)
                    dense_chunks.append(dense)
                if return_sparse:
                    sparse_chunks.extend(chunk_output["lexical_weights"])
                if return_colbert_vecs:
                    colbert_chunks.extend(chunk_output["colbert_vecs"])

            output: dict[str, Any] = {}
            if return_dense:
                output["dense_vecs"] = np.concatenate(dense_chunks, axis=0)
            if return_sparse:
                output["lexical_weights"] = sparse_chunks
            if return_colbert_vecs:
                output["colbert_vecs"] = colbert_chunks
            return output
        except Exception as error:
            if not (torch.cuda.is_available() and "out of memory" in str(error).lower() and current_batch_size > 1):
                raise
            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def compute_colbert_scores_batched(
    *,
    query_colbert: np.ndarray,
    candidate_colbert: list[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("--colbert-score-batch-size must be greater than 0.")
    if not candidate_colbert:
        return np.empty((0,), dtype=np.float32)

    def _score_on_device(device: torch.device, current_batch_size: int) -> np.ndarray:
        q_reps = torch.as_tensor(query_colbert, dtype=torch.float32, device=device)
        if q_reps.ndim != 2 or q_reps.shape[0] == 0:
            return np.zeros(len(candidate_colbert), dtype=np.float32)

        score_chunks: list[np.ndarray] = []
        for start in range(0, len(candidate_colbert), current_batch_size):
            batch = candidate_colbert[start : start + current_batch_size]
            batch_scores = np.zeros(len(batch), dtype=np.float32)
            non_empty_indices: list[int] = []
            non_empty_docs: list[np.ndarray] = []

            for local_idx, doc_colbert in enumerate(batch):
                if isinstance(doc_colbert, np.ndarray) and doc_colbert.ndim == 2 and doc_colbert.shape[0] > 0:
                    non_empty_indices.append(local_idx)
                    non_empty_docs.append(doc_colbert)

            if not non_empty_docs:
                score_chunks.append(batch_scores)
                continue

            max_doc_tokens = max(doc.shape[0] for doc in non_empty_docs)
            embedding_dim = q_reps.shape[1]
            doc_tensor = torch.zeros((len(non_empty_docs), max_doc_tokens, embedding_dim), dtype=torch.float32, device=device)
            doc_mask = torch.zeros((len(non_empty_docs), max_doc_tokens), dtype=torch.bool, device=device)

            for tensor_row, doc_colbert in enumerate(non_empty_docs):
                doc_reps = torch.as_tensor(doc_colbert, dtype=torch.float32, device=device)
                doc_length = doc_reps.shape[0]
                doc_tensor[tensor_row, :doc_length] = doc_reps
                doc_mask[tensor_row, :doc_length] = True

            token_scores = torch.einsum("qd,bkd->bqk", q_reps, doc_tensor)
            token_scores = token_scores.masked_fill(~doc_mask.unsqueeze(1), float("-inf"))
            max_scores = token_scores.max(dim=-1).values
            reduced_scores = max_scores.sum(dim=-1) / q_reps.shape[0]
            reduced_scores_np = reduced_scores.detach().cpu().numpy().astype(np.float32, copy=False)

            for output_idx, score in zip(non_empty_indices, reduced_scores_np):
                batch_scores[output_idx] = float(score)

            score_chunks.append(batch_scores)

        return np.concatenate(score_chunks, axis=0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    current_batch_size = batch_size

    while True:
        try:
            return _score_on_device(device=device, current_batch_size=current_batch_size)
        except Exception as error:
            is_cuda_oom = use_cuda and "out of memory" in str(error).lower()
            if not is_cuda_oom:
                raise

            if current_batch_size > 1:
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    f"CUDA OOM while ColBERT scoring with batch_size={current_batch_size}. "
                    f"Retrying with batch_size={next_batch_size}."
                )
                current_batch_size = next_batch_size
                torch.cuda.empty_cache()
                continue

            if device.type == "cuda":
                print("CUDA OOM while ColBERT scoring at batch_size=1. Falling back to CPU scoring.")
                device = torch.device("cpu")
                torch.cuda.empty_cache()
                continue

            raise


def encode_colbert_vectors(
    *,
    model: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    label: str,
) -> list[np.ndarray]:
    if not texts:
        return []

    output = encode_bge_m3(
        model=model,
        texts=texts,
        batch_size=batch_size,
        label=label,
        max_length=max_length,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )
    encoded = output["colbert_vecs"]
    return [np.asarray(v, dtype=np.float32) for v in encoded]


def build_faiss_index(corpus_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings)
    return index


def extract_terms(texts: Iterable[str], stopwords: set[str], max_terms: int) -> list[str]:
    if max_terms <= 0:
        return []

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text, stopwords))

    terms: list[str] = []
    for term, _ in counter.most_common(max_terms * 2):
        if term not in terms:
            terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def summarize_documents(records: list[dict[str, Any]], doc_indices: list[int], *, max_terms: int, stopwords: set[str]) -> str:
    snippets: list[str] = []
    for doc_index in doc_indices:
        if 0 <= doc_index < len(records):
            record = records[doc_index]
            title = record.get("title", "")
            abstract = record.get("abstract", "")
            snippets.append(title if title else record.get("text", ""))
            if abstract:
                snippets.append(abstract)
    return normalize_text(" ".join(extract_terms(snippets, stopwords, max_terms=max_terms)))


def build_representation(
    original: str,
    query_terms: list[str],
    doc_terms: list[str],
    *,
    prefix: str = "",
) -> str:
    parts = [original]
    if prefix:
        parts.insert(0, prefix)
    if query_terms:
        parts.append(" ".join(query_terms))
    if doc_terms:
        parts.append(" ".join(doc_terms))
    return merge_texts(parts)


def build_query_variants_for_language(
    *,
    source_queries: list[QuerySource],
    model: Any,
    corpus_pubkeys: list[str],
    corpus_records: list[dict[str, Any]],
    dense_index: faiss.IndexFlatIP,
    corpus_sparse_cache: list[dict[str, float]],
    batch_size: int,
    max_length: int,
    top_k: int,
    candidate_multiplier: int,
    colbert_score_batch_size: int,
    colbert_corpus_cache_size: int,
    colbert_corpus_encode_batch_size: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be greater than 0.")
    if colbert_corpus_cache_size < 0:
        raise ValueError("--colbert-corpus-cache-size must be greater than or equal to 0.")
    if colbert_corpus_encode_batch_size <= 0:
        raise ValueError("--colbert-corpus-encode-batch-size must be greater than 0.")

    stopwords_by_language: dict[str, set[str]] = defaultdict(set)
    for source in source_queries:
        stopwords_by_language[source.language] = load_stopwords(source.language)

    query_texts = [source.original for source in source_queries]
    query_output = encode_bge_m3(
        model=model,
        texts=query_texts,
        batch_size=batch_size,
        label="queries",
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    query_dense = np.asarray(query_output["dense_vecs"], dtype=np.float32)
    query_sparse = query_output["lexical_weights"]
    query_colbert = query_output["colbert_vecs"]

    dense_candidate_k = min(len(corpus_pubkeys), max(top_k, top_k * candidate_multiplier))
    _, dense_indices = dense_index.search(query_dense, dense_candidate_k)

    merged_rows: dict[tuple[str, str], dict[str, Any]] = {}
    colbert_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    for row, source in enumerate(source_queries):
        stopwords = stopwords_by_language[source.language]
        query_terms = extract_terms([source.original], stopwords, max_terms=8)

        sparse_scores = np.asarray(
            model.compute_lexical_matching_score([query_sparse[row]], corpus_sparse_cache),
            dtype=np.float32,
        ).reshape(-1)
        sparse_ranked = np.argsort(-sparse_scores)[:top_k]

        dense_ranked = [int(idx) for idx in dense_indices[row][:top_k] if 0 <= int(idx) < len(corpus_pubkeys)]

        colbert_candidate_indices = [int(idx) for idx in dense_indices[row][:dense_candidate_k] if 0 <= int(idx) < len(corpus_pubkeys)]
        if colbert_corpus_cache_size > 0:
            missing_indices = [idx for idx in colbert_candidate_indices if idx not in colbert_cache]
            if missing_indices:
                missing_texts = [corpus_records[idx]["text"] for idx in missing_indices]
                missing_vectors = encode_colbert_vectors(
                    model=model,
                    texts=missing_texts,
                    batch_size=colbert_corpus_encode_batch_size,
                    max_length=max_length,
                    label="colbert-corpus-candidates",
                )
                for idx, vector in zip(missing_indices, missing_vectors):
                    colbert_cache[idx] = vector
                    colbert_cache.move_to_end(idx)
                    while len(colbert_cache) > colbert_corpus_cache_size:
                        colbert_cache.popitem(last=False)

            colbert_candidates: list[np.ndarray] = []
            for idx in colbert_candidate_indices:
                vector = colbert_cache.get(idx)
                if vector is None:
                    continue
                colbert_cache.move_to_end(idx)
                colbert_candidates.append(vector)

            # Keep score alignment with candidate indices even if transient vectors were evicted.
            if len(colbert_candidates) != len(colbert_candidate_indices):
                refreshed_vectors = encode_colbert_vectors(
                    model=model,
                    texts=[corpus_records[idx]["text"] for idx in colbert_candidate_indices],
                    batch_size=colbert_corpus_encode_batch_size,
                    max_length=max_length,
                    label="colbert-corpus-candidates-refresh",
                )
                colbert_candidates = refreshed_vectors
                for idx, vector in zip(colbert_candidate_indices, refreshed_vectors):
                    colbert_cache[idx] = vector
                    colbert_cache.move_to_end(idx)
                    while len(colbert_cache) > colbert_corpus_cache_size:
                        colbert_cache.popitem(last=False)
        else:
            colbert_candidates = encode_colbert_vectors(
                model=model,
                texts=[corpus_records[idx]["text"] for idx in colbert_candidate_indices],
                batch_size=colbert_corpus_encode_batch_size,
                max_length=max_length,
                label="colbert-corpus-candidates",
            )

        colbert_scores = compute_colbert_scores_batched(
            query_colbert=query_colbert[row],
            candidate_colbert=colbert_candidates,
            batch_size=colbert_score_batch_size,
        )
        colbert_ranked = [colbert_candidate_indices[pos] for pos in np.argsort(-colbert_scores)[:top_k]]

        sparse_terms = extract_terms((corpus_records[idx]["text"] for idx in sparse_ranked), stopwords, max_terms=18)
        dense_terms = extract_terms((corpus_records[idx]["title"] or corpus_records[idx]["text"] for idx in dense_ranked), stopwords, max_terms=12)
        colbert_terms = extract_terms((corpus_records[idx]["text"] for idx in colbert_ranked), stopwords, max_terms=24)

        sparse_text = build_representation(source.original, query_terms, sparse_terms)
        embedding_text = build_representation(source.original, query_terms[:4], dense_terms, prefix="")
        colbert_text = build_representation(source.original, query_terms, colbert_terms)

        key = (source.index, source.pubkey)
        record = merged_rows.setdefault(
            key,
            {
                "index": maybe_int(source.index),
                "pubkey": maybe_int(source.pubkey),
                "languages": [],
                "sources": [],
                "_original_parts": [],
                "_keywords_parts": [],
                "_sparse_parts": [],
                "_embedding_parts": [],
                "_colbert_parts": [],
            },
        )

        record["languages"].append(source.language)
        record["sources"].append(
            {
                "language": source.language,
                "index": maybe_int(source.index),
                "pubkey": maybe_int(source.pubkey),
                "original": source.original,
                "keywords": " ".join(query_terms),
                "sparse": sparse_text,
                "embedding": embedding_text,
                "colbert": colbert_text,
                "expanded": sparse_text,
            }
        )
        record["_original_parts"].append(source.original)
        record["_keywords_parts"].append(" ".join(query_terms))
        record["_sparse_parts"].append(sparse_text)
        record["_embedding_parts"].append(embedding_text)
        record["_colbert_parts"].append(colbert_text)

    final_rows: list[dict[str, Any]] = []
    for record in merged_rows.values():
        languages = dedupe_terms(record["languages"])
        original = merge_texts(record.pop("_original_parts"))
        keywords = merge_texts(record.pop("_keywords_parts"))
        sparse = merge_texts(record.pop("_sparse_parts"))
        embedding = merge_texts(record.pop("_embedding_parts"))
        colbert = merge_texts(record.pop("_colbert_parts"))

        record["language"] = languages[0] if len(languages) == 1 else "merged"
        record["languages"] = languages
        record["original"] = original
        record["keywords"] = keywords
        record["sparse"] = sparse
        record["embedding"] = embedding
        record["colbert"] = colbert
        record["expanded"] = sparse

        final_rows.append(record)

    final_rows.sort(key=lambda item: (str(item["index"]), str(item["pubkey"])))
    return final_rows


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be greater than 0.")
    if args.colbert_score_batch_size <= 0:
        raise ValueError("--colbert-score-batch-size must be greater than 0.")
    if args.colbert_corpus_cache_size < 0:
        raise ValueError("--colbert-corpus-cache-size must be greater than or equal to 0.")
    if args.colbert_corpus_encode_batch_size <= 0:
        raise ValueError("--colbert-corpus-encode-batch-size must be greater than 0.")

    if args.inputs:
        input_paths = args.inputs
    else:
        input_paths = sorted(CODE_ROOT.joinpath("data").glob(DEFAULT_INPUT_GLOB))
        input_paths = [path for path in input_paths if path.name not in {args.output.name, args.compat_output.name}]

    if not input_paths:
        raise ValueError("No input query files were found.")

    corpus_pubkeys, corpus_texts, corpus_records = load_corpus(args.corpus)
    model = build_bge_m3_model(args.model)

    corpus_output = encode_bge_m3(
        model=model,
        texts=corpus_texts,
        batch_size=args.batch_size,
        label="corpus",
        max_length=args.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    corpus_embeddings = np.asarray(corpus_output["dense_vecs"], dtype=np.float32)
    corpus_sparse_cache = corpus_output["lexical_weights"]
    dense_index = build_faiss_index(corpus_embeddings)

    query_sources = load_query_sources(input_paths, args.max_queries)

    merged_rows = build_query_variants_for_language(
        source_queries=query_sources,
        model=model,
        corpus_pubkeys=corpus_pubkeys,
        corpus_records=corpus_records,
        dense_index=dense_index,
        corpus_sparse_cache=corpus_sparse_cache,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
        colbert_score_batch_size=args.colbert_score_batch_size,
        colbert_corpus_cache_size=args.colbert_corpus_cache_size,
        colbert_corpus_encode_batch_size=args.colbert_corpus_encode_batch_size,
    )

    write_json(args.output, merged_rows)
    if args.compat_output != args.output:
        write_json(args.compat_output, merged_rows)

    print(f"Saved {len(merged_rows)} merged query records to {args.output}")
    if args.compat_output != args.output:
        print(f"Saved compatibility copy to {args.compat_output}")


if __name__ == "__main__":
    main()

