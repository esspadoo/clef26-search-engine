import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import torch
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_BGE_M3_MAX_LENGTH = 1024
DEFAULT_RETRIEVAL_MODE = "auto"
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_MULTIVECTOR_MAX_LENGTH = 256
DEFAULT_DENSE_WEIGHT = 0.4
DEFAULT_SPARSE_WEIGHT = 0.2
DEFAULT_COLBERT_WEIGHT = 0.4

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]
REPO_ROOT = SCRIPT_PATH.parents[7]

DEFAULT_QUERIES = CODE_ROOT / "data" / "expanded_queries_bge_large.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "bi_encoder_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode queries and corpus with a bi-encoder and run dense retrieval."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "dense", "hybrid", "multivector"),
        default=DEFAULT_RETRIEVAL_MODE,
        help=(
            "Retrieval strategy. "
            '"auto" uses "hybrid" for BAAI/bge-m3 and "dense" otherwise. '
            '"multivector" is accepted as alias of "hybrid".'
        ),
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=DEFAULT_CANDIDATE_MULTIPLIER,
        help=(
            "For hybrid mode: retrieve candidate_multiplier * top_k with dense FAISS "
            "before combining dense+sparse+colbert scores."
        ),
    )
    parser.add_argument(
        "--multivector-max-length",
        "--hybrid-max-length",
        dest="multivector_max_length",
        type=int,
        default=DEFAULT_MULTIVECTOR_MAX_LENGTH,
        help="Max token length used for sparse+colbert scoring in hybrid mode.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help="Dense score weight in hybrid mode.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=DEFAULT_SPARSE_WEIGHT,
        help="Sparse lexical score weight in hybrid mode.",
    )
    parser.add_argument(
        "--colbert-weight",
        type=float,
        default=DEFAULT_COLBERT_WEIGHT,
        help="Multi-vector ColBERT score weight in hybrid mode.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Optional maximum sequence length for the encoder. "
            "If not provided and model is BAAI/bge-m3, defaults to 1024."
        ),
    )
    parser.add_argument("--query-field", default="auto")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional debug limit. If set, only the first N queries are processed.",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="Optional prefix added to every query before encoding.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_retrieval_mode(model_name: str, requested_mode: str) -> str:
    if requested_mode == "multivector":
        return "hybrid"
    if requested_mode != "auto":
        return requested_mode
    if model_name.lower().startswith("baai/bge-m3"):
        return "hybrid"
    return "dense"


def resolve_execution_mode() -> tuple[str, list[str]]:
    """
    Automatically select execution mode based on visible CUDA devices:
    - 0 GPUs -> CPU
    - 1 GPU  -> single-GPU
    - 2+ GPUs -> multi-GPU
    """
    if not torch.cuda.is_available():
        return "cpu", []

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return "cuda", []

    devices = [f"cuda:{gpu_idx}" for gpu_idx in range(gpu_count)]
    return "multi-gpu", devices


def build_bge_m3_model(model_name: str, execution_mode: str, target_devices: list[str]) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        raise ImportError(
            "FlagEmbedding is required for bge-m3 retrieval. "
            "Install it with: pip install -U FlagEmbedding"
        ) from error

    init_kwargs: dict[str, Any] = {"use_fp16": execution_mode != "cpu"}
    if execution_mode == "cpu":
        init_kwargs["device"] = "cpu"
    elif execution_mode == "cuda":
        init_kwargs["device"] = "cuda:0"
    else:
        init_kwargs["devices"] = target_devices

    try:
        return BGEM3FlagModel(model_name, **init_kwargs)
    except TypeError:
        if "devices" in init_kwargs:
            fallback_device = target_devices[0] if target_devices else "cuda:0"
            print(
                "Installed FlagEmbedding does not support `devices=[...]`; "
                f"falling back to single device {fallback_device}."
            )
            init_kwargs.pop("devices")
            init_kwargs["device"] = fallback_device
        return BGEM3FlagModel(model_name, **init_kwargs)


def infer_query_prefix(model_name: str, provided_prefix: str | None) -> str:
    if provided_prefix is not None:
        return provided_prefix
    if model_name.lower().startswith("baai/bge-m3"):
        return ""
    if model_name.lower().startswith("baai/bge"):
        return "Represent this sentence for searching relevant passages: "
    return ""


def resolve_effective_max_length(model_name: str, provided_max_length: int | None) -> int | None:
    if provided_max_length is not None:
        if provided_max_length <= 0:
            raise ValueError("--max-length must be greater than 0 when provided.")
        return provided_max_length

    if model_name.lower().startswith("baai/bge-m3"):
        return DEFAULT_BGE_M3_MAX_LENGTH

    return None


def is_cuda_oom(error: Exception) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def score_to_float(score: Any) -> float:
    if hasattr(score, "item"):
        return float(score.item())
    return float(score)


def encode_bge_m3(
    model: Any,
    texts: list[str],
    batch_size: int,
    label: str,
    max_length: int | None,
    return_dense: bool,
    return_sparse: bool,
    return_colbert_vecs: bool,
) -> dict[str, Any]:
    current_batch_size = batch_size
    while True:
        print(
            f"Encoding {label} ({len(texts)} items) with bge-m3, "
            f"batch_size={current_batch_size}..."
        )
        try:
            output = model.encode(
                texts,
                batch_size=current_batch_size,
                max_length=max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert_vecs,
            )
            if return_dense:
                dense = np.asarray(output["dense_vecs"], dtype=np.float32)
                faiss.normalize_L2(dense)
                output["dense_vecs"] = dense
            return output
        except Exception as error:
            if not (torch.cuda.is_available() and is_cuda_oom(error) and current_batch_size > 1):
                raise

            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while bge-m3 encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def pick_query_text(item: dict[str, Any], query_field: str) -> str:
    if query_field != "auto":
        value = item.get(query_field, "")
        return value.strip() if isinstance(value, str) else ""

    for field_name in ("expanded", "original", "text", "query"):
        value = item.get(field_name, "")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


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


def load_queries(
    path: Path,
    query_field: str,
    max_queries: int | None,
    query_prefix: str,
) -> tuple[list[str], list[str]]:
    raw_queries = load_json(path)

    query_ids: list[str] = []
    query_texts: list[str] = []
    skipped = 0

    for qid, item in iter_query_records(raw_queries):
        text = pick_query_text(item, query_field)
        if not text:
            skipped += 1
            continue

        query_ids.append(qid)
        query_texts.append(f"{query_prefix}{text}" if query_prefix else text)

        if max_queries is not None and len(query_ids) >= max_queries:
            break

    print(f"Loaded {len(query_ids)} queries from {path}")
    if skipped:
        print(f"Skipped {skipped} queries with empty text")

    if not query_ids:
        raise ValueError("No valid queries were loaded.")

    return query_ids, query_texts


def build_document_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("title", "")).strip(),
        str(item.get("abstract", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    raw_corpus = load_json(path)
    if not isinstance(raw_corpus, list):
        raise ValueError("Corpus JSON must be a list of documents.")

    pubkeys: list[str] = []
    documents: list[str] = []
    seen_pubkeys: set[str] = set()
    skipped = 0

    for item in raw_corpus:
        if not isinstance(item, dict):
            skipped += 1
            continue

        pubkey = item.get("pubkey", item.get("id"))
        if pubkey is None:
            skipped += 1
            continue

        pubkey_str = str(pubkey)
        if pubkey_str in seen_pubkeys:
            continue

        text = build_document_text(item)
        if not text:
            skipped += 1
            continue

        seen_pubkeys.add(pubkey_str)
        pubkeys.append(pubkey_str)
        documents.append(text)

    print(f"Loaded {len(pubkeys)} documents from {path}")
    if skipped:
        print(f"Skipped {skipped} corpus entries without usable text or id")

    if not pubkeys:
        raise ValueError("No valid corpus documents were loaded.")

    return pubkeys, documents


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    label: str,
) -> np.ndarray:
    current_batch_size = batch_size
    while True:
        print(f"Encoding {label} ({len(texts)} items), batch_size={current_batch_size}...")
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as error:
            if not (torch.cuda.is_available() and is_cuda_oom(error) and current_batch_size > 1):
                raise

            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def encode_texts_multi_gpu(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    label: str,
    pool: Any,
) -> np.ndarray:
    current_batch_size = batch_size
    while True:
        print(
            f"Encoding {label} ({len(texts)} items) with multi-GPU, "
            f"batch_size={current_batch_size}..."
        )
        try:
            embeddings = model.encode_multi_process(
                texts,
                pool=pool,
                batch_size=current_batch_size,
            )
            normalized = np.asarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(normalized)
            return normalized
        except Exception as error:
            if not (is_cuda_oom(error) and current_batch_size > 1):
                raise

            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while multi-GPU encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def build_faiss_index(corpus_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings)
    return index


def dense_search(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_pubkeys: list[str],
    index: faiss.IndexFlatIP,
    top_k: int,
) -> dict[str, list[str]]:
    effective_top_k = min(top_k, len(corpus_pubkeys))
    _, indices = index.search(query_embeddings, effective_top_k)

    results: dict[str, list[str]] = {}
    for row, qid in enumerate(query_ids):
        ranked_pubkeys = [
            corpus_pubkeys[col]
            for col in indices[row]
            if 0 <= col < len(corpus_pubkeys)
        ]
        results[qid] = ranked_pubkeys

    print(f"Completed dense search for {len(query_ids)} queries")
    return results


def hybrid_search(
    model: Any,
    query_ids: list[str],
    query_texts: list[str],
    corpus_pubkeys: list[str],
    corpus_texts: list[str],
    index: faiss.IndexFlatIP,
    top_k: int,
    candidate_multiplier: int,
    batch_size: int,
    hybrid_max_length: int,
    dense_weight: float,
    sparse_weight: float,
    colbert_weight: float,
) -> dict[str, list[str]]:
    if candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be greater than 0.")
    if hybrid_max_length <= 0:
        raise ValueError("--multivector-max-length/--hybrid-max-length must be greater than 0.")
    if dense_weight == 0.0 and sparse_weight == 0.0 and colbert_weight == 0.0:
        raise ValueError(
            "At least one hybrid weight must be non-zero "
            "(--dense-weight, --sparse-weight, --colbert-weight)."
        )

    candidate_k = min(
        len(corpus_pubkeys),
        max(top_k, top_k * candidate_multiplier),
    )

    query_output = encode_bge_m3(
        model=model,
        texts=query_texts,
        batch_size=batch_size,
        label="queries (hybrid)",
        max_length=hybrid_max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    query_embeddings = np.asarray(query_output["dense_vecs"], dtype=np.float32)
    query_sparse = query_output["lexical_weights"]
    query_colbert = query_output["colbert_vecs"]
    dense_scores, dense_indices = index.search(query_embeddings, candidate_k)

    results: dict[str, list[str]] = {}
    for row, qid in enumerate(query_ids):
        candidate_doc_indices = [
            int(doc_idx)
            for doc_idx in dense_indices[row]
            if 0 <= int(doc_idx) < len(corpus_pubkeys)
        ]

        if not candidate_doc_indices:
            results[qid] = []
            continue

        candidate_texts = [corpus_texts[doc_idx] for doc_idx in candidate_doc_indices]
        candidate_output = encode_bge_m3(
            model=model,
            texts=candidate_texts,
            batch_size=batch_size,
            label=f"candidates for query {qid} (hybrid)",
            max_length=hybrid_max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        candidate_sparse = candidate_output["lexical_weights"]
        candidate_colbert = candidate_output["colbert_vecs"]

        scored_candidates: list[tuple[str, float]] = []
        q_sparse = query_sparse[row]
        q_colbert = query_colbert[row]
        for local_idx, doc_colbert in enumerate(candidate_colbert):
            d_sparse = candidate_sparse[local_idx]
            sparse_score = score_to_float(model.compute_lexical_matching_score(q_sparse, d_sparse))
            colbert_score = score_to_float(model.colbert_score(q_colbert, doc_colbert))
            dense_score = score_to_float(dense_scores[row][local_idx])
            final_score = (
                dense_weight * dense_score
                + sparse_weight * sparse_score
                + colbert_weight * colbert_score
            )
            scored_candidates.append(
                (corpus_pubkeys[candidate_doc_indices[local_idx]], final_score)
            )

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        results[qid] = [pubkey for pubkey, _ in scored_candidates[:top_k]]

    print(f"Completed hybrid search for {len(query_ids)} queries")
    return results


def save_results(path: Path, results: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} query rankings to {path}")


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be greater than 0.")
    if args.multivector_max_length <= 0:
        raise ValueError("--multivector-max-length/--hybrid-max-length must be greater than 0.")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be greater than 0 when provided.")

    query_prefix = infer_query_prefix(args.model, args.query_prefix)
    effective_max_length = resolve_effective_max_length(args.model, args.max_length)
    retrieval_mode = resolve_retrieval_mode(args.model, args.retrieval_mode)
    execution_mode, target_devices = resolve_execution_mode()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Detected GPUs: {gpu_count}")
    if execution_mode == "cpu":
        print("Execution mode: CPU (no CUDA GPU detected)")
    elif execution_mode == "cuda":
        print("Execution mode: single-GPU")
        print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Execution mode: multi-GPU ({', '.join(target_devices)})")

    print(f"Bi-encoder model: {args.model}")
    print(f"Retrieval mode: {retrieval_mode}")
    if query_prefix:
        print(f"Using query prefix: {query_prefix}")
    if effective_max_length is not None:
        print(f"Encoder max length: {effective_max_length}")

    query_ids, query_texts = load_queries(
        path=args.queries,
        query_field=args.query_field,
        max_queries=args.max_queries,
        query_prefix=query_prefix,
    )
    corpus_pubkeys, corpus_texts = load_corpus(args.corpus)

    if retrieval_mode == "hybrid" and not args.model.lower().startswith("baai/bge-m3"):
        raise ValueError(
            '--retrieval-mode "hybrid" requires BGE-M3 (e.g., --model BAAI/bge-m3).'
        )

    if args.model.lower().startswith("baai/bge-m3"):
        model = build_bge_m3_model(
            model_name=args.model,
            execution_mode=execution_mode,
            target_devices=target_devices,
        )
        corpus_dense_output = encode_bge_m3(
            model=model,
            texts=corpus_texts,
            batch_size=args.batch_size,
            label="corpus (dense index)",
            max_length=effective_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        corpus_embeddings = np.asarray(corpus_dense_output["dense_vecs"], dtype=np.float32)

        if retrieval_mode == "hybrid":
            print("Building FAISS index...")
            index = build_faiss_index(corpus_embeddings)
            results = hybrid_search(
                model=model,
                query_ids=query_ids,
                query_texts=query_texts,
                corpus_pubkeys=corpus_pubkeys,
                corpus_texts=corpus_texts,
                index=index,
                top_k=args.top_k,
                candidate_multiplier=args.candidate_multiplier,
                batch_size=args.batch_size,
                hybrid_max_length=args.multivector_max_length,
                dense_weight=args.dense_weight,
                sparse_weight=args.sparse_weight,
                colbert_weight=args.colbert_weight,
            )
            save_results(args.output, results)
            return

        query_dense_output = encode_bge_m3(
            model=model,
            texts=query_texts,
            batch_size=args.batch_size,
            label="queries (dense)",
            max_length=effective_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_embeddings = np.asarray(query_dense_output["dense_vecs"], dtype=np.float32)
    else:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for dense retrieval mode. "
                "Install it with: pip install -U sentence-transformers"
            )
        if execution_mode == "multi-gpu":
            model = SentenceTransformer(args.model)
            if effective_max_length is not None:
                model.max_seq_length = effective_max_length
            pool = model.start_multi_process_pool(target_devices=target_devices)
            try:
                corpus_embeddings = encode_texts_multi_gpu(
                    model=model,
                    texts=corpus_texts,
                    batch_size=args.batch_size,
                    label="corpus",
                    pool=pool,
                )
                query_embeddings = encode_texts_multi_gpu(
                    model=model,
                    texts=query_texts,
                    batch_size=args.batch_size,
                    label="queries",
                    pool=pool,
                )
            finally:
                model.stop_multi_process_pool(pool)
        else:
            model = SentenceTransformer(args.model, device=execution_mode)
            if effective_max_length is not None:
                model.max_seq_length = effective_max_length
            corpus_embeddings = encode_texts(
                model=model,
                texts=corpus_texts,
                batch_size=args.batch_size,
                label="corpus",
            )
            query_embeddings = encode_texts(
                model=model,
                texts=query_texts,
                batch_size=args.batch_size,
                label="queries",
            )

    print("Building FAISS index...")
    index = build_faiss_index(corpus_embeddings)

    results = dense_search(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        corpus_pubkeys=corpus_pubkeys,
        index=index,
        top_k=args.top_k,
    )
    save_results(args.output, results)


if __name__ == "__main__":
    main()
