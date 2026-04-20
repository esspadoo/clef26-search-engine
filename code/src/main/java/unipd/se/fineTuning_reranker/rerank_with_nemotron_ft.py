"""
rerank_with_nemotron_ft.py

Inference con llama-nemotron-rerank-1b-v2 fine-tuned.

IMPORTANTE:
  - Nemotron NON usa prefissi né template speciali sulle query.
    Solo clean_text() — identico al training.
  - trust_remote_code=True è OBBLIGATORIO per caricare il modello
    (architettura bidirectional attention non standard).
  - I logits di Nemotron sono negativi (la testa di classificazione
    produce logit non normalizzati): usare il logit raw è corretto
    per il ranking (ordine preservato), ma se vuoi score in [0,1]
    applica torch.sigmoid(). Qui usiamo il logit raw per il ranking.

Uso:
  python rerank_with_nemotron_ft.py \
    --model_dir    models/nemotron-rerank-1b-retrix/final \
    --nemotron_run reranked_results_nemotron_topk400.json \
    --collection   collection_data.json \
    --queries      en_train.json \
    --output       reranked_results_nemotron_ft_top100.json \
    --rerank_depth 100 \
    --batch_size   64
"""

import json
import logging
import re
import gc
import argparse
import unicodedata
from pathlib import Path

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Preprocessing ──────────────────────────────────────────────────────────────
# IDENTICO a prepare_data_nemotron_finetune.py — nessun prefisso.

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&amp;",  "&",  text)
    text = re.sub(r"&lt;",   "<",  text)
    text = re.sub(r"&gt;",   ">",  text)
    text = re.sub(r"&quot;", '"',  text)
    text = re.sub(r"&#39;",  "'",  text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("So", "Cs", "Sk")
    )
    return re.sub(r"\s+", " ", text).strip()


def doc_text(item: dict, max_chars: int) -> str:
    t = (item.get("title")    or "").strip()
    a = (item.get("abstract") or "").strip()
    return f"{t}. {a}".strip(". ")[:max_chars]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_collection(path: str, max_chars: int) -> dict[str, str]:
    """Autodetect pubkey (2026) vs cord_uid (2025)."""
    data   = json.load(open(path, encoding="utf-8"))
    corpus = {}
    for item in data:
        if "pubkey" in item and item["pubkey"] is not None:
            key = str(item["pubkey"])
        elif "cord_uid" in item and item["cord_uid"] is not None:
            key = str(item["cord_uid"])
        else:
            continue
        text = doc_text(item, max_chars)
        if text:
            corpus[key] = text
    log.info(f"Corpus: {len(corpus):,} documenti  ← {path}")
    return corpus


def load_queries(path: str) -> dict[str, str]:
    """Solo clean_text — nessun prefisso."""
    data    = json.load(open(path, encoding="utf-8"))
    queries = {}
    for item in data:
        qid          = str(item["index"])
        queries[qid] = clean_text(item["text"])
    log.info(f"Queries: {len(queries):,}  ← {path}")
    return queries


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    raw = json.load(open(path, encoding="utf-8"))
    run = {str(k): [str(d) for d in v] for k, v in raw.items()}
    log.info(f"Nemotron run: {len(run):,} queries  ← {path}")
    return run


# ── Scoring ────────────────────────────────────────────────────────────────────

def predict_with_oom_fallback(
        model:       CrossEncoder,
        pairs:       list[list[str]],
        batch_size:  int,
        use_sigmoid: bool = True,
) -> list[float]:
    """
    Scoring con OOM fallback.

    Nemotron produce logit raw negativi (es. -8.3, -6.1).
    use_sigmoid=True applica torch.sigmoid() per portare gli score
    in [0, 1] — l'ordine relativo è preservato, ma i valori diventano
    interpretabili come probabilità di rilevanza.
    use_sigmoid=False usa i logit raw (ugualmente validi per il ranking).
    """
    current_bs = batch_size
    while current_bs >= 1:
        try:
            scores = model.predict(
                pairs,
                batch_size=current_bs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if isinstance(scores, (float, int)):
                raw = torch.tensor([float(scores)])
            else:
                raw = torch.tensor(scores, dtype=torch.float32)

            if use_sigmoid:
                raw = torch.sigmoid(raw)

            return raw.tolist()

        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM con batch_size={current_bs}, provo {current_bs // 2}...")
            torch.cuda.empty_cache()
            gc.collect()
            current_bs //= 2

    log.error("OOM anche con batch_size=1. Restituisco 0.5 per tutti.")
    return [0.5] * len(pairs)


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(results: dict, nemotron_run: dict, rerank_depth: int):
    if not results:
        log.error("SANITY CHECK FAILED: output vuoto!")
        return
    coverage = len(results) / max(1, len(nemotron_run))
    avg_len  = sum(len(v) for v in results.values()) / len(results)
    log.info(f"Sanity check: {len(results):,} queries ({coverage:.1%} coverage), "
             f"avg {avg_len:.1f} doc/query (atteso ≤{rerank_depth})")
    sample_qid = next(iter(results))
    log.info(f"  Sample qid={sample_qid} top-3: {results[sample_qid][:3]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reranking con llama-nemotron-rerank-1b-v2 fine-tuned"
    )
    parser.add_argument("--model_dir",     required=True)
    parser.add_argument("--nemotron_run",  required=True)
    parser.add_argument("--collection",    required=True)
    parser.add_argument("--queries",       required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--rerank_depth",  type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--max_doc_chars", type=int, default=4000)
    parser.add_argument("--no_sigmoid",    action="store_true",
                        help="Usa logit raw invece di sigmoid (default: sigmoid attiva)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # trust_remote_code=True è OBBLIGATORIO per Nemotron
    log.info(f"Caricamento modello da {args.model_dir} (trust_remote_code=True)...")
    model = CrossEncoder(
        args.model_dir,
        device=device,
        trust_remote_code=True,
        automodel_args={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    )
    log.info(f"max_length={model.max_length}, num_labels={model.num_labels}")

    corpus       = load_collection(args.collection, args.max_doc_chars)
    queries      = load_queries(args.queries)
    nemotron_run = load_nemotron_run(args.nemotron_run)

    results      = {}
    n_missing_q  = 0
    n_empty_docs = 0

    for qid, nemotron_docs in tqdm(nemotron_run.items(), desc="Reranking"):
        if qid not in queries:
            n_missing_q += 1
            continue

        query         = queries[qid]
        candidate_ids = nemotron_docs[:args.rerank_depth]
        valid_ids     = [d for d in candidate_ids if d in corpus]
        valid_docs    = [corpus[d] for d in valid_ids]

        if not valid_ids:
            n_empty_docs += 1
            continue

        pairs  = [[query, doc] for doc in valid_docs]
        scores = predict_with_oom_fallback(
            model, pairs, args.batch_size,
            use_sigmoid=not args.no_sigmoid,
        )

        ranked       = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [doc_id for doc_id, _ in ranked]

    if n_missing_q:
        log.warning(f"Query mancanti: {n_missing_q}")
    if n_empty_docs:
        log.warning(f"Query senza doc nel corpus: {n_empty_docs}")

    sanity_check(results, nemotron_run, args.rerank_depth)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"✓ Output: {out_path}  ({len(results):,} queries)")


if __name__ == "__main__":
    main()