"""
rerank_with_finetuned_crossencoder.py

Reranka i risultati già rerankati da Nemotron usando il CrossEncoder ModernBERT fine-tuned.

Input:
  - models/modernbert-large-retrix-nemotron-distilled/final  (modello fine-tuned)
  - reranked_results_nemotron_topk400.json  {query_index: [doc_id, ...]}
  - collection_data.json                   [{pubkey, title, abstract, ...}]
  - en_train.json                          [{index, text, pubkey}]  ← per le query

Output:
  - reranked_results_modernbert_ft_topkN.json   {query_index: [doc_id, ...]}
    (stesso formato del Nemotron run, drop-in nel tuo pipeline)

Pipeline:
  BM25 → Nemotron (top 400) → ModernBERT fine-tuned (top N) → Evaluator.java

Uso:
  python rerank_with_finetuned_crossencoder.py \
    --model_dir    models/modernbert-large-retrix-nemotron-distilled/final \
    --nemotron_run reranked_results_nemotron_topk400.json \
    --collection   collection_data.json \
    --queries      en_train.json \
    --output       reranked_results_modernbert_ft_top100.json \
    --rerank_depth 100 \
    --batch_size   64
"""

import json
import emoji
import logging
import re
import gc
import argparse
from pathlib import Path

import torch
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Caricamento file JSON ─────────────────────────────────────────────────────
def load_collection(path: str, max_chars: int = 2000) -> dict[str, str]:
    """
    Carica collection_data.json.
    Ritorna {str(pubkey): "title. abstract"}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = {}
    for item in data:
        key      = str(item["pubkey"])
        title    = item.get("title", "")    or ""
        abstract = item.get("abstract", "") or ""
        text     = f"{title}. {abstract}".strip(". ")[:max_chars]
        if text:
            corpus[key] = text

    log.info(f"Corpus caricato: {len(corpus):,} documenti  ← {path}")
    return corpus


def load_queries(path: str) -> dict[str, str]:
    """
    Carica en_train.json (o qualsiasi file con {index, text, pubkey}).
    Ritorna {str(index): query_text_cleaned}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    queries = {}
    for item in data:
        qid          = str(item["index"])
        queries[qid] = clean_tweet(item["text"])

    log.info(f"Queries caricate: {len(queries):,}  ← {path}")
    return queries


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    """
    Carica reranked_results_nemotron_topk400.json.
    Formato: {query_index: [doc_id_rank1, doc_id_rank2, ...]}
    Chiavi e valori vengono normalizzati a stringa.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    run = {str(k): [str(d) for d in v] for k, v in raw.items()}
    total_docs = sum(len(v) for v in run.values())
    log.info(f"Nemotron run caricato: {len(run):,} queries, {total_docs:,} doc totali  ← {path}")
    return run


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_query(
        model:      CrossEncoder,
        query:      str,
        docs:       list[str],
        doc_ids:    list[str],
        batch_size: int,
        device:     str,
) -> list[tuple[str, float]]:
    """
    Calcola score per tutti i documenti di una query.
    Ritorna lista di (doc_id, score) NON ordinata.
    Gestisce OOM riducendo il batch_size a metà.
    """
    pairs  = [[query, doc] for doc in docs]
    scores = _predict_with_oom_fallback(model, pairs, batch_size)
    return list(zip(doc_ids, scores))


def _predict_with_oom_fallback(
        model:      CrossEncoder,
        pairs:      list[list[str]],
        batch_size: int,
) -> list[float]:
    """predict() con fallback su OOM: dimezza il batch finché ce la fa."""
    current_bs = batch_size
    while current_bs >= 1:
        try:
            scores = model.predict(
                pairs,
                batch_size=current_bs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # scores può essere array 1-D o scalare
            if isinstance(scores, (float, int)):
                scores = [float(scores)]
            else:
                scores = scores.tolist()
            return scores

        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM con batch_size={current_bs}. Provo con {current_bs // 2}...")
            torch.cuda.empty_cache()
            gc.collect()
            current_bs //= 2

    # Ultima risorsa: processo uno per uno senza autocast
    log.error("OOM anche con batch_size=1. Restituisco score 0.0 per tutti.")
    return [0.0] * len(pairs)


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(results: dict, nemotron_run: dict, rerank_depth: int):
    if not results:
        log.error("SANITY CHECK FAILED: output vuoto!")
        return

    n_out   = len(results)
    n_in    = len(nemotron_run)
    avg_len = sum(len(v) for v in results.values()) / n_out
    coverage = n_out / max(1, n_in)

    log.info("── Sanity check ──────────────────────────────")
    log.info(f"  Queries in input (Nemotron): {n_in:,}")
    log.info(f"  Queries in output:           {n_out:,}  ({coverage:.1%} coverage)")
    log.info(f"  Avg doc per query:           {avg_len:.1f}  (atteso: ≤{rerank_depth})")

    # Campiona una query per ispezione manuale
    sample_qid    = next(iter(results))
    sample_top5   = results[sample_qid][:5]
    log.info(f"  Sample qid={sample_qid} top-5:")
    for rank, doc_id in enumerate(sample_top5, 1):
        log.info(f"    rank {rank}: doc_id={doc_id}")
    log.info("──────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Reranking con CrossEncoder ModernBERT fine-tuned sui risultati Nemotron"
    )
    parser.add_argument("--model_dir",    required=True,
                        help="Path al modello fine-tuned (es. models/.../final)")
    parser.add_argument("--nemotron_run", required=True,
                        help="reranked_results_nemotron_topk400.json")
    parser.add_argument("--collection",  required=True,
                        help="collection_data.json")
    parser.add_argument("--queries",     required=True,
                        help="en_train.json (contiene index + text delle query)")
    parser.add_argument("--output",      required=True,
                        help="Path output JSON, es. reranked_results_modernbert_ft_top100.json")
    parser.add_argument("--rerank_depth", type=int, default=100,
                        help="Quanti doc dal Nemotron run considerare (default: 100)")
    parser.add_argument("--batch_size",  type=int, default=64,
                        help="Batch size per CrossEncoder.predict (default: 64)")
    parser.add_argument("--max_doc_chars", type=int, default=4000,
                        help="Troncamento testo documenti in caratteri (default: 2000)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── 1. Carica modello ──────────────────────────────────────────────────
    log.info(f"Caricamento CrossEncoder da {args.model_dir} ...")
    model = CrossEncoder(
        args.model_dir,
        device=device,
        automodel_args={"torch_dtype": torch.bfloat16},
    )
    log.info(f"  max_length={model.max_length}, num_labels={model.num_labels}")

    # ── 2. Carica dati ─────────────────────────────────────────────────────
    corpus       = load_collection(args.collection, max_chars=args.max_doc_chars)
    queries      = load_queries(args.queries)
    nemotron_run = load_nemotron_run(args.nemotron_run)

    # ── 3. Reranking ───────────────────────────────────────────────────────
    results      = {}
    n_missing_q  = 0
    n_empty_docs = 0

    for qid, nemotron_docs in tqdm(nemotron_run.items(), desc="Reranking"):
        # Query mancante nel file queries
        if qid not in queries:
            n_missing_q += 1
            continue

        query = queries[qid]

        # Prendi i primi rerank_depth documenti dal Nemotron run
        candidate_ids = nemotron_docs[:args.rerank_depth]

        # Filtra documenti presenti nel corpus
        valid_ids  = [d for d in candidate_ids if d in corpus]
        valid_docs = [corpus[d] for d in valid_ids]

        if not valid_ids:
            n_empty_docs += 1
            continue

        # Scoraggio
        scored = score_query(
            model=model,
            query=query,
            docs=valid_docs,
            doc_ids=valid_ids,
            batch_size=args.batch_size,
            device=device,
        )

        # Ordina per score decrescente
        ranked    = sorted(scored, key=lambda x: x[1], reverse=True)
        results[qid] = [doc_id for doc_id, _ in ranked]

    log.info(f"Query mancanti nel file queries: {n_missing_q}")
    log.info(f"Query senza documenti nel corpus: {n_empty_docs}")

    # ── 4. Sanity check ────────────────────────────────────────────────────
    sanity_check(results, nemotron_run, args.rerank_depth)

    # ── 5. Salvataggio ─────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"✓ Output salvato: {out_path}  ({len(results):,} queries)")


if __name__ == "__main__":
    main()