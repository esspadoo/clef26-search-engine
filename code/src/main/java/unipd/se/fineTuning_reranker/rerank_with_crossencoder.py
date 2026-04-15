"""
rerank_with_crossencoder.py

Inference con il CrossEncoder fine-tuned (sentence-transformers).
Drop-in nel pipeline esistente, compatibile con Evaluator.java.

Differenza rispetto a rerank_with_finetuned.py:
  Usa CrossEncoder.predict() invece di AutoModelForSequenceClassification,
  il che è più semplice e gestisce automaticamente la tokenizzazione.

Uso:
  python rerank_with_crossencoder.py \
    --model_dir  models/modernbert-large-retrix-reranker/final \
    --bm25_run   results/BM25TopK1000_run.txt \
    --corpus     corpus.tsv \
    --queries    queries.tsv \
    --output_dir results/ \
    --rerank_depth 130 \
    --top_k 100
"""

import logging
import re
import gc
from pathlib import Path
from collections import defaultdict

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_corpus(path: str) -> dict[str, str]:
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                corpus[parts[0]] = clean_tweet(parts[1])
    log.info(f"Corpus: {len(corpus):,} documenti")
    return corpus


def load_queries(path: str) -> dict[str, str]:
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = clean_tweet(parts[1])
    return queries


def load_run(path: str) -> dict[str, list[tuple[int, str, float]]]:
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                run[parts[0]].append((int(parts[3]), parts[2], float(parts[4])))
    for qid in run:
        run[qid].sort()
    return dict(run)


def write_trec_run(results: dict, output_path: str, run_name: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in sorted(results, key=lambda x: int(x) if x.isdigit() else x):
            for rank, (doc_id, score) in enumerate(results[qid], start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")
    log.info(f"Run salvata: {output_path}")


def sanity_check(results: dict, top_k: int):
    if not results:
        log.error("SANITY CHECK FAILED: risultati vuoti!")
        return
    avg = sum(len(v) for v in results.values()) / len(results)
    log.info(f"Sanity check: {len(results):,} queries, avg docs={avg:.1f} (atteso {top_k})")
    sample_qid = next(iter(results))
    log.info(f"  Sample qid={sample_qid}: {results[sample_qid][:3]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--bm25_run",     required=True)
    parser.add_argument("--corpus",       required=True)
    parser.add_argument("--queries",      required=True)
    parser.add_argument("--output_dir",   default="./results")
    parser.add_argument("--rerank_depth", type=int, default=130)
    parser.add_argument("--top_k",        type=int, default=100)
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--max_length",   type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Carica CrossEncoder — usa sigmoid=True per scores in [0,1]
    log.info(f"Caricamento CrossEncoder da {args.model_dir}")
    model = CrossEncoder(
        args.model_dir,
        max_length=args.max_length,
        device=device,
        automodel_args={"torch_dtype": torch.bfloat16},
    )

    corpus  = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    bm25    = load_run(args.bm25_run)

    results  = {}
    run_name = "ModernBERT-large-FT-CE"

    for qid, ranked_docs in tqdm(bm25.items(), desc="Reranking"):
        if qid not in queries:
            continue

        query   = queries[qid]
        doc_ids = [d for _, d, _ in ranked_docs[:args.rerank_depth]]
        docs    = [corpus.get(d, "") for d in doc_ids]

        valid = [(did, doc) for did, doc in zip(doc_ids, docs) if doc.strip()]
        if not valid:
            continue
        v_ids, v_docs = zip(*valid)

        # CrossEncoder.predict() gestisce automaticamente batching e tokenizzazione
        try:
            pairs  = [[query, doc] for doc in v_docs]
            scores = model.predict(
                pairs,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM su qid={qid}. Riduco batch_size a 8.")
            torch.cuda.empty_cache()
            gc.collect()
            pairs  = [[query, doc] for doc in v_docs]
            scores = model.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        ranked = sorted(zip(v_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        results[qid] = ranked[:args.top_k]

    sanity_check(results, args.top_k)

    bm25_stem   = Path(args.bm25_run).stem
    out_path    = Path(args.output_dir) / f"{bm25_stem}_RerankTop{args.rerank_depth}_ModernBERT-large-FT_top{args.top_k}_run.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run(results, str(out_path), run_name)
    log.info(f"✓ Output: {out_path}")


if __name__ == "__main__":
    main()