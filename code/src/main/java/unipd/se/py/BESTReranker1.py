"""
Reranker.py — cross-Encoder re-ranking dei risultati BM25

Legge:
  - data/expanded_queries_4.json   (query originali, per avere il testo originale)
  - data/collection_data.json      (corpus, per recuperare titolo+abstract dei candidati)
  - results/bm25_results.json      (output di Main.java: { qid → [pubkey, ...] })

Scrive:
  - results/reranked_results.json  (stesso formato: { qid → [pubkey, ...] })

Uso:
  python Reranker.py
  # oppure con argomenti:
  python Reranker.py --queries data/expanded_queries_4.json \
                     --papers  data/collection_data.json \
                     --bm25    results/bm25_results.json \
                     --output  results/reranked_results.json \
                     --top_k   100
"""

import json
import argparse
import os
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--queries", default="../../../../../../data/expanded_queries_4.json")
parser.add_argument("--papers",  default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",    default="../../../../../../../results/bm25_results.json")
parser.add_argument("--output",  default="../../../../../../../results/reranked_results.json")
parser.add_argument("--top_k",   type=int, default=100,
                    help="Quanti candidati BM25 passare al re-ranker (default: 100)")
parser.add_argument("--batch",   type=int, default=64,
                    help="Batch size per il cross-encoder (default: 64)")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────
# Carica dati
# ──────────────────────────────────────────────────────────────
print("Loading data...")

with open(args.queries, "r", encoding="utf-8") as f:
    queries_raw = json.load(f)

with open(args.papers, "r", encoding="utf-8") as f:
    papers_raw = json.load(f)

with open(args.bm25, "r", encoding="utf-8") as f:
    bm25_results: dict[str, list[str]] = json.load(f)

# FIX: normalizza SEMPRE le chiavi a stringa.
# collection_data.json ha pubkey come int (9394),
# ma Lucene serializza i campi StringField come string ("9394").
# Senza str() il lookup fallisce silenziosamente per TUTTI i documenti
# e il re-ranker restituisce l'ordine BM25 invariato.
print("Building paper index (normalizing pubkeys to str)...")
paper_texts: dict[str, str] = {
    str(p["pubkey"]): (p.get("title", "") + " " + p.get("abstract", "")).strip()
    for p in papers_raw
}

# Normalizza anche i query index a stringa per coerenza con bm25_results
query_texts: dict[str, str] = {
    str(q["index"]): q.get("original", q.get("text", ""))
    for q in queries_raw
}

# Diagnostica: verifica che il lookup funzioni su un campione
sample_bm25_key = next(iter(bm25_results))
sample_candidates = bm25_results[sample_bm25_key][:3]
hits = sum(1 for pk in sample_candidates if str(pk) in paper_texts)
print(f"Sanity check — query '{sample_bm25_key}': {hits}/{len(sample_candidates)} candidates found in paper_texts")
if hits == 0:
    print("WARNING: 0 candidates found. Check that pubkey types match between bm25_results and collection_data.")

# ──────────────────────────────────────────────────────────────
# Carica il cross-encoder
# Il modello ms-marco-MiniLM-L-6-v2 è addestrato su MS MARCO per
# il ranking di passaggi — funziona bene anche su abstract scientifici.
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading cross-encoder: {MODEL_NAME} ...")
reranker = CrossEncoder(MODEL_NAME, device=DEVICE, max_length=512)

# ──────────────────────────────────────────────────────────────
# Re-ranking
# Per ogni query:
#   1. prende i top_k candidati BM25
#   2. costruisce le coppie (query_originale, testo_doc)
#   3. il cross-encoder assegna uno score a ogni coppia
#   4. riordina per score decrescente
# ──────────────────────────────────────────────────────────────
print(f"Re-ranking {len(bm25_results)} queries (top_k={args.top_k})...")

reranked_results: dict[str, list[str]] = {}
missing_docs = 0
total_pairs = 0

for qid, candidate_pubkeys in tqdm(bm25_results.items(), desc="Re-ranking"):

    query_text = query_texts.get(qid, "")
    if not query_text:
        # Query non trovata: mantieni l'ordine BM25
        reranked_results[qid] = candidate_pubkeys[:args.top_k]
        continue

    # Prendi solo i top_k candidati (BM25 ne restituisce già 100 di default)
    candidates = candidate_pubkeys[:args.top_k]

    # Costruisci le coppie (query, documento)
    pairs: list[tuple[str, str]] = []
    valid_keys: list[str] = []

    for pk in candidates:
        # FIX: normalizza anche la chiave di lookup a stringa
        doc_text = paper_texts.get(str(pk))
        if doc_text:
            pairs.append((query_text, doc_text))
            valid_keys.append(pk)
        else:
            missing_docs += 1

    if not pairs:
        reranked_results[qid] = candidates
        continue

    # Score con il cross-encoder (batched)
    total_pairs += len(pairs)
    scores = reranker.predict(pairs, batch_size=args.batch, show_progress_bar=False)

    # Ordina per score decrescente
    ranked = sorted(zip(valid_keys, scores), key=lambda x: -x[1])
    reranked_results[qid] = [pk for pk, _ in ranked]

print(f"\nTotal pairs scored: {total_pairs}")
if missing_docs > 0:
    print(f"  Warning: {missing_docs} candidate pubkeys not found in collection (skipped)")

# ──────────────────────────────────────────────────────────────
# Salva
# ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(reranked_results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(reranked_results)} re-ranked queries → {args.output}")
print("Done. Now run the Java evaluator pointing to reranked_results.json.")