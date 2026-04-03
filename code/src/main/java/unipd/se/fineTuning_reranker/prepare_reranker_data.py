"""
prepare_reranker_data.py — Costruisce il dataset di fine-tuning per Qwen3-Reranker
====================================================================================
Legge:
  - data/en_train.json          (query: index, text, pubkey)
  - data/collection_data.json   (corpus: pubkey, title, abstract)
  - results/bm25_results.json   (candidati BM25: {qid -> [pubkey, ...]})

Scrive (in data/finetune_reranker/):
  - train.jsonl                 (triple con hard negatives — 80% delle query)
  - dev.jsonl                   (10% delle query — per eval durante training)
  - test.jsonl                  (10% delle query — eval finale, non toccare)
  - split_indices.json          (seed + indici per riproducibilità)
  - stats.json                  (statistiche del dataset)

Formato JSONL output (ogni riga):
  {
    "query": "testo query pulita",
    "positive": "titolo. abstract del paper gold",
    "negatives": ["doc hard neg 1", "doc hard neg 2", ...],  // N_HARD_NEG documenti
    "qid": 42,
    "pubkey_gold": 1234
  }

Hard negatives: top-K candidati BM25 che NON sono il gold pubkey.
Questo è il setup più efficace per domain adaptation del reranker perché
il modello deve imparare a distinguere paper topicamente simili ma non corretti
(quelli che BM25 confonde) — esattamente il caso in cui il reranker fallisce.

Uso:
  python prepare_reranker_data.py
  python prepare_reranker_data.py --queries data/en_train.json \\
      --papers data/collection_data.json \\
      --bm25 results/bm25_results.json \\
      --output data/finetune_reranker \\
      --n_hard_neg 15 \\
      --bm25_pool 100 \\
      --seed 42
"""

import json
import re
import os
import random
import argparse
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--queries",    default="../../../../../../data/en_train.json")
parser.add_argument("--papers",     default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",       default="../../../../../../../results/bm25_results.json")
parser.add_argument("--output",     default="../../../../../../../data/finetune_reranker")
parser.add_argument("--n_hard_neg", type=int, default=15,
                    help="Hard negatives per query (default: 15). "
                         "Aumenta fino a 31 se hai memoria sufficiente.")
parser.add_argument("--bm25_pool",  type=int, default=100,
                    help="Quanti candidati BM25 considerare per pescare negativi (default: 100)")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--train_ratio", type=float, default=0.80)
parser.add_argument("--dev_ratio",   type=float, default=0.10)
# test_ratio = 1 - train_ratio - dev_ratio (implicito)
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Cleaning (identico a CausalReranker.py per coerenza)
# ─────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    text = re.sub(r'@\w+\s?', '', text)          # rimuovi @mention
    text = re.sub(r'http\S+', '', text)           # rimuovi URL
    text = re.sub(r'#(\w+)', r'\1', text)         # rimuovi # ma mantieni testo
    text = re.sub(r'^\d+\/\s*', '', text, flags=re.MULTILINE)  # numeri thread
    text = text.encode('ascii', 'ignore').decode()# rimuovi non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────
print("Loading data...")

with open(args.queries, "r", encoding="utf-8") as f:
    queries_raw = json.load(f)

with open(args.papers, "r", encoding="utf-8") as f:
    papers_raw = json.load(f)

with open(args.bm25, "r", encoding="utf-8") as f:
    bm25_results: dict = json.load(f)  # {str(qid) -> [pubkey, ...]}

# Corpus: pubkey (int) -> "titolo. abstract"
paper_texts: dict = {}
for p in papers_raw:
    pk = p["pubkey"]
    title    = p.get("title", "") or ""
    abstract = p.get("abstract", "") or ""
    paper_texts[pk] = (title + ". " + abstract).strip()

print(f"  Corpus: {len(paper_texts)} documenti")
print(f"  Queries raw: {len(queries_raw)}")
print(f"  BM25 results: {len(bm25_results)} entries")


# ─────────────────────────────────────────────────────────────
# Build examples
# ─────────────────────────────────────────────────────────────
# bm25_results può avere chiavi str o int — normalizziamo a str
bm25_by_qid = {str(k): v for k, v in bm25_results.items()}

random.seed(args.seed)
all_examples = []

stats = {
    "total_queries":         len(queries_raw),
    "skipped_empty_query":   0,
    "skipped_no_gold_doc":   0,
    "skipped_no_bm25":       0,
    "skipped_zero_hardnegs": 0,
    "used":                  0,
    "hard_negs_avg":         0.0,
    "fallback_random_negs":  0,   # query per cui BM25 non aveva abbastanza hard neg
}

total_hard_negs = 0

# Tutti i pubkeys del corpus come pool per fallback negativi random
all_pubkeys = list(paper_texts.keys())

for q in queries_raw:
    qid     = q["index"]
    raw_text = q.get("text", "")
    pubkey_gold = q.get("pubkey")

    # Pulisci query
    query = clean_tweet(raw_text)
    if len(query) < 10:
        stats["skipped_empty_query"] += 1
        continue

    # Controlla che il documento gold esista nel corpus
    if pubkey_gold not in paper_texts:
        stats["skipped_no_gold_doc"] += 1
        continue

    positive_doc = paper_texts[pubkey_gold]
    if len(positive_doc) < 50:
        stats["skipped_no_gold_doc"] += 1
        continue

    # Hard negatives da BM25
    bm25_candidates = bm25_by_qid.get(str(qid), [])
    if not bm25_candidates:
        stats["skipped_no_bm25"] += 1
        continue

    # Prendi i primi bm25_pool candidati, escludi il gold
    hard_neg_pubkeys = [
        pk for pk in bm25_candidates[:args.bm25_pool]
        if pk != pubkey_gold and pk in paper_texts
    ]

    if len(hard_neg_pubkeys) == 0:
        # Nessun hard negative trovato: usa random dal corpus come fallback
        stats["skipped_zero_hardnegs"] += 1
        continue

    # Se i candidati BM25 non bastano, integra con random negatives
    if len(hard_neg_pubkeys) < args.n_hard_neg:
        # Fallback: pesca random dal corpus (escludi gold)
        fallback_pool = [pk for pk in all_pubkeys
                         if pk != pubkey_gold and pk not in set(hard_neg_pubkeys)]
        n_missing = args.n_hard_neg - len(hard_neg_pubkeys)
        hard_neg_pubkeys += random.sample(fallback_pool, min(n_missing, len(fallback_pool)))
        stats["fallback_random_negs"] += 1

    # Tronca a n_hard_neg
    selected_negs = hard_neg_pubkeys[:args.n_hard_neg]
    neg_docs = [paper_texts[pk] for pk in selected_negs]

    total_hard_negs += len(selected_negs)
    stats["used"] += 1

    all_examples.append({
        "query":       query,
        "positive":    positive_doc,
        "negatives":   neg_docs,
        "qid":         qid,
        "pubkey_gold": pubkey_gold,
    })

stats["hard_negs_avg"] = total_hard_negs / max(stats["used"], 1)

print(f"\n=== Build stats ===")
print(f"  Esempi validi:             {stats['used']}")
print(f"  Skipped (query vuota):     {stats['skipped_empty_query']}")
print(f"  Skipped (no gold doc):     {stats['skipped_no_gold_doc']}")
print(f"  Skipped (no BM25):         {stats['skipped_no_bm25']}")
print(f"  Skipped (zero hard negs):  {stats['skipped_zero_hardnegs']}")
print(f"  Fallback random negs:      {stats['fallback_random_negs']}")
print(f"  Hard negs avg per query:   {stats['hard_negs_avg']:.2f}")


# ─────────────────────────────────────────────────────────────
# Train / Dev / Test split
# ─────────────────────────────────────────────────────────────
random.seed(args.seed)
indices = list(range(len(all_examples)))
random.shuffle(indices)

n_total = len(indices)
n_train = int(n_total * args.train_ratio)
n_dev   = int(n_total * args.dev_ratio)
n_test  = n_total - n_train - n_dev

train_idx = indices[:n_train]
dev_idx   = indices[n_train: n_train + n_dev]
test_idx  = indices[n_train + n_dev:]

train_examples = [all_examples[i] for i in train_idx]
dev_examples   = [all_examples[i] for i in dev_idx]
test_examples  = [all_examples[i] for i in test_idx]

print(f"\n=== Split ===")
print(f"  Train: {len(train_examples)} ({len(train_examples)/n_total*100:.1f}%)")
print(f"  Dev:   {len(dev_examples)}   ({len(dev_examples)/n_total*100:.1f}%)")
print(f"  Test:  {len(test_examples)}  ({len(test_examples)/n_total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
os.makedirs(args.output, exist_ok=True)

def write_jsonl(path: str, examples: list):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Scritto: {path} ({len(examples)} esempi)")

write_jsonl(os.path.join(args.output, "train.jsonl"), train_examples)
write_jsonl(os.path.join(args.output, "dev.jsonl"),   dev_examples)
write_jsonl(os.path.join(args.output, "test.jsonl"),  test_examples)

# Salva split indices per riproducibilità
split_meta = {
    "seed":       args.seed,
    "n_total":    n_total,
    "train_idx":  train_idx,
    "dev_idx":    dev_idx,
    "test_idx":   test_idx,
    "args": {
        "n_hard_neg":   args.n_hard_neg,
        "bm25_pool":    args.bm25_pool,
        "train_ratio":  args.train_ratio,
        "dev_ratio":    args.dev_ratio,
    }
}
with open(os.path.join(args.output, "split_indices.json"), "w") as f:
    json.dump(split_meta, f, indent=2)

stats.update({
    "n_train": len(train_examples),
    "n_dev":   len(dev_examples),
    "n_test":  len(test_examples),
})
with open(os.path.join(args.output, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nDone. Dataset salvato in: {args.output}/")
print("Prossimo passo: python finetune_qwen3_reranker.py")