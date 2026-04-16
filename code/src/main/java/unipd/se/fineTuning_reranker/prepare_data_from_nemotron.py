"""
prepare_data_from_nemotron.py

Prepara il dataset di fine-tuning per ModernBERT CrossEncoder partendo da:

CORPUS 2026 (progetto corrente):
  - en_train.json          → ground truth: {index, text (query), pubkey (doc corretto)}
  - collection_data.json   → corpus: {pubkey, title, abstract, ...}
  - reranked_results_nemotron_topk400.json → {query_index: [doc_id_rank1, ...]}

CORPUS 2025 (progetto anno scorso, opzionale):
  - en_train.tsv           → ground truth: post_id, tweet_text, cord_uid
  - collection_data2025.json → corpus: {cord_uid, title, abstract, ...}
  - reranked_results_nemotron_2025.json (opzionale, se lo hai rerankato)

Strategia hard negatives:
  Il Nemotron run ci dà una lista ordinata per score decrescente.
  I documenti in quella lista che NON sono il positivo sono hard negatives
  di qualità superiore ai BM25 negatives: Nemotron li ha già giudicati
  "plausibili" ma noi sappiamo che sono falsi positivi.

  In pratica prendiamo:
    - Positivo:       il documento con pubkey/cord_uid corretto (da en_train)
    - Hard negatives: i top-K documenti nel Nemotron run, escludendo il positivo
      (concentrati nei primissimi rank: sono i più ambigui e informativi)

Output:
  training_data/
    train_pairs.jsonl
    dev_pairs.jsonl
    stats.json

Formato output (per CrossEncoderTrainer con BinaryCrossEntropyLoss):
  labeled-pair: {query, passage, label}
  con label=1 per positivi, label=0 per negativi

Uso:
  python prepare_data_from_nemotron.py \
    --train_json      en_train.json \
    --collection_json collection_data.json \
    --nemotron_run    reranked_results_nemotron_topk400.json \
    --out_dir         training_data \
    [--train_tsv      en_train.tsv] \
    [--collection_json2025 collection_data2025.json] \
    [--nemotron_run2025    reranked_results_nemotron_2025.json] \
    [--num_hard_neg   5] \
    [--neg_offset     0]
"""

import json
import random
import logging
import argparse
import csv
import emoji
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    num_hard_neg:  int   = 5      # quanti hard negatives per positivo
    # neg_offset: salta i primissimi negativi nel Nemotron run.
    # Utile se vuoi escludere i "quasi-positivi" più ambigui (rank 1-2)
    # e partire da quelli leggermente più facili. Default 0 = prendi i top.
    neg_offset:    int   = 0
    dev_ratio:     float = 0.1
    seed:          int   = 42
    max_doc_chars: int   = 6000   # troncamento testo documenti
# ────────────────────────────────────────────────────────────────────────────


import re

def clean_tweet(text: str) -> str:
    """Preprocessing allineato con il pipeline Java/Lucene del progetto."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = emoji.replace_emoji(text, replace='')  # Rimuovi emoji

    text = re.sub(r"\s+", " ", text).strip()
    return text


def doc_to_text(doc: dict, max_chars: int) -> str:
    """
    Costruisce il testo del documento da title + abstract.
    Gestisce sia il formato 2026 (title/abstract) che 2025 (stessa struttura).
    """
    title    = doc.get("title", "") or ""
    abstract = doc.get("abstract", "") or ""
    text     = f"{title}. {abstract}".strip(". ")
    return text[:max_chars]


# ── Caricamento corpus ───────────────────────────────────────────────────────
def load_collection_2026(path: str, max_chars: int) -> dict[str, str]:
    """
    Carica collection_data.json (corpus 2026).
    Chiave: str(pubkey)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = {}
    for item in data:
        key  = str(item["pubkey"])
        text = doc_to_text(item, max_chars)
        if text:
            corpus[key] = text

    log.info(f"Corpus 2026: {len(corpus):,} documenti (da {path})")
    return corpus


def load_collection_2025(path: str, max_chars: int) -> dict[str, str]:
    """
    Carica collection_data2025.json (corpus 2025).
    Chiave: cord_uid (stringa)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = {}
    for item in data:
        key  = str(item["cord_uid"])
        text = doc_to_text(item, max_chars)
        if text:
            corpus[key] = text

    log.info(f"Corpus 2025: {len(corpus):,} documenti (da {path})")
    return corpus


# ── Caricamento ground truth ─────────────────────────────────────────────────
def load_ground_truth_2026(path: str) -> dict[str, str]:
    """
    Carica en_train.json.
    Ritorna: {str(index): str(pubkey)}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    gt = {}
    for item in data:
        qid     = str(item["index"])
        pos_id  = str(item["pubkey"])
        gt[qid] = pos_id

    log.info(f"Ground truth 2026: {len(gt):,} query-documento pairs (da {path})")
    return gt


def load_ground_truth_2025(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Carica en_train.tsv.
    Ritorna:
      queries:    {str(post_id): tweet_text}
      ground_truth: {str(post_id): cord_uid}
    """
    queries = {}
    gt      = {}

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid             = str(row["post_id"])
            queries[qid]    = row["tweet_text"]
            gt[qid]         = str(row["cord_uid"])

    log.info(f"Ground truth 2025: {len(gt):,} query-documento pairs (da {path})")
    return queries, gt


def load_queries_2026(train_json_path: str) -> dict[str, str]:
    """Estrae le query da en_train.json. Ritorna {str(index): text}"""
    with open(train_json_path, encoding="utf-8") as f:
        data = json.load(f)
    queries = {str(item["index"]): clean_tweet(item["text"]) for item in data}
    log.info(f"Queries 2026: {len(queries):,}")
    return queries


# ── Caricamento Nemotron run ─────────────────────────────────────────────────
def load_nemotron_run(path: str) -> dict[str, list[str]]:
    """
    Carica il file JSON del Nemotron reranked run.
    Formato atteso: {str(query_index): [doc_id_1, doc_id_2, ...]}
    (ordinati per score decrescente = rank 1 = più rilevante secondo Nemotron)

    Gestisce sia chiavi int che str.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    run = {str(k): [str(d) for d in v] for k, v in raw.items()}
    n_docs = sum(len(v) for v in run.values())
    log.info(f"Nemotron run: {len(run):,} queries, {n_docs:,} doc totali (da {path})")
    return run


# ── Costruzione esempi ───────────────────────────────────────────────────────
def build_examples(
        queries:      dict[str, str],    # {qid: query_text}
        corpus:       dict[str, str],    # {doc_id: doc_text}
        ground_truth: dict[str, str],    # {qid: positive_doc_id}
        nemotron_run: dict[str, list[str]],  # {qid: [doc_id ranked by Nemotron]}
        cfg:          Config,
        split_tag:    str = "",
) -> list[dict]:
    """
    Costruisce coppie (query, passage, label) usando:
      - Positivo: il doc corretto da ground_truth
      - Hard negatives: i top documenti nel Nemotron run, escludendo il positivo

    Logica di selezione negatives:
      Nemotron run[qid] = [d1, d2, d3, ..., d400]  (già ordinati per score desc)
      Escludiamo il positivo, poi prendiamo i primi num_hard_neg dalla lista.
      Con neg_offset > 0 saltiamo i primissimi (i più ambigui).

    Perché questo è meglio dei BM25 negatives:
      Nemotron ha già "filtrato" i candidati plausibili.
      Un documento che Nemotron rankava al rank 2-10 ma non è il positivo
      è un hard negative di qualità altissima per insegnare a ModernBERT
      a distinguere il vero positivo dai quasi-positivi.
    """
    examples     = []
    stats        = Counter()

    for qid, query_text in queries.items():
        pos_id = ground_truth.get(qid)
        if pos_id is None:
            stats["no_gt"] += 1
            continue

        if pos_id not in corpus:
            stats["pos_not_in_corpus"] += 1
            continue

        if qid not in nemotron_run:
            stats["no_nemotron_run"] += 1
            # Fallback: niente hard negatives, skip
            continue

        nemotron_docs = nemotron_run[qid]  # già ordinati per rank Nemotron

        # Hard negatives: tutti i doc nel Nemotron run tranne il positivo
        hard_negs = [d for d in nemotron_docs if d != pos_id and d in corpus]

        if len(hard_negs) < 2:
            stats["too_few_negs"] += 1
            continue

        # Applica offset e prendi num_hard_neg
        sliced = hard_negs[cfg.neg_offset : cfg.neg_offset + cfg.num_hard_neg]
        if len(sliced) < cfg.num_hard_neg:
            # Se l'offset è troppo alto, completa con i successivi disponibili
            sliced = hard_negs[:cfg.num_hard_neg]

        # Riga positiva
        examples.append({
            "query":   query_text,
            "passage": corpus[pos_id],
            "label":   1,
            "qid":     qid,
            "doc_id":  pos_id,
        })
        stats["positives"] += 1

        # Righe negative
        for neg_id in sliced:
            examples.append({
                "query":   query_text,
                "passage": corpus[neg_id],
                "label":   0,
                "qid":     qid,
                "doc_id":  neg_id,
            })
            stats["negatives"] += 1

    log.info(
        f"[{split_tag}] Esempi: {len(examples):,} "
        f"({stats['positives']:,} pos + {stats['negatives']:,} neg). "
        f"Saltati: no_gt={stats['no_gt']}, no_corpus={stats['pos_not_in_corpus']}, "
        f"no_run={stats['no_nemotron_run']}, pochi_neg={stats['too_few_negs']}"
    )
    return examples


# ── Split e salvataggio ──────────────────────────────────────────────────────
def split_by_query(examples: list[dict], dev_ratio: float, seed: int):
    """
    Split train/dev per query (non per riga).
    Tutti gli esempi di una stessa query vanno nello stesso split,
    per evitare data leakage.
    """
    # Raggruppa per qid
    qid_to_examples = defaultdict(list)
    for ex in examples:
        qid_to_examples[ex["qid"]].append(ex)

    qids = list(qid_to_examples.keys())
    random.Random(seed).shuffle(qids)

    n_dev      = max(10, int(len(qids) * dev_ratio))
    dev_qids   = set(qids[:n_dev])
    train_qids = set(qids[n_dev:])

    train = [ex for qid in train_qids for ex in qid_to_examples[qid]]
    dev   = [ex for qid in dev_qids   for ex in qid_to_examples[qid]]

    return train, dev


def save_jsonl(examples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"Salvato: {path} ({len(examples):,} righe)")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Prepara dati di fine-tuning usando Nemotron run come hard negatives"
    )
    # Corpus 2026 (obbligatorio)
    parser.add_argument("--train_json",      required=True,
                        help="en_train.json — ground truth 2026")
    parser.add_argument("--collection_json", required=True,
                        help="collection_data.json — corpus 2026")
    parser.add_argument("--nemotron_run",    required=True,
                        help="reranked_results_nemotron_topk400.json")

    # Corpus 2025 (opzionale)
    parser.add_argument("--train_tsv",          default=None,
                        help="en_train.tsv — ground truth 2025 (opzionale)")
    parser.add_argument("--collection_json2025", default=None,
                        help="collection_data2025.json — corpus 2025 (opzionale)")
    parser.add_argument("--nemotron_run2025",    default=None,
                        help="reranked_results_nemotron_2025.json (opzionale)")

    # Parametri
    parser.add_argument("--out_dir",     default="training_data")
    parser.add_argument("--num_hard_neg", type=int,   default=5)
    parser.add_argument("--neg_offset",   type=int,   default=0,
                        help="Salta i primi N negativi nel Nemotron run (default 0)")
    parser.add_argument("--dev_ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--max_doc_chars", type=int,  default=5000)
    args = parser.parse_args()

    cfg = Config(
        num_hard_neg=args.num_hard_neg,
        neg_offset=args.neg_offset,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        max_doc_chars=args.max_doc_chars,
    )

    random.seed(cfg.seed)

    all_examples = []

    # ── Corpus 2026 ──────────────────────────────────────────────────────────
    log.info("=== Caricamento corpus 2026 ===")
    corpus_2026   = load_collection_2026(args.collection_json, cfg.max_doc_chars)
    queries_2026  = load_queries_2026(args.train_json)
    gt_2026       = load_ground_truth_2026(args.train_json)
    nemotron_2026 = load_nemotron_run(args.nemotron_run)

    examples_2026 = build_examples(
        queries_2026, corpus_2026, gt_2026, nemotron_2026, cfg, split_tag="2026"
    )
    all_examples.extend(examples_2026)
    log.info(f"Esempi 2026: {len(examples_2026):,}")

    # ── Corpus 2025 (opzionale) ───────────────────────────────────────────────
    if args.train_tsv and args.collection_json2025:
        log.info("=== Caricamento corpus 2025 ===")
        corpus_2025          = load_collection_2025(args.collection_json2025, cfg.max_doc_chars)
        queries_2025, gt_2025 = load_ground_truth_2025(args.train_tsv)
        # Pulisci queries 2025
        queries_2025 = {qid: clean_tweet(text) for qid, text in queries_2025.items()}

        if args.nemotron_run2025:
            nemotron_2025 = load_nemotron_run(args.nemotron_run2025)
        else:
            # Senza Nemotron run 2025, non abbiamo hard negatives di qualità.
            # Usiamo comunque i dati ma con un Nemotron run vuoto
            # → build_examples skipperà queste query (no_nemotron_run).
            # Meglio essere conservativi che usare random negatives scadenti.
            log.warning(
                "Nessun Nemotron run 2025 fornito. "
                "I dati 2025 verranno usati solo se fornisci --nemotron_run2025. "
                "Consiglio: reranka il corpus 2025 con Nemotron prima di usarlo."
            )
            nemotron_2025 = {}

        examples_2025 = build_examples(
            queries_2025, corpus_2025, gt_2025, nemotron_2025, cfg, split_tag="2025"
        )
        all_examples.extend(examples_2025)
        log.info(f"Esempi 2025: {len(examples_2025):,}")
    else:
        log.info("Corpus 2025 non fornito, skip.")

    # ── Split e salvataggio ───────────────────────────────────────────────────
    if not all_examples:
        log.error("Nessun esempio costruito! Controlla i file di input.")
        return

    log.info(f"\nTotale esempi: {len(all_examples):,}")
    n_pos = sum(1 for ex in all_examples if ex["label"] == 1)
    n_neg = sum(1 for ex in all_examples if ex["label"] == 0)
    log.info(f"  Positivi: {n_pos:,} ({100*n_pos/len(all_examples):.1f}%)")
    log.info(f"  Negativi: {n_neg:,} ({100*n_neg/len(all_examples):.1f}%)")
    log.info(f"  Ratio neg/pos effettivo: {n_neg/max(1,n_pos):.1f}x")

    train_examples, dev_examples = split_by_query(all_examples, cfg.dev_ratio, cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_examples, out_dir / "train_pairs.jsonl")
    save_jsonl(dev_examples,   out_dir / "dev_pairs.jsonl")

    # Salva statistiche
    stats = {
        "total": len(all_examples),
        "train": len(train_examples),
        "dev":   len(dev_examples),
        "positives": n_pos,
        "negatives": n_neg,
        "ratio_neg_pos": round(n_neg / max(1, n_pos), 2),
        "corpus_2026_size": len(corpus_2026),
        "num_hard_neg": cfg.num_hard_neg,
        "neg_offset": cfg.neg_offset,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n✓ Dataset pronto in {out_dir}/")
    log.info(f"  train_pairs.jsonl: {len(train_examples):,} righe")
    log.info(f"  dev_pairs.jsonl:   {len(dev_examples):,} righe")
    log.info(
        f"\nProssimo step:\n"
        f"  python finetune_modernbert_crossencoder.py \\\n"
        f"    --train_jsonl {out_dir}/train_pairs.jsonl \\\n"
        f"    --dev_jsonl   {out_dir}/dev_pairs.jsonl \\\n"
        f"    --output_dir  models/modernbert-large-retrix-nemotron-distilled"
    )


if __name__ == "__main__":
    main()