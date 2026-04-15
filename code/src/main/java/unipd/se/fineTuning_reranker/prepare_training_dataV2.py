"""
prepare_training_data_v2.py

Versione migliorata della preparazione dati che sfrutta:
  1. en_train.json  — coppie (query_text, pubkey_positivo) ground truth
  2. reranked_results_nemotron_topkN.json — run Nemotron già rerankeato
  3. corpus TSV — testo dei documenti

Vantaggio rispetto alla versione BM25-only:
  - I negativi vengono dal run Nemotron (già filtrato su 400 candidati BM25)
  - I documenti PRIMA del positivo nel ranking sono "falsi positivi convincenti"
    → hard negatives di qualità molto superiore rispetto al BM25 grezzo
  - Il testo delle query è già in en_train.json, nessun join necessario

Strategia di campionamento:
  "above_positive"  → prende i documenti rankeati SOPRA il positivo nel run Nemotron
                      (i più fuorvianti, massima difficoltà)
  "mixed"           → metà above_positive + metà below_positive
                      (più varietà, training più stabile)
  "top_k"           → semplicemente i top-K dal run escludendo il positivo
                      (simile a BM25 ma su candidati già filtrati)

Uso:
  python prepare_training_data_v2.py \
    --train_json     en_train.json \
    --reranked_json  reranked_results_nemotron_topk400.json \
    --corpus         corpus.tsv \
    --out_dir        training_data_v2 \
    [--strategy above_positive|mixed|top_k] \
    [--num_hard_neg 5] \
    [--dev_ratio 0.1]
"""

import json
import random
import re
import logging
import argparse
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_HARD_NEG = 5
DEV_RATIO    = 0.1
RANDOM_SEED  = 42


# ── Preprocessing ──────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Caricamento ────────────────────────────────────────────────────────────────
def load_corpus(path: str) -> dict[str, str]:
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                corpus[parts[0]] = parts[1]
    log.info(f"Corpus: {len(corpus):,} documenti")
    return corpus


def load_train_json(path: str) -> list[dict]:
    """
    Carica en_train.json.
    Formato atteso: lista di oggetti con almeno:
      {"index": 0, "text": "...", "pubkey": 999}
    Ritorna lista ordinata per index.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Normalizza: accetta sia lista che dict {index: {...}}
    if isinstance(data, dict):
        data = [{"index": int(k), **v} for k, v in data.items()]

    data.sort(key=lambda x: x["index"])
    log.info(f"en_train.json: {len(data):,} esempi")
    return data


def load_reranked_results(path: str) -> dict[str, list[str]]:
    """
    Carica reranked_results_nemotron_topkN.json.
    Formato: {"0": ["doc_id1", "doc_id2", ...], "1": [...], ...}
    Chiave = query index (stringa), valore = lista doc_id ordinata per score DESC
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"Reranked results: {len(data):,} queries")
    return data


# ── Analisi qualità del run ────────────────────────────────────────────────────
def analyze_run_quality(
        train_data:      list[dict],
        reranked_run:    dict[str, list[str]],
        max_rank_check:  int = 50,
):
    """
    Analizza dove si trova il positivo nel run Nemotron.
    Utile per capire la qualità dei negativi disponibili.
    """
    ranks = []
    not_found = 0

    for item in train_data:
        qidx    = str(item["index"])
        pos_key = str(item["pubkey"])

        if qidx not in reranked_run:
            continue

        ranked_docs = reranked_run[qidx]
        try:
            rank = ranked_docs.index(pos_key) + 1  # 1-indexed
            ranks.append(rank)
        except ValueError:
            not_found += 1

    if not ranks:
        log.warning("Nessun positivo trovato nel run! Controlla i formati dei file.")
        return

    ranks.sort()
    n = len(ranks)
    log.info(f"\n{'='*50}")
    log.info(f"ANALISI QUALITÀ RUN NEMOTRON")
    log.info(f"{'='*50}")
    log.info(f"  Queries analizzate: {n:,}")
    log.info(f"  Positivi non trovati nel run: {not_found:,}")
    log.info(f"  Rank medio del positivo:    {sum(ranks)/n:.1f}")
    log.info(f"  Rank mediano del positivo:  {ranks[n//2]}")
    log.info(f"  Positivi al rank 1:         {sum(1 for r in ranks if r==1):,} ({100*sum(1 for r in ranks if r==1)/n:.1f}%)")
    log.info(f"  Positivi nei top-5:         {sum(1 for r in ranks if r<=5):,} ({100*sum(1 for r in ranks if r<=5)/n:.1f}%)")
    log.info(f"  Positivi nei top-10:        {sum(1 for r in ranks if r<=10):,} ({100*sum(1 for r in ranks if r<=10)/n:.1f}%)")
    log.info(f"  Positivi fuori top-50:      {sum(1 for r in ranks if r>50):,} ({100*sum(1 for r in ranks if r>50)/n:.1f}%)")
    log.info(f"{'='*50}\n")


# ── Costruzione dataset ────────────────────────────────────────────────────────
def build_examples(
        train_data:   list[dict],
        reranked_run: dict[str, list[str]],
        corpus:       dict[str, str],
        num_hard_neg: int,
        strategy:     str,
) -> list[dict]:
    """
    Costruisce esempi di training listwise.

    Strategie per i negativi:
      "above_positive": documenti rankeati sopra il positivo nel run Nemotron
                        → massima difficoltà, modello impara a superare Nemotron
      "mixed":          metà above + metà below
                        → più varietà, training più stabile
      "top_k":          top-K escludendo il positivo
                        → bilanciato, buona scelta di default
    """
    examples          = []
    skipped_not_found = 0
    skipped_no_neg    = 0
    skipped_missing   = 0
    used_above_only   = 0

    for item in train_data:
        qidx    = str(item["index"])
        pos_key = str(item["pubkey"])
        query   = clean_tweet(item["text"])

        if qidx not in reranked_run:
            skipped_not_found += 1
            continue

        if pos_key not in corpus:
            skipped_missing += 1
            continue

        ranked_docs = reranked_run[qidx]

        # Trova posizione del positivo nel ranking
        try:
            pos_rank = ranked_docs.index(pos_key)  # 0-indexed
        except ValueError:
            # Positivo non nel run → usa comunque l'esempio, tutti gli altri sono negativi
            pos_rank = len(ranked_docs)  # sentinella: "dopo tutto il run"

        # Documenti sopra il positivo (più difficili)
        above_pos = [
            d for d in ranked_docs[:pos_rank]
            if d != pos_key and d in corpus
        ]
        # Documenti sotto il positivo (più facili)
        below_pos = [
            d for d in ranked_docs[pos_rank + 1:]
            if d != pos_key and d in corpus
        ]

        # Seleziona negativi in base alla strategia
        if strategy == "above_positive":
            if len(above_pos) >= 2:
                # Prendi i più vicini al positivo (rank leggermente sopra = i più difficili)
                candidates = above_pos[-num_hard_neg:]  # ultimi = più vicini al positivo
                if len(candidates) < num_hard_neg and below_pos:
                    # Integra con below se non abbastanza above
                    candidates += below_pos[:num_hard_neg - len(candidates)]
            else:
                used_above_only += 1
                candidates = (above_pos + below_pos)[:num_hard_neg]

        elif strategy == "mixed":
            n_above = num_hard_neg // 2
            n_below = num_hard_neg - n_above
            # Above: i più vicini al positivo
            sel_above = above_pos[-n_above:] if above_pos else []
            # Below: i primi sotto il positivo (ancora relativamente difficili)
            sel_below = below_pos[:n_below] if below_pos else []
            candidates = sel_above + sel_below
            # Padding se non abbastanza
            if len(candidates) < num_hard_neg:
                all_others = [d for d in ranked_docs if d != pos_key and d in corpus]
                candidates = all_others[:num_hard_neg]

        else:  # "top_k"
            candidates = [d for d in ranked_docs if d != pos_key and d in corpus][:num_hard_neg]

        if len(candidates) < 2:
            skipped_no_neg += 1
            continue

        examples.append({
            "query":     query,
            "positive":  clean_tweet(corpus[pos_key]),
            "negatives": [clean_tweet(corpus[d]) for d in candidates],
            "qidx":      qidx,
            "pos_key":   pos_key,
            "pos_rank_in_nemotron": pos_rank + 1,  # per diagnostica
        })

    log.info(f"Esempi costruiti: {len(examples):,} (strategia: '{strategy}')")
    log.info(f"  Saltati (query non nel run):   {skipped_not_found}")
    log.info(f"  Saltati (positivo mancante):   {skipped_missing}")
    log.info(f"  Saltati (troppo pochi neg):    {skipped_no_neg}")
    if strategy == "above_positive":
        log.info(f"  Usato fallback above+below:    {used_above_only}")

    return examples


def build_labeled_pairs(examples: list[dict]) -> tuple[list, list, list]:
    """
    Converte esempi listwise in coppie (query, doc, label) per BCE loss.
    Ritorna (queries, passages, labels).
    """
    queries, passages, labels = [], [], []
    for ex in examples:
        queries.append(ex["query"])
        passages.append(ex["positive"])
        labels.append(1)
        for neg in ex["negatives"]:
            queries.append(ex["query"])
            passages.append(neg)
            labels.append(0)
    return queries, passages, labels


def split_and_save(
        examples:  list[dict],
        out_dir:   str,
        dev_ratio: float,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    random.shuffle(examples)
    n_dev = max(50, int(len(examples) * dev_ratio))
    dev_ex   = examples[:n_dev]
    train_ex = examples[n_dev:]

    # Salva formato listwise (per finetune_modernbert_reranker.py con InfoNCE)
    for split, data in [("train", train_ex), ("dev", dev_ex)]:
        path = out_path / f"{split}_pairs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                # Rimuovi campo diagnostico prima del salvataggio (opzionale)
                out = {k: v for k, v in ex.items() if k != "pos_rank_in_nemotron"}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        log.info(f"Salvato {split}_pairs.jsonl: {len(data):,} esempi → {path}")

    # Salva anche statistiche rank
    ranks = [ex["pos_rank_in_nemotron"] for ex in examples]
    stats = {
        "total_examples":     len(examples),
        "train_examples":     len(train_ex),
        "dev_examples":       len(dev_ex),
        "avg_pos_rank":       sum(ranks) / len(ranks),
        "median_pos_rank":    sorted(ranks)[len(ranks)//2],
        "pos_at_rank1_pct":   100 * sum(1 for r in ranks if r == 1) / len(ranks),
        "pos_in_top5_pct":    100 * sum(1 for r in ranks if r <= 5) / len(ranks),
    }
    with open(out_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats salvate in {out_path / 'dataset_stats.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepara dati da en_train.json + reranked Nemotron run"
    )
    parser.add_argument("--train_json",    required=True,
                        help="en_train.json con {index, text, pubkey}")
    parser.add_argument("--reranked_json", required=True,
                        help='reranked_results_nemotron_topkN.json con {"qidx": [doc_ids]}')
    parser.add_argument("--corpus",        required=True,
                        help="corpus.tsv (doc_id \\t text)")
    parser.add_argument("--out_dir",       default="training_data_v2")
    parser.add_argument("--strategy",
                        choices=["above_positive", "mixed", "top_k"],
                        default="mixed",
                        help=(
                            "above_positive: documenti sopra il positivo nel run Nemotron (più difficili); "
                            "mixed: metà sopra + metà sotto; "
                            "top_k: semplicemente top-K escludendo il positivo"
                        ))
    parser.add_argument("--num_hard_neg",  type=int, default=NUM_HARD_NEG)
    parser.add_argument("--dev_ratio",     type=float, default=DEV_RATIO)
    parser.add_argument("--seed",          type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    # Carica
    train_data   = load_train_json(args.train_json)
    reranked_run = load_reranked_results(args.reranked_json)
    corpus       = load_corpus(args.corpus)

    # Analizza qualità del run
    analyze_run_quality(train_data, reranked_run)

    # Costruisci esempi
    examples = build_examples(
        train_data, reranked_run, corpus,
        num_hard_neg=args.num_hard_neg,
        strategy=args.strategy,
    )

    # Salva
    split_and_save(examples, args.out_dir, args.dev_ratio)
    log.info("✓ Preparazione dati completata.")


if __name__ == "__main__":
    main()