"""
prepare_data_nemotron_finetune.py

Prepara il dataset per fine-tuning di nvidia/llama-nemotron-rerank-1b-v2.

Formato identico a prepare_data_unified_v3.py (BGE) con una differenza:
  - Nemotron NON usa prefissi né template speciali sulle query.
    L'inference originale passa semplicemente (query, document) come coppia.
  - Il formato output è {query, pos, neg} senza campo "prompt".

Nota sulla strategia di hard negatives:
  Qui il Nemotron run che usiamo come hard negatives è prodotto dal modello
  BASE (non fine-tuned). Il fine-tuning insegna al modello a correggere
  i propri errori sul dominio biomedico COVID — questo è il bootstrap
  più efficace possibile per questo modello specifico.

USO minimo (solo 2026):
  python prepare_data_nemotron_finetune.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400.json

USO completo:
  python prepare_data_nemotron_finetune.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400.json \
    --collection_2025     collection_data2025.json \
    --queries_2025        subtask4b_query_tweets_train.json \
                          subtask4b_query_tweets_dev.json \
                          subtask4b_query_tweets_test.json \
    --nemotron_runs_2025  nemotron_2025_train.json \
                          nemotron_2025_dev.json \
                          nemotron_2025_test.json
"""

import json, re, random, logging, argparse, unicodedata
from pathlib import Path
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Nemotron NON usa prefissi sulle query — solo pulizia del testo.


# ── Preprocessing ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Pulizia allineata con il pipeline Java/Lucene.
    Nessun prefisso — Nemotron non ne usa.
    """
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


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_queries_and_gt(path: str) -> tuple[dict[str, str], dict[str, str]]:
    data = json.load(open(path, encoding="utf-8"))
    queries, gt = {}, {}
    for item in data:
        qid          = str(item["index"])
        queries[qid] = clean_text(item["text"])
        gt[qid]      = str(item["pubkey"])
    log.info(f"    {Path(path).name}: {len(queries):,} query")
    return queries, gt


def load_corpus(path: str, max_chars: int, label: str = "") -> dict[str, str]:
    """
    Carica un corpus JSON dove ogni documento ha il campo "pubkey".
    Funziona sia per collection_data.json (pubkey int, es. 9394)
    che per collection_data2025.json (pubkey str, es. "umvrwgaw").
    Entrambi i corpus usano "pubkey" come chiave — non esiste "cord_uid".
    """
    data = json.load(open(path, encoding="utf-8"))
    c = {}
    for d in data:
        key  = str(d["pubkey"])
        text = _doc_text(d, max_chars)
        if text:
            c[key] = text
    log.info(f"  Corpus {label} ({Path(path).name}): {len(c):,} documenti")
    return c


def load_corpus_2026(path: str, max_chars: int) -> dict[str, str]:
    return load_corpus(path, max_chars, label="2026")


def load_corpus_2025(path: str, max_chars: int) -> dict[str, str]:
    return load_corpus(path, max_chars, label="2025")


def _doc_text(item: dict, max_chars: int) -> str:
    t = (item.get("title")    or "").strip()
    a = (item.get("abstract") or "").strip()
    return f"{t}. {a}".strip(". ")[:max_chars]


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    raw   = json.load(open(path, encoding="utf-8"))
    run   = {str(k): [str(d) for d in v] for k, v in raw.items()}
    total = sum(len(v) for v in run.values())
    log.info(f"    Nemotron run {Path(path).name}: {len(run):,} query, {total:,} doc")
    return run


# ── Costruzione gruppi ─────────────────────────────────────────────────────────

def build_groups(
    queries:      dict[str, str],
    corpus:       dict[str, str],
    gt:           dict[str, str],
    nemotron_run: dict[str, list[str]],
    num_hard_neg: int,
    neg_offset:   int,
    tag:          str,
) -> list[dict]:
    """
    Formato sentence-transformers CrossEncoderTrainer:
      {"query": str, "pos": [str], "neg": [str, ...]}

    I negativi sono i documenti che il modello Nemotron base rankava in
    alto ma che non sono il positivo corretto — esattamente i casi su
    cui il modello sbaglia e su cui deve imparare.
    """
    groups = []
    stats  = Counter()

    for qid, query_text in queries.items():
        pos_id = gt.get(qid)
        if pos_id is None:          stats["no_gt"]       += 1; continue
        if pos_id not in corpus:    stats["pos_missing"]  += 1; continue
        if qid not in nemotron_run: stats["no_run"]       += 1; continue

        hard_negs = [
            d for d in nemotron_run[qid]
            if d != pos_id and d in corpus
        ]
        if len(hard_negs) < 2:     stats["few_negs"]     += 1; continue

        selected = hard_negs[neg_offset : neg_offset + num_hard_neg]
        if len(selected) < num_hard_neg:
            selected = hard_negs[:num_hard_neg]

        groups.append({
            "query": query_text,
            "pos":   [corpus[pos_id]],
            "neg":   [corpus[d] for d in selected],
            "_qid":    f"{tag}_{qid}",
            "_pos_id": pos_id,
        })
        stats["ok"] += 1

    log.info(
        f"  [{tag}] {stats['ok']:,} gruppi "
        f"(1 pos + {num_hard_neg} neg) = "
        f"{stats['ok'] * (1 + num_hard_neg):,} esempi equivalenti  "
        f"| skip: no_gt={stats['no_gt']} pos_missing={stats['pos_missing']} "
        f"no_run={stats['no_run']} few_negs={stats['few_negs']}"
    )
    return groups


# ── Split train/dev ────────────────────────────────────────────────────────────

def split_2026_by_query(
    groups_2026: list[dict],
    dev_ratio:   float,
    seed:        int,
) -> tuple[list[dict], list[dict]]:
    by_qid = defaultdict(list)
    for g in groups_2026:
        by_qid[g["_qid"]].append(g)

    qids = list(by_qid.keys())
    random.Random(seed).shuffle(qids)

    n_dev      = max(10, int(len(qids) * dev_ratio))
    dev_qids   = set(qids[:n_dev])
    train_qids = set(qids[n_dev:])

    train = [g for qid in train_qids for g in by_qid[qid]]
    dev   = [g for qid in dev_qids   for g in by_qid[qid]]

    log.info(f"  Split 2026: {len(train):,} train / {len(dev):,} dev query")
    return train, dev


def save_jsonl(groups: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for g in groups:
            out = {k: v for k, v in g.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    log.info(f"  → {path.name}: {len(groups):,} gruppi")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Prepara dati per fine-tuning llama-nemotron-rerank-1b-v2"
    )
    g26 = p.add_argument_group("Corpus 2026 (obbligatorio)")
    g26.add_argument("--collection_2026",   required=True)
    g26.add_argument("--queries_2026",      required=True)
    g26.add_argument("--nemotron_run_2026", required=True)

    g25 = p.add_argument_group("Corpus 2025 (opzionale)")
    g25.add_argument("--collection_2025",    default=None)
    g25.add_argument("--queries_2025",       nargs="+", default=[])
    g25.add_argument("--nemotron_runs_2025", nargs="+", default=[])

    p.add_argument("--out_dir",        default="training_data_nemotron_ft")
    p.add_argument("--num_hard_neg",   type=int,   default=5)
    p.add_argument("--neg_offset",     type=int,   default=0)
    p.add_argument("--dev_ratio",      type=float, default=0.1)
    p.add_argument("--max_doc_chars",  type=int,   default=4000)
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    if args.queries_2025 and args.nemotron_runs_2025:
        if len(args.queries_2025) != len(args.nemotron_runs_2025):
            p.error(
                f"--queries_2025 ha {len(args.queries_2025)} file ma "
                f"--nemotron_runs_2025 ne ha {len(args.nemotron_runs_2025)}. "
                "Devono corrispondere 1:1."
            )

    random.seed(args.seed)
    n26, n25 = 0, 0

    log.info("══ Corpus 2026 ══════════════════════════════════════════════")
    corpus_2026           = load_corpus_2026(args.collection_2026, args.max_doc_chars)
    queries_2026, gt_2026 = load_queries_and_gt(args.queries_2026)
    nemotron_2026         = load_nemotron_run(args.nemotron_run_2026)
    groups_2026 = build_groups(
        queries_2026, corpus_2026, gt_2026, nemotron_2026,
        args.num_hard_neg, args.neg_offset, tag="2026"
    )
    train_2026, dev_2026 = split_2026_by_query(groups_2026, args.dev_ratio, args.seed)
    n26 = len(groups_2026)

    train_2025 = []
    if args.collection_2025 and args.queries_2025:
        log.info("══ Corpus 2025 ══════════════════════════════════════════════")
        if not args.nemotron_runs_2025:
            log.warning("⚠  --nemotron_runs_2025 non fornito: dati 2025 SKIPPATI.")
        else:
            corpus_2025 = load_corpus_2025(args.collection_2025, args.max_doc_chars)
            for q_path, n_path in zip(args.queries_2025, args.nemotron_runs_2025):
                stem = Path(q_path).stem
                log.info(f"  Dataset: {stem}")
                queries_q, gt_q = load_queries_and_gt(q_path)
                nemotron_q      = load_nemotron_run(n_path)
                groups_q = build_groups(
                    queries_q, corpus_2025, gt_q, nemotron_q,
                    args.num_hard_neg, args.neg_offset, tag=f"2025_{stem}"
                )
                train_2025.extend(groups_q)
            n25 = len(train_2025)

    final_train = train_2026 + train_2025
    final_dev   = dev_2026

    if not final_train:
        log.error("Nessun gruppo costruito. Controlla i file di input.")
        return

    random.shuffle(final_train)

    log.info("══ Composizione finale ══════════════════════════════════════")
    log.info(f"  Training: {len(final_train):,} gruppi  (2026={len(train_2026):,} + 2025={n25:,})")
    log.info(f"  Dev (solo 2026): {len(final_dev):,} gruppi")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(final_train, out_dir / "train_groups.jsonl")
    save_jsonl(final_dev,   out_dir / "dev_groups.jsonl")

    tgs = 1 + args.num_hard_neg
    stats = {
        "train_groups": len(final_train), "dev_groups": len(final_dev),
        "train_from_2026": len(train_2026), "train_from_2025": n25,
        "dev_from_2026_only": True,
        "num_hard_neg": args.num_hard_neg,
        "train_group_size_for_training": tgs,
        "neg_offset": args.neg_offset,
        "note": (
            "Nemotron NON usa prefissi query. Solo pulizia testo. "
            f"train_group_size={tgs}. Hard negatives = errori del modello base."
        ),
    }
    json.dump(stats, open(out_dir / "stats.json", "w"), indent=2)

    log.info(f"\n✓ Dataset pronto in '{out_dir}/'")
    log.info(f"  train_groups.jsonl : {len(final_train):,} righe")
    log.info(f"  dev_groups.jsonl   : {len(final_dev):,} righe")
    log.info(f"\nProssimo step:")
    log.info(f"  python finetune_nemotron_reranker_v2.py \\")
    log.info(f"    --train_file {out_dir}/train_groups.jsonl \\")
    log.info(f"    --dev_file   {out_dir}/dev_groups.jsonl \\")
    log.info(f"    --output_dir models/nemotron-rerank-1b-retrix")


if __name__ == "__main__":
    main()