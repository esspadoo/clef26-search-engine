"""
finetune_modernbert_crossencoder.py

Fine-tuning di ModernBERT-large come CrossEncoder reranker usando
il framework sentence-transformers, adattato al dominio tweet/social
del progetto seupd2526-retrix.

Differenze rispetto allo script del gruppo reference:
  1. ModernBERT-LARGE invece di base (più capacity)
  2. Hard negatives presi dal BM25 run esistente (domain-specific)
     invece di mine_hard_negatives con embedding model generico
  3. Preprocessing clean_tweet allineato al pipeline Java/Lucene
  4. Evaluator su dati reali del progetto invece di GooAQ
  5. Supporto opzionale a gradient_checkpointing per VRAM

Loss: BinaryCrossEntropyLoss con pos_weight=num_hard_negatives
  - Ogni coppia (query, doc) è un esempio binario: 1=rilevante, 0=non-rilevante
  - pos_weight bilancia il class imbalance (1 positivo vs N negativi)
  - Più semplice e stabile di InfoNCE su dataset medio-piccoli

Uso:
  python finetune_modernbert_crossencoder.py \
    --corpus     corpus.tsv \
    --queries    queries.tsv \
    --qrels      qrels.txt \
    --bm25_run   results/BM25TopK1000_run.txt \
    --output_dir models/modernbert-large-retrix-reranker \
    [--num_hard_neg 5] [--epochs 3] [--batch_size 16]

Dipendenze:
  pip install sentence-transformers>=3.0 datasets torch
"""

import logging
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Configurazione ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    model_name:       str   = "answerdotai/ModernBERT-large"
    max_length:       int   = 512
    num_hard_neg:     int   = 5       # hard negatives per positivo (come il reference)
    train_batch_size: int   = 16      # come il reference; riduci a 8 se OOM
    num_epochs:       int   = 3
    learning_rate:    float = 2e-5    # come il reference
    warmup_ratio:     float = 0.1
    bf16:             bool  = True    # L40S supporta bf16
    fp16:             bool  = False
    eval_steps:       int   = 500
    save_steps:       int   = 500
    save_total_limit: int   = 2
    logging_steps:    int   = 100
    dev_ratio:        float = 0.1     # frazione queries per dev set
    seed:             int   = 42
    # Hard negative sampling strategy:
    # "top"    = i più difficili dal BM25 (rank più alto ma non rilevanti)
    # "random" = random sampling tra i non-rilevanti recuperati
    neg_strategy:     str   = "top"
# ───────────────────────────────────────────────────────────────────────────────


# ── Preprocessing ──────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    """
    Preprocessing allineato con clean_tweet del pipeline Java/Lucene.
    Rimuove URL, anonimizza mention, normalizza hashtag e spazi.
    """
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Caricamento file ───────────────────────────────────────────────────────────
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
    log.info(f"Queries: {len(queries):,}")
    return queries


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qrels[parts[0]][parts[2]] = int(parts[3])
    log.info(f"Qrels: {len(qrels):,} queries con giudizi")
    return qrels


def load_run(path: str, top_k: int = 100) -> dict[str, list[str]]:
    """Carica run BM25, ritorna {qid: [doc_id ordinati per rank]}"""
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                run[parts[0]].append((int(parts[3]), parts[2]))
    result = {}
    for qid, items in run.items():
        items.sort()
        result[qid] = [d for _, d in items[:top_k]]
    log.info(f"BM25 run: {len(result):,} queries")
    return result


# ── Costruzione dataset ────────────────────────────────────────────────────────
def build_labeled_pair_dataset(
        queries:  dict[str, str],
        corpus:   dict[str, str],
        qrels:    dict[str, dict[str, int]],
        bm25_run: dict[str, list[str]],
        cfg:      Config,
        split:    str = "train",
) -> Dataset:
    """
    Costruisce un HuggingFace Dataset in formato 'labeled-pair':
      (query, passage, label)  con label ∈ {0, 1}

    Questo è il formato atteso da BinaryCrossEntropyLoss di sentence-transformers.

    Strategia negatives:
      - Positivi: doc con qrel >= 1
      - Hard negatives: doc recuperati da BM25 con qrel == 0
        (ordinati per rank BM25 = i più "difficili")
    """
    rows_query   = []
    rows_passage = []
    rows_label   = []

    skipped = 0

    for qid, query_text in queries.items():
        if qid not in qrels or qid not in bm25_run:
            continue

        doc_rels  = qrels[qid]
        retrieved = bm25_run[qid]

        positives = [d for d, r in doc_rels.items() if r >= 1 and d in corpus]
        hard_neg  = [d for d in retrieved if doc_rels.get(d, 0) == 0 and d in corpus]

        if not positives or len(hard_neg) < 2:
            skipped += 1
            continue

        for pos_id in positives:
            if pos_id not in corpus:
                continue

            # Seleziona N hard negatives
            if cfg.neg_strategy == "top":
                selected_neg = hard_neg[:cfg.num_hard_neg]
            else:
                k = min(cfg.num_hard_neg, len(hard_neg))
                selected_neg = random.sample(hard_neg, k)

            # Aggiungi riga positiva
            rows_query.append(query_text)
            rows_passage.append(corpus[pos_id])
            rows_label.append(1)

            # Aggiungi righe negative
            for neg_id in selected_neg:
                rows_query.append(query_text)
                rows_passage.append(corpus[neg_id])
                rows_label.append(0)

    log.info(
        f"Dataset {split}: {len(rows_label):,} coppie "
        f"({sum(rows_label):,} pos, {len(rows_label)-sum(rows_label):,} neg). "
        f"Queries saltate: {skipped}"
    )

    return Dataset.from_dict({
        "query":   rows_query,
        "passage": rows_passage,
        "label":   rows_label,
    })


def build_reranking_eval_samples(
        queries:  dict[str, str],
        corpus:   dict[str, str],
        qrels:    dict[str, dict[str, int]],
        bm25_run: dict[str, list[str]],
        rerank_depth: int = 30,
        max_queries:  int = 500,
) -> list[dict]:
    """
    Costruisce samples per CrossEncoderRerankingEvaluator.
    Formato:
      {"query": "...", "positive": ["doc1", ...], "documents": ["doc1", "doc2", ...]}
    """
    samples  = []
    qids     = [q for q in queries if q in qrels and q in bm25_run]
    # Limita per velocità di eval
    if len(qids) > max_queries:
        qids = random.sample(qids, max_queries)

    for qid in qids:
        doc_rels  = qrels[qid]
        retrieved = bm25_run[qid][:rerank_depth]

        positives = [
            corpus[d] for d in retrieved
            if doc_rels.get(d, 0) >= 1 and d in corpus
        ]
        documents = [corpus[d] for d in retrieved if d in corpus]

        if not positives or len(documents) < 2:
            continue

        samples.append({
            "query":     queries[qid],
            "positive":  positives,
            "documents": documents,
        })

    log.info(f"Eval samples costruiti: {len(samples):,} queries")
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    random.seed(args.seed if hasattr(args, "seed") else 42)
    cfg = Config(
        num_hard_neg=args.num_hard_neg,
        train_batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        model_name=args.model_name,
        neg_strategy=args.neg_strategy,
    )

    # ── 1. Carica dati ─────────────────────────────────────────────────────────
    log.info("Caricamento dati...")
    corpus  = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels   = load_qrels(args.qrels)
    bm25    = load_run(args.bm25_run, top_k=100)

    # Split train/dev per query ID
    all_qids   = list(queries.keys())
    random.shuffle(all_qids)
    n_dev      = max(50, int(len(all_qids) * cfg.dev_ratio))
    dev_qids   = set(all_qids[:n_dev])
    train_qids = set(all_qids[n_dev:])

    train_queries = {q: queries[q] for q in train_qids}
    dev_queries   = {q: queries[q] for q in dev_qids}

    log.info(f"Split: {len(train_queries):,} train / {len(dev_queries):,} dev queries")

    # ── 2. Costruisce dataset ──────────────────────────────────────────────────
    log.info("Costruzione dataset...")
    train_dataset = build_labeled_pair_dataset(
        train_queries, corpus, qrels, bm25, cfg, split="train"
    )
    # Dev dataset per loss durante il training
    dev_dataset = build_labeled_pair_dataset(
        dev_queries, corpus, qrels, bm25, cfg, split="dev"
    )

    # ── 3. Carica modello CrossEncoder ─────────────────────────────────────────
    log.info(f"Caricamento modello: {cfg.model_name}")
    model = CrossEncoder(
        cfg.model_name,
        max_length=cfg.max_length,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"ModernBERT-large fine-tuned on {Path(args.corpus).stem} corpus",
        ),
    )
    log.info(f"Model max_length: {model.max_length}, num_labels: {model.num_labels}")

    # ── 4. Loss: BCE con pos_weight ────────────────────────────────────────────
    # pos_weight = num_hard_neg bilancia il class imbalance
    # (1 positivo ogni num_hard_neg negativi)
    loss = BinaryCrossEntropyLoss(
        model=model,
        pos_weight=torch.tensor(float(cfg.num_hard_neg)),
    )
    log.info(f"Loss: BinaryCrossEntropy con pos_weight={cfg.num_hard_neg}")

    # ── 5. Evaluator su dati reali ─────────────────────────────────────────────
    log.info("Costruzione evaluator...")
    eval_samples = build_reranking_eval_samples(
        dev_queries, corpus, qrels, bm25,
        rerank_depth=30, max_queries=500,
    )
    evaluator = CrossEncoderRerankingEvaluator(
        samples=eval_samples,
        batch_size=cfg.train_batch_size,
        name="retrix-dev",
        always_rerank_positives=False,
    )

    # Esegui evaluator sul modello base (baseline pre-training)
    log.info("Valutazione baseline (modello non fine-tuned)...")
    evaluator(model)

    # ── 6. Training arguments ──────────────────────────────────────────────────
    short_name = cfg.model_name.split("/")[-1]
    run_name   = f"reranker-{short_name}-retrix"
    output_dir = Path(args.output_dir)

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.train_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        dataloader_num_workers=4,
        # Salva e carica il miglior modello (su NDCG@10 del dev)
        load_best_model_at_end=True,
        metric_for_best_model="eval_retrix-dev_ndcg@10",
        greater_is_better=True,
        # Eval e save strategy
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        # Riproducibilità
        seed=cfg.seed,
        run_name=run_name,
        # Gradient checkpointing per ridurre VRAM (utile su large)
        gradient_checkpointing=True,
        # Weight decay leggero
        weight_decay=0.01,
    )

    # ── 7. Trainer ─────────────────────────────────────────────────────────────
    log.info("Avvio training...")
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # ── 8. Valutazione finale ──────────────────────────────────────────────────
    log.info("Valutazione finale del modello best...")
    final_metrics = evaluator(model)
    log.info(f"Metriche finali: {final_metrics}")

    # ── 9. Salvataggio ─────────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    log.info(f"✓ Modello salvato in {final_dir}")

    # Salva anche metriche finali
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    return model


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning ModernBERT-large CrossEncoder reranker"
    )
    parser.add_argument("--corpus",      required=True, help="corpus.tsv  (doc_id \\t text)")
    parser.add_argument("--queries",     required=True, help="queries.tsv (query_id \\t text)")
    parser.add_argument("--qrels",       required=True, help="qrels TREC format")
    parser.add_argument("--bm25_run",    required=True, help="BM25 run TREC format")
    parser.add_argument("--output_dir",  default="models/modernbert-large-retrix-reranker")
    parser.add_argument("--model_name",  default="answerdotai/ModernBERT-large")
    parser.add_argument("--num_hard_neg", type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--neg_strategy", choices=["top", "random"], default="top",
                        help="'top'=hard negatives dal BM25, 'random'=sampling casuale")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    main(args)