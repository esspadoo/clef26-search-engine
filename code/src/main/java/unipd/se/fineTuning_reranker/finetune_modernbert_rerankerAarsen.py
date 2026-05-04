"""
finetune_modernbert_reranker.py

Fine-tuning di Alibaba-NLP/gte-reranker-modernbert-base con
sentence-transformers v4 CrossEncoderTrainer.

Segue la pipeline descritta in:
  https://huggingface.co/blog/train-reranker

═══════════════════════════════════════════════════════════════════
PERCHÉ sentence-transformers v4 e NON HF Trainer nativo:
  - gte-reranker-modernbert-base è un AutoModelForSequenceClassification
    standard, senza trust_remote_code e senza architettura custom.
    Il CrossEncoderTrainer di ST v4 lo supporta nativamente.
  - CrossEncoderRerankingEvaluator fornisce MRR e NDCG durante il
    training senza scrivere codice custom.
  - BinaryCrossEntropyLoss con labeled-pair è la scelta raccomandata
    dall'articolo HF per questo tipo di task.

PERCHÉ BinaryCrossEntropyLoss e NON ListNet/MarginLoss:
  - L'articolo HF dice esplicitamente di non sottovalutare BCE —
    è forte, stabile e la più testata su corpus biomedici.
  - Con il nostro dataset (1 pos + hard neg + soft neg per query)
    BCE con labeled-pair è la scelta naturale: ogni coppia ha label
    float 1.0 (positivo) o 0.0 (negativo).
  - ListNet richiede che ogni batch contenga query complete con tutti
    i loro candidati — difficile da garantire con il nostro JSONL.

DATASET ATTESO (output di prepare_data_nemotron_finetune_v2.py):
  JSONL con campi: query, pos, neg, hard_neg (opzionale)
  Viene convertito internamente in labeled-pair format:
    (query, pos_doc, 1.0) e (query, neg_doc, 0.0)

STRATEGIA NEGATIVI:
  - hard_neg (rank 1-5 del reranker base): peso x2 (oversampling)
  - neg (rank 10-15 del reranker base): peso x1
  Il mix hard+soft stabilizza il training: hard neg = segnale
  informativo, soft neg = gradiente stabile nelle prime iterazioni.

USO:
  python finetune_modernbert_rerankerAarsen.py \
    --train_file training_data_nemotron_ft_v2/train_groups.jsonl \
    --dev_file   training_data_nemotron_ft_v2/dev_groups.jsonl \
    --output_dir models/gte-modernbert-base-retrixAarsen

  # Con tutti i parametri espliciti:
  python finetune_modernbert_reranker.py \
    --train_file training_data_nemotron_ft_v2/train_groups.jsonl \
    --dev_file   training_data_nemotron_ft_v2/dev_groups.jsonl \
    --output_dir models/gte-modernbert-base-retrix \
    --model_name Alibaba-NLP/gte-reranker-modernbert-base \
    --epochs 2 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 1024

REQUISITI:
  pip install sentence-transformers>=4.0.0 datasets
"""

import json
import logging
import random
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Caricamento JSONL
# ══════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Caricato {path}: {len(rows):,} gruppi")
    return rows


# ══════════════════════════════════════════════════════════════════
# Conversione gruppi → labeled-pair Dataset
# ══════════════════════════════════════════════════════════════════

def groups_to_labeled_pairs(
        groups: list[dict],
        hard_neg_oversample: int = 2,
        seed: int = 42,
) -> Dataset:
    """
    Converte il formato JSONL {query, pos, neg, hard_neg} nel formato
    labeled-pair richiesto da BinaryCrossEntropyLoss:
      (query, document, label)  dove label ∈ {0.0, 1.0}

    STRATEGIA OVERSAMPLING hard_neg:
      I hard negatives (top-k errori del reranker base) vengono inclusi
      `hard_neg_oversample` volte. Questo li rende più frequenti nel
      training senza aumentare il numero di query necessarie.
      Default: x2 — bilanciamento empiricamente buono tra segnale
      informativo e rischio di overfitting sui negativi difficili.

    PERCHÉ labeled-pair e non triplet/group:
      - Compatibile con BinaryCrossEntropyLoss (la loss raccomandata HF)
      - Flessibile: ogni coppia è indipendente, il batch sampler può
        mescolare liberamente positivi e negativi di query diverse
      - Semplice da debuggare: label 1.0/0.0 chiari e verificabili
    """
    queries   = []
    documents = []
    labels    = []

    rng = random.Random(seed)

    n_pos       = 0
    n_hard_neg  = 0
    n_soft_neg  = 0

    for g in groups:
        query = g["query"]

        # Positivi — label 1.0
        for doc in g.get("pos", []):
            queries.append(query)
            documents.append(doc)
            labels.append(1.0)
            n_pos += 1

        # Hard negatives — label 0.0, oversampling x hard_neg_oversample
        for doc in g.get("hard_neg", []):
            for _ in range(hard_neg_oversample):
                queries.append(query)
                documents.append(doc)
                labels.append(0.0)
            n_hard_neg += 1

        # Soft negatives — label 0.0, peso normale
        for doc in g.get("neg", []):
            queries.append(query)
            documents.append(doc)
            labels.append(0.0)
            n_soft_neg += 1

    log.info(
        f"  Coppie totali: {len(queries):,}  "
        f"(pos={n_pos:,} | hard_neg={n_hard_neg:,} x{hard_neg_oversample} | "
        f"soft_neg={n_soft_neg:,})"
    )
    log.info(
        f"  Ratio neg/pos: {(n_hard_neg * hard_neg_oversample + n_soft_neg) / max(1, n_pos):.1f}x"
    )

    # Shuffle per mescolare positivi e negativi di query diverse
    # (evita che il modello veda tutti i pos di seguito)
    indices = list(range(len(queries)))
    rng.shuffle(indices)

    return Dataset.from_dict({
        "query":    [queries[i]    for i in indices],
        "document": [documents[i]  for i in indices],
        "label":    [labels[i]     for i in indices],
    })


# ══════════════════════════════════════════════════════════════════
# Costruzione evaluator
# ══════════════════════════════════════════════════════════════════

def build_reranking_evaluator(
        dev_groups:  list[dict],
        max_queries: int = 500,
        seed:        int = 42,
        name:        str = "dev",
) -> CrossEncoderRerankingEvaluator:
    """
    Costruisce un CrossEncoderRerankingEvaluator dal formato JSONL.

    CrossEncoderRerankingEvaluator si aspetta una lista di dict:
      {"query": str, "positive": [str, ...], "negative": [str, ...]}

    Nota: usa "positive" (singolare) come chiave, non "pos".
    Il campo "negative" include sia hard_neg che neg — l'evaluator
    non distingue tra i due tipi, valuta solo il ranking finale.

    Metriche prodotte: MRR@10, NDCG@10 (default ST v4).
    """
    samples = []
    for g in dev_groups:
        positives = g.get("pos", [])
        negatives = g.get("hard_neg", []) + g.get("neg", [])
        if not positives or not negatives:
            continue
        samples.append({
            "query":    g["query"],
            "positive": positives,       # lista di documenti rilevanti
            "negative": negatives,       # lista di documenti non rilevanti
        })

    if len(samples) > max_queries:
        random.Random(seed).shuffle(samples)
        samples = samples[:max_queries]

    log.info(f"  Evaluator '{name}': {len(samples):,} query")

    return CrossEncoderRerankingEvaluator(
        samples=samples,
        name=name,
        # at_k=10 è il default — valuta MRR@10 e NDCG@10
        # write_csv=True scrive i risultati su file nella output_dir
        write_csv=True,
    )


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning gte-reranker-modernbert-base con ST v4"
    )
    parser.add_argument("--train_file",   required=True,
                        help="JSONL con {query, pos, neg, hard_neg}")
    parser.add_argument("--dev_file",     required=True,
                        help="JSONL dev per valutazione durante training")
    parser.add_argument("--output_dir",   default="models/gte-modernbert-base-retrix")
    parser.add_argument("--model_name",   default="Alibaba-NLP/gte-reranker-modernbert-base")
    #parser.add_argument("--model_name",   default="answerdotai/ModernBERT-large")
    parser.add_argument("--epochs",       type=int,   default=1,
                        help="Epoche di training (default: 1). "
                             "Con ~29k gruppi, 2 epoche sono solitamente sufficienti.")
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Batch size per device (default: 16). "
                             "Con L40S 48GB puoi alzare fino a 32 con max_length=512.")
    parser.add_argument("--grad_accum",   type=int,   default=2,
                        help="Gradient accumulation steps (default: 2). "
                             "Effective batch size = batch_size * grad_accum = 32.")
    parser.add_argument("--lr",           type=float, default=2e-5,
                        help="Learning rate (default: 2e-5). "
                             "Valore raccomandato dall'articolo HF per ModernBERT.")
    parser.add_argument("--max_length",   type=int,   default=1024,
                        help="Lunghezza massima token (default: 1024). "
                             "Allineato con il reranker base non fine-tunato.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio (default: 0.1).")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01).")
    parser.add_argument("--eval_steps",   type=int,   default=100,
                        help="Frequenza valutazione in step (default: 200).")
    parser.add_argument("--hard_neg_oversample", type=int, default=2,
                        help="Moltiplicatore oversampling hard_neg (default: 2).")
    parser.add_argument("--max_eval_queries",    type=int, default=500,
                        help="Max query per l'evaluator (default: 500).")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info(f"Modello:    {args.model_name}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Epochs:     {args.epochs}")
    log.info(f"LR:         {args.lr}")
    log.info(f"Batch:      {args.batch_size} × grad_accum={args.grad_accum} "
             f"= {args.batch_size * args.grad_accum} effective")
    log.info(f"Max length: {args.max_length}")
    log.info(f"Hard neg oversample: {args.hard_neg_oversample}x")
    log.info("═" * 60)

    # ── 1. Carica il modello con CrossEncoder ST v4 ────────────────
    # CrossEncoder wrappa AutoModelForSequenceClassification con num_labels=1.
    # gte-reranker-modernbert-base è compatibile nativamente — nessun
    # trust_remote_code necessario, nessun template manuale sulle query.
    # max_length viene passato al tokenizer internamente dal trainer.
    log.info(f"Caricamento modello: {args.model_name}")
    model = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=args.max_length,
        # automodel_args per caricare in bfloat16 — L40S supporta BF16
        # e ModernBERT è stato pre-trainato con BF16
        automodel_args={"torch_dtype": torch.bfloat16},
        default_activation_function=torch.nn.Sigmoid(),
        # device viene rilevato automaticamente (cuda se disponibile)
    )
    log.info(f"  Parametri: {sum(p.numel() for p in model.model.parameters()):,}")

    # ── 2. Carica e converte i dataset ────────────────────────────
    log.info("Caricamento dataset...")
    train_groups = load_jsonl(args.train_file)
    dev_groups   = load_jsonl(args.dev_file)

    log.info("Conversione train → labeled-pair...")
    train_dataset = groups_to_labeled_pairs(
        train_groups,
        hard_neg_oversample=args.hard_neg_oversample,
        seed=args.seed,
    )

    log.info("Conversione dev → labeled-pair (per eval_loss)...")
    dev_dataset = groups_to_labeled_pairs(
        dev_groups,
        hard_neg_oversample=1,   # no oversampling per il dev — valutazione neutra
        seed=args.seed,
    )

    log.info(f"  Train: {len(train_dataset):,} coppie")
    log.info(f"  Dev:   {len(dev_dataset):,} coppie")

    # ── 3. Evaluator reranking ────────────────────────────────────
    # CrossEncoderRerankingEvaluator valuta MRR@10 e NDCG@10 su un
    # subset del dev set — più informativo della sola eval_loss.
    log.info("Costruzione evaluator reranking...")
    reranking_evaluator = build_reranking_evaluator(
        dev_groups   = dev_groups,
        max_queries  = args.max_eval_queries,
        seed         = args.seed,
        name         = "retrix-dev",
    )

    # ── 4. Loss function ──────────────────────────────────────────
    # BinaryCrossEntropyLoss è la loss raccomandata dall'articolo HF
    # per il formato labeled-pair con label float {0.0, 1.0}.
    # Internamente fa: BCE(sigmoid(logit), label)
    # Compatibile con il modello che già usa sigmoid() come activation.
    #
    # PERCHÉ NON ListNet/CachedMNRL:
    #   ListNet richiede che il batch contenga query complete — difficile
    #   con il nostro shuffle che mescola coppie di query diverse.
    #   BCE è più robusta al batching casuale.
    loss = BinaryCrossEntropyLoss(model=model)
    log.info(f"Loss: {loss.__class__.__name__}")

    # ── 5. Training arguments ─────────────────────────────────────
    training_args = CrossEncoderTrainingArguments(
        output_dir             = str(output_dir),
        num_train_epochs       = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate          = args.lr,
        warmup_ratio           = args.warmup_ratio,
        weight_decay           = args.weight_decay,
        fp16                   = False,
        bf16                   = True,       # L40S supporta BF16 nativo
        dataloader_num_workers = 3,
        eval_strategy          = "steps",
        eval_steps             = args.eval_steps,
        save_strategy          = "steps",
        save_steps             = args.eval_steps,
        save_total_limit       = 2,
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,
        logging_steps          = 50,
        logging_first_step     = True,
        seed                   = args.seed,
        run_name               = "gte-modernbert-base-retrix",
        report_to              = "none",
    )

    # ── 6. Trainer ────────────────────────────────────────────────
    # CrossEncoderTrainer gestisce automaticamente:
    #   - Tokenizzazione delle coppie (query, document) con pair-mode
    #   - Chiamata all'evaluator ogni eval_steps
    #   - Salvataggio del best checkpoint
    #   - Early stopping (se aggiunto come callback)
    trainer = CrossEncoderTrainer(
        model       = model,
        args        = training_args,
        train_dataset = train_dataset,
        eval_dataset  = dev_dataset,
        loss          = loss,
        evaluator     = reranking_evaluator,
    )

    # ── 7. Baseline pre-training ──────────────────────────────────
    log.info("Baseline pre-training (modello non fine-tunato):")
    baseline_results = reranking_evaluator(model, output_path=str(output_dir))
    log.info(f"  Baseline: {baseline_results}")

    # ── 8. Training ───────────────────────────────────────────────
    log.info("Avvio training...")
    trainer.train()

    # ── 9. Valutazione finale ─────────────────────────────────────
    log.info("Valutazione finale (best checkpoint):")
    final_results = reranking_evaluator(trainer.model, output_path=str(output_dir))
    log.info(f"  Finale: {final_results}")

    # Riepilogo confronto
    log.info("\n" + "═" * 60)
    log.info("RIEPILOGO CONFRONTO BASELINE vs FINE-TUNED")
    log.info("═" * 60)
    for k, v in final_results.items():
        base_v = baseline_results.get(k, float("nan"))
        delta  = v - base_v
        sign   = "▲" if delta >= 0 else "▼"
        log.info(f"  {k}: baseline={base_v:.4f}  fine-tuned={v:.4f}  {sign}{abs(delta):.4f}")
    log.info("═" * 60)

    # ── 10. Salvataggio finale ────────────────────────────────────
    # CrossEncoder.save_pretrained() salva:
    #   - Il modello HF (config.json, model.safetensors)
    #   - Il tokenizer
    #   - La configurazione CrossEncoder (sentence_bert_config.json)
    # Non ci sono problemi di trust_remote_code o file custom mancanti
    # perché gte-reranker-modernbert-base è un modello standard HF.
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    log.info(f"✓ Modello salvato in {final_dir}")
    log.info(f"  Contenuto: {sorted(f.name for f in final_dir.iterdir())}")

    log.info("\nPer usarlo a inference:")
    log.info("  from sentence_transformers import CrossEncoder")
    log.info(f"  model = CrossEncoder('{final_dir}')")
    log.info("  scores = model.predict([['query', 'doc1'], ['query', 'doc2']])")


if __name__ == "__main__":
    main()