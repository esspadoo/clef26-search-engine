"""
finetune_nemotron_reranker_v3.py

Fine-tuning di nvidia/llama-nemotron-rerank-1b-v2 con HuggingFace Trainer
nativo + LoRA (PEFT).

═══════════════════════════════════════════════════════════════════
CAMBIAMENTI RISPETTO A v2 — RIEPILOGO
═══════════════════════════════════════════════════════════════════

[FIX 1] modules_to_save=["score"] RIMOSSO
  → In v2 la head "score" veniva clonata e re-inizializzata da PEFT,
    perdendo la calibrazione pre-trained. Ora la head è congelata
    di default (solo LoRA aggiorna q/k/v/o).

[FIX 2] Learning rate abbassata: 2e-4 → 5e-5
  → Con LoRA rank=32 e alpha=64 il LR effettivo era troppo aggressivo.
    5e-5 è più conservativo e riduce il rischio di catastrophic forgetting.

[FIX 3] LoRA rank abbassato: 32 → 16, dropout alzato: 0.05 → 0.1
  → Meno parametri trainabili = meno overfitting.
    Dropout più alto = maggiore regolarizzazione.

[FIX 4] Loss: BCEWithLogitsLoss → PairwiseMarginLoss (con fallback a BCE)
  → BCE ottimizza la calibrazione assoluta dei logit (buona per classificazione).
    Per il ranking quello che conta è l'ORDINE relativo pos > neg.
    La MarginRankingLoss ottimizza direttamente: score(pos) - score(neg) > margin.
    Fallback a BCE se il batch non contiene sia pos che neg.

[FIX 5] load_best_model_at_end=True, save_total_limit=3
  → In v2 veniva salvato solo il modello finale (potenzialmente il peggiore).
    Ora si salva il checkpoint con eval_loss minima.

[FIX 6] Epochs ridotte: 3 → 2, eval_steps ridotti: 500 → 200
  → Meno epoche = meno rischio di overfitting.
    Eval più frequente = catch tempestivo della degradazione.

[FIX 7] weight_decay alzato: 0.01 → 0.05
  → Maggiore regolarizzazione L2 per limitare il drift dei pesi LoRA.

[FIX 8] warmup_ratio alzato: 0.1 → 0.15
  → Warm-up più lungo per stabilizzare il training nelle prime iterazioni.

[FIX 9] PairDataset ora supporta hard_neg separati da soft_neg
  → Se il JSONL ha un campo "hard_neg", quei negativi vengono campionati
    con probabilità doppia rispetto ai negativi normali.
    Hard negatives (es. BM25 top-k) sono molto più informativi per il ranking.

[FIX 10] Aggiunto EarlyStoppingCallback
  → Il training si ferma automaticamente se eval_loss non migliora
    per patience=3 eval consecutive. Previene il sovraallenamento.
═══════════════════════════════════════════════════════════════════
"""

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Dataset
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


class PairDataset(TorchDataset):
    """
    Dataset di coppie (query, passage) con label binaria.

    [FIX 9] Supporto hard_neg con campionamento pesato.
    Il campo "hard_neg" nel JSONL (es. top-k BM25 non rilevanti)
    viene incluso con peso doppio rispetto ai normali "neg".
    Hard negatives sono molto più informativi perché sono simili
    ai positivi ma non rilevanti — esattamente il caso difficile
    che il reranker deve saper gestire.

    Formato JSONL atteso:
      {"query": "...", "pos": ["doc1", ...], "neg": ["doc2", ...]}
      oppure con hard negatives:
      {"query": "...", "pos": [...], "neg": [...], "hard_neg": [...]}
    """

    def __init__(self, groups: list[dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs: list[tuple[str, str, float]] = []

        n_hard = 0
        for g in groups:
            query = g["query"]
            for doc in g.get("pos", []):
                self.pairs.append((query, doc, 1.0))
            for doc in g.get("neg", []):
                self.pairs.append((query, doc, 0.0))
            # [FIX 9] Hard negatives aggiunti due volte per oversampling
            for doc in g.get("hard_neg", []):
                self.pairs.append((query, doc, 0.0))
                self.pairs.append((query, doc, 0.0))  # peso doppio
                n_hard += 1

        if n_hard > 0:
            log.info(f"  → {n_hard:,} hard negatives inclusi (x2 oversampling)")
        log.info(f"PairDataset: {len(self.pairs):,} coppie totali")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        query, passage, label = self.pairs[idx]
        enc = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            truncation=True,
            padding=False,        # padding dinamico nel DataCollator
            return_tensors=None,  # liste Python, non tensori
        )
        enc["labels"] = label
        return enc


def compute_pos_weight(dataset: PairDataset) -> float:
    n_pos = sum(1 for _, _, label in dataset.pairs if label >= 1.0)
    n_neg = sum(1 for _, _, label in dataset.pairs if label < 1.0)
    ratio = n_neg / max(1, n_pos)
    log.info(f"pos_weight: {ratio:.2f}  ({n_neg:,} neg / {n_pos:,} pos)")
    return ratio


# ══════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════

def build_eval_samples(
        groups: list[dict],
        max_queries: int = 500,
        seed: int = 42,
) -> list[dict]:
    samples = []
    for g in groups:
        positives = set(g.get("pos", []))
        documents = g.get("pos", []) + g.get("neg", []) + g.get("hard_neg", [])
        if positives and len(documents) >= 2:
            samples.append({
                "query":     g["query"],
                "positives": positives,
                "documents": documents,
            })
    if len(samples) > max_queries:
        random.Random(seed).shuffle(samples)
        samples = samples[:max_queries]
    log.info(f"Eval samples: {len(samples):,} queries")
    return samples


def score_pairs_batched(
        model,
        tokenizer,
        pairs: list[tuple[str, str]],
        batch_size: int,
        max_length: int,
        device: torch.device,
) -> list[float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            queries  = [p[0] for p in batch_pairs]
            passages = [p[1] for p in batch_pairs]
            enc = tokenizer(
                queries,
                passages,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())
    return scores


def ndcg_at_k(relevances: list[int], k: int) -> float:
    k = min(k, len(relevances))
    dcg  = sum(relevances[i] / math.log2(i + 2) for i in range(k))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(relevances: list[int], k: int) -> float:
    for rank, rel in enumerate(relevances[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def evaluate_reranking(
        model,
        tokenizer,
        eval_samples: list[dict],
        batch_size: int,
        max_length: int,
        device: torch.device,
        ks: tuple[int, ...] = (5, 10),
        prefix: str = "",
) -> dict[str, float]:
    all_ndcg = {k: [] for k in ks}
    all_mrr  = {k: [] for k in ks}

    for sample in eval_samples:
        query     = sample["query"]
        docs      = sample["documents"]
        positives = sample["positives"]
        pairs  = [(query, doc) for doc in docs]
        logits = score_pairs_batched(
            model, tokenizer, pairs, batch_size, max_length, device
        )
        ranked = sorted(zip(logits, docs), key=lambda x: x[0], reverse=True)
        relevances = [1 if doc in positives else 0 for _, doc in ranked]
        for k in ks:
            all_ndcg[k].append(ndcg_at_k(relevances, k))
            all_mrr[k].append(mrr_at_k(relevances, k))

    metrics = {}
    for k in ks:
        metrics[f"{prefix}ndcg@{k}"] = float(np.mean(all_ndcg[k]))
        metrics[f"{prefix}mrr@{k}"]  = float(np.mean(all_mrr[k]))

    log.info("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


# ══════════════════════════════════════════════════════════════════
# NemotronTrainer — con PairwiseMarginLoss
# ══════════════════════════════════════════════════════════════════

class NemotronTrainer(Trainer):
    """
    Trainer custom con Pairwise Margin Loss + fallback a BCE.

    [FIX 4] PERCHÉ MARGIN LOSS INVECE DI BCE:
    La BCEWithLogitsLoss ottimizza la calibrazione ASSOLUTA dei logit:
    vuole che score(pos) → +∞ e score(neg) → -∞ in termini assoluti.
    Per il reranking quello che conta è l'ORDINE RELATIVO:
    score(pos) > score(neg) + margin.
    La MarginRankingLoss ottimizza direttamente questa distanza relativa,
    ed è quindi più adatta a un task di ranking.

    Implementazione:
      Per ogni batch, costruiamo tutte le coppie (pos_i, neg_j) e
      calcoliamo: loss = mean(max(0, margin - (score_pos - score_neg)))
      margin=1.0 è un valore standard per questo tipo di loss.

    Fallback a BCE:
      Se il batch contiene solo pos o solo neg (può capitare con batch
      piccoli), la margin loss non è applicabile: si cade su BCE.
    """

    def __init__(self, *args, pos_weight: float = 1.0, margin: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_value = pos_weight
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.dim() > 1:
            logits = logits.squeeze(-1)  # (batch,)

        pos_mask = labels >= 1.0
        neg_mask = labels < 1.0

        # [FIX 4] Pairwise margin loss se abbiamo sia pos che neg nel batch
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = logits[pos_mask]   # shape (n_pos,)
            neg_scores = logits[neg_mask]   # shape (n_neg,)

            # Broadcast su tutte le coppie pos×neg: shape (n_pos, n_neg)
            pos_exp = pos_scores.unsqueeze(1)
            neg_exp = neg_scores.unsqueeze(0)

            # max(0, margin - (pos - neg)): vogliamo pos - neg > margin
            pair_loss = torch.clamp(self.margin - (pos_exp - neg_exp), min=0.0)
            loss = pair_loss.mean()
        else:
            # Fallback a BCE se il batch è mono-label
            pw = torch.tensor(
                self.pos_weight_value,
                dtype=logits.dtype,
                device=logits.device,
            )
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════
# RerankingEvalCallback
# ══════════════════════════════════════════════════════════════════

class RerankingEvalCallback(TrainerCallback):
    """
    Callback che esegue la valutazione reranking ad ogni on_evaluate.
    Logga NDCG e MRR a più soglie k.
    """

    def __init__(
            self,
            eval_samples: list[dict],
            tokenizer,
            max_length: int,
            batch_size: int,
            device: torch.device,
            ks: tuple[int, ...] = (5, 10),
    ):
        self.eval_samples = eval_samples
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.batch_size   = batch_size
        self.device       = device
        self.ks           = ks

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        log.info(f"[Step {state.global_step}] Valutazione reranking:")
        evaluate_reranking(
            model=model,
            tokenizer=self.tokenizer,
            eval_samples=self.eval_samples,
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device,
            ks=self.ks,
            prefix=f"step{state.global_step}_",
        )


# ══════════════════════════════════════════════════════════════════
# LoRA
# ══════════════════════════════════════════════════════════════════

def apply_lora(model, lora_rank: int, lora_alpha: int, lora_dropout: float):
    """
    Applica LoRA al modello.

    [FIX 1] modules_to_save=["score"] RIMOSSO.
    In v2 PEFT clonava la head "score" e la re-inizializzava come
    parametro separato. Questo cancellava la calibrazione pre-trained
    della head, che è la parte più critica per il ranking.
    Ora la head rimane congelata (usa i pesi originali) e solo
    i moduli q/k/v/o vengono aggiornati via LoRA.

    [FIX 3] rank abbassato (32→16 default), dropout alzato (0.05→0.1).
    Meno parametri = meno overfitting.
    Dropout più alto = maggiore regolarizzazione dei pesi LoRA.
    """
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,   # [FIX 3]
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # modules_to_save=["score"],  # [FIX 1] RIMOSSO — head congelata
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    return model


# ══════════════════════════════════════════════════════════════════
# DataCollator
# ══════════════════════════════════════════════════════════════════

@dataclass
class PairCollator:
    """
    DataCollator con padding dinamico e conversione label in float tensor.
    Necessario perché DataCollatorWithPadding non gestisce le label float.
    """
    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict]) -> dict:
        labels = [float(f.pop("labels")) for f in features]
        base = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch = base(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        # Rimetti le label nelle features originali per non corrompere il dataset
        for f, label in zip(features, labels):
            f["labels"] = label
        return batch


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning llama-nemotron-rerank-1b-v2 — v3"
    )
    parser.add_argument("--train_file",       required=True)
    parser.add_argument("--dev_file",         required=True)
    parser.add_argument("--output_dir",       default="models/nemotron-rerank-1b-retrixV3")
    parser.add_argument("--model_name",       default="nvidia/llama-nemotron-rerank-1b-v2")
    parser.add_argument("--no_lora",          action="store_true")
    parser.add_argument("--lora_rank",        type=int,   default=16)     # [FIX 3] era 32
    parser.add_argument("--lora_alpha",       type=int,   default=32)     # [FIX 3] era 64
    parser.add_argument("--lora_dropout",     type=float, default=0.1)    # [FIX 3] era 0.05
    parser.add_argument("--margin",           type=float, default=1.0)    # [FIX 4] per margin loss
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--grad_accum",       type=int,   default=4)
    parser.add_argument("--epochs",           type=int,   default=1)      # [FIX 6] era 3
    parser.add_argument("--lr",               type=float, default=None)
    parser.add_argument("--max_length",       type=int,   default=1024)
    parser.add_argument("--warmup_ratio",     type=float, default=0.1)   # [FIX 8] era 0.1
    parser.add_argument("--weight_decay",     type=float, default=0.05)   # [FIX 7] era 0.01
    parser.add_argument("--eval_steps",       type=int,   default=100)    # [FIX 6] era 500
    parser.add_argument("--save_steps",       type=int,   default=100)    # [FIX 5] era 10000
    parser.add_argument("--patience",         type=int,   default=3)      # [FIX 10] early stopping
    parser.add_argument("--max_eval_queries", type=int,   default=500)
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    use_lora = not args.no_lora
    if args.lr is None:
        # [FIX 2] LR abbassato: 2e-4 → 5e-5 per LoRA, invariato per full FT
        args.lr = 2e-5 if use_lora else 5e-6

    log.info(f"Modalità: {'LoRA' if use_lora else 'Full fine-tuning'} | LR={args.lr}")
    log.info(f"  rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    log.info(f"  margin={args.margin}, epochs={args.epochs}, patience={args.patience}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 1. Tokenizer ───────────────────────────────────────────────
    log.info(f"Caricamento tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token impostato a eos_token")

    # ── 2. Modello ─────────────────────────────────────────────────
    log.info(f"Caricamento modello: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    log.info(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. LoRA ────────────────────────────────────────────────────
    if use_lora:
        log.info(
            f"Applicazione LoRA: rank={args.lora_rank}, "
            f"alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )
        model = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    else:
        log.info("Full fine-tuning (no LoRA)")
        try:
            model.gradient_checkpointing_enable()
            log.info("gradient_checkpointing abilitato")
        except Exception as e:
            log.warning(f"gradient_checkpointing non abilitabile: {e}")

    # ── 4. Dataset ─────────────────────────────────────────────────
    log.info("Caricamento dataset...")
    train_groups  = load_jsonl(args.train_file)
    dev_groups    = load_jsonl(args.dev_file)
    train_dataset = PairDataset(train_groups, tokenizer, args.max_length)
    dev_dataset   = PairDataset(dev_groups,   tokenizer, args.max_length)
    pos_weight    = compute_pos_weight(train_dataset)

    # ── 5. Eval samples ────────────────────────────────────────────
    eval_samples = build_eval_samples(
        dev_groups, max_queries=args.max_eval_queries, seed=args.seed
    )

    # ── 6. Baseline ────────────────────────────────────────────────
    log.info("Baseline pre-training:")
    model.to(device)
    baseline_metrics = evaluate_reranking(
        model=model,
        tokenizer=tokenizer,
        eval_samples=eval_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        prefix="baseline_",
    )

    # ── 7. Training args ───────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,          # [FIX 6] 2 invece di 3
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,                 # [FIX 2] 5e-5 invece di 2e-4
        warmup_ratio=args.warmup_ratio,        # [FIX 8] 0.15 invece di 0.1
        weight_decay=args.weight_decay,        # [FIX 7] 0.05 invece di 0.01
        fp16=False,
        bf16=True,
        dataloader_num_workers=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,            # [FIX 6] 200 invece di 500
        save_strategy="steps",
        save_steps=args.save_steps,            # [FIX 5] 200 invece di 10000
        save_total_limit=1,                    # [FIX 5] era 0 (nessun salvataggio intermedio)
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,
        seed=args.seed,
        run_name="nemotron-rerank-1b-retrix-v3",
        gradient_checkpointing=not use_lora,
        # [FIX 5] Salva il checkpoint con eval_loss minima
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_only_model=True,
        report_to="none",
    )

    # ── 8. Callbacks ────────────────────────────────────────────────
    callbacks = [
        # [FIX 10] Early stopping: ferma il training se eval_loss
        # non migliora per `patience` eval consecutive.
        # Previene l'overfitting e risparmia tempo di compute.
        EarlyStoppingCallback(early_stopping_patience=args.patience),

        # Callback di valutazione reranking (NDCG, MRR)
        RerankingEvalCallback(
            eval_samples=eval_samples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        ),
    ]

    # ── 9. Trainer ─────────────────────────────────────────────────
    trainer = NemotronTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=PairCollator(tokenizer=tokenizer),
        pos_weight=pos_weight,     # usato solo nel fallback BCE
        margin=args.margin,        # [FIX 4] per margin loss
        callbacks=callbacks,
    )

    log.info("Avvio training...")
    trainer.train()

    # ── 10. Valutazione finale ─────────────────────────────────────
    log.info("Valutazione finale (best checkpoint):")
    final_metrics = evaluate_reranking(
        model=trainer.model,       # usa il best model caricato da load_best_model_at_end
        tokenizer=tokenizer,
        eval_samples=eval_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        prefix="final_",
    )

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump({**baseline_metrics, **final_metrics}, f, indent=2)
    log.info(f"Metriche salvate in {output_dir / 'final_metrics.json'}")

    # ── 11. Salvataggio ────────────────────────────────────────────
    #
    # PERCHÉ QUESTO BLOCCO È CRITICO PER NEMOTRON:
    #
    # Nemotron usa trust_remote_code=True — significa che la sua architettura
    # è definita in file Python custom scaricati da HuggingFace Hub e cachati
    # localmente (modeling_nemotron.py, configuration_nemotron.py, ecc.).
    # Quando si salva con save_pretrained(), questi file NON vengono copiati
    # automaticamente nella directory di output, causando:
    #   - ValueError: Unrecognized model ... Should have a `model_type` key
    #   - Tokenizer che non si carica perché mancano i file custom
    #
    # La soluzione è copiare esplicitamente tutti i file custom dalla cache
    # HuggingFace nella directory finale, in modo che il modello sia
    # completamente auto-contenuto e caricabile senza connessione internet.
    #
    # PROBLEMA AGGIUNTIVO con merge_and_unload() + LoRA:
    # Dopo il merge, il modello risultante è un oggetto Python "nudo" che
    # ha perso il riferimento alla cache HF. save_pretrained() salva i pesi
    # ma NON i file custom dell'architettura. Bisogna copiarli a mano.

    import shutil
    from huggingface_hub import snapshot_download

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: scarica/localizza i file custom del modello base nella cache HF
    # (se già in cache non fa nessuna richiesta di rete)
    log.info("Localizzazione file custom architettura Nemotron...")
    try:
        base_model_cache = snapshot_download(
            repo_id=args.model_name,
            local_files_only=False,   # usa cache se disponibile, scarica se no
        )
        log.info(f"  Cache base model: {base_model_cache}")
    except Exception as e:
        log.warning(f"  snapshot_download fallito ({e}), provo cache locale...")
        # Fallback: cerca nella cache HF standard
        from huggingface_hub import hf_hub_download
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        # Cerca la directory del modello nella cache
        model_cache_name = args.model_name.replace("/", "--")
        base_model_cache = None
        for root, dirs, files in os.walk(cache_dir):
            if model_cache_name in root and "snapshots" in root:
                base_model_cache = root
                break
        if base_model_cache is None:
            log.error("Impossibile trovare la cache del modello base. "
                      "I file custom NON verranno copiati — il modello "
                      "potrebbe non caricarsi con trust_remote_code=True.")

    # Step 2: salva i pesi del modello fine-tunato
    if use_lora:
        log.info("Merging LoRA adapter nel modello base...")
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(str(final_dir))
            log.info("✓ Merge completato e pesi salvati")
        except Exception as e:
            log.error(f"Merge fallito: {e}. Salvo adapter separati.")
            trainer.model.save_pretrained(str(final_dir))
    else:
        trainer.model.save_pretrained(str(final_dir))

    # Step 3: salva il tokenizer dal modello base (più affidabile del tokenizer
    # della sessione corrente, che potrebbe avere stati inconsistenti dopo PEFT)
    log.info("Salvataggio tokenizer dal modello base...")
    try:
        from transformers import AutoTokenizer as _AutoTokenizer
        base_tokenizer = _AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        base_tokenizer.save_pretrained(str(final_dir))
        log.info("✓ Tokenizer salvato dal modello base")
    except Exception as e:
        log.warning(f"Tokenizer base non disponibile ({e}), uso tokenizer sessione...")
        tokenizer.save_pretrained(str(final_dir))

    # Step 4: copia i file custom dell'architettura dalla cache HF
    # Questi file sono necessari per trust_remote_code=True al momento
    # del caricamento — senza di essi AutoModelForSequenceClassification
    # non riconosce il model_type e crasha.
    if base_model_cache is not None:
        import os
        custom_extensions = (".py", ".json")
        copied = []
        skipped = []
        for fname in os.listdir(base_model_cache):
            src = os.path.join(base_model_cache, fname)
            dst = final_dir / fname
            if not os.path.isfile(src):
                continue
            # Copia tutti i .py (architettura custom) e i .json (config, tokenizer)
            # ma NON sovrascrivere i .json che abbiamo già salvato con i pesi
            # aggiornati (config.json ha num_labels=1, tokenizer_config.json aggiornato)
            if fname.endswith(".py"):
                shutil.copy2(src, dst)
                copied.append(fname)
            elif fname.endswith(".json") and not dst.exists():
                # Copia solo i .json mancanti (non sovrascrivere config.json salvato)
                shutil.copy2(src, dst)
                copied.append(fname)
            else:
                skipped.append(fname)
        log.info(f"  File custom copiati ({len(copied)}): {copied}")
        if skipped:
            log.info(f"  File skippati (già esistenti): {[f for f in skipped if f.endswith('.json')]}")
    else:
        log.warning("⚠  File custom architettura NON copiati. "
                    "Per caricare il modello usa:\n"
                    f"  AutoModelForSequenceClassification.from_pretrained(\n"
                    f"    '{final_dir}',\n"
                    f"    trust_remote_code=True,\n"
                    f"    config=AutoConfig.from_pretrained('{args.model_name}', trust_remote_code=True)\n"
                    f"  )")

    # Step 5: verifica che config.json abbia model_type
    config_path = final_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "model_type" not in cfg:
            log.warning("  config.json manca di 'model_type' — patch automatica...")
            # Carica config dal base e aggiorna con i nostri parametri
            from transformers import AutoConfig
            base_cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
            base_cfg.num_labels = 1
            base_cfg.pad_token_id = tokenizer.pad_token_id
            base_cfg.save_pretrained(str(final_dir))
            log.info("  ✓ config.json aggiornato con model_type corretto")
        else:
            log.info(f"  ✓ config.json OK (model_type='{cfg['model_type']}')")
    else:
        log.error("  config.json non trovato! Il modello non si caricherà.")

    log.info(f"✓ Modello finale completo salvato in {final_dir}")
    log.info(f"  Contenuto: {sorted(f.name for f in final_dir.iterdir())}")

    # ── 12. Riepilogo confronto baseline vs fine-tuned ─────────────
    log.info("\n" + "═" * 60)
    log.info("RIEPILOGO CONFRONTO")
    log.info("═" * 60)
    for k in (5, 10):
        b_ndcg = baseline_metrics.get(f"baseline_ndcg@{k}", float("nan"))
        f_ndcg = final_metrics.get(f"final_ndcg@{k}", float("nan"))
        b_mrr  = baseline_metrics.get(f"baseline_mrr@{k}", float("nan"))
        f_mrr  = final_metrics.get(f"final_mrr@{k}", float("nan"))
        delta_ndcg = f_ndcg - b_ndcg
        delta_mrr  = f_mrr  - b_mrr
        sign_ndcg  = "▲" if delta_ndcg >= 0 else "▼"
        sign_mrr   = "▲" if delta_mrr  >= 0 else "▼"
        log.info(
            f"  NDCG@{k}: baseline={b_ndcg:.4f}  fine-tuned={f_ndcg:.4f}  "
            f"{sign_ndcg}{abs(delta_ndcg):.4f}"
        )
        log.info(
            f"  MRR@{k}:  baseline={b_mrr:.4f}  fine-tuned={f_mrr:.4f}  "
            f"{sign_mrr}{abs(delta_mrr):.4f}"
        )
    log.info("═" * 60)

    log.info("\nPer usarlo a inference:")
    log.info("  from transformers import AutoTokenizer, AutoModelForSequenceClassification")
    log.info(f"  tokenizer = AutoTokenizer.from_pretrained('{final_dir}', trust_remote_code=True)")
    log.info(f"  model = AutoModelForSequenceClassification.from_pretrained('{final_dir}', trust_remote_code=True)")


if __name__ == "__main__":
    main()