"""
finetune_nemotron_reranker_v2.py

Fine-tuning di nvidia/llama-nemotron-rerank-1b-v2 con HuggingFace Trainer
nativo + LoRA (PEFT).

PERCHÉ NON USIAMO sentence-transformers CrossEncoderTrainer:
  sentence-transformers si aspetta che il modello ritorni un dict con chiave
  "scores" nel forward(). Nemotron ritorna un oggetto SequenceClassifierOutput
  con chiave "logits". Questo causa KeyError: 'scores' e, prima ancora,
  TypeError: embedding(): argument 'indices' must be Tensor, not BatchEncoding
  (perché PEFT non spacchetta il BatchEncoding automaticamente).
  Anziché wrappare CrossEncoder con patch fragili, usiamo direttamente
  HuggingFace Trainer, che è la scelta corretta per modelli custom con
  trust_remote_code.

ARCHITETTURA Nemotron:
  - Basato su Llama-3.2-1B con bidirectional attention modificata
  - Richiede trust_remote_code=True
  - AutoModelForSequenceClassification con head "score" (non "classifier")
  - Output: SequenceClassifierOutput con .logits di shape (batch, 1)
  - num_labels=1 → regressione/ranking con BCE loss

LoRA:
  - task_type=SEQ_CLS (corretto per AutoModelForSequenceClassification nativo)
  - modules_to_save=["score"] per aggiornare la head custom di Nemotron
  - target_modules: moduli q/k/v/o di Llama
  - merge_and_unload() prima del salvataggio finale

Loss: BCEWithLogitsLoss con pos_weight dinamico

Evaluator: NDCG@5, NDCG@10, MRR@5, MRR@10 implementati da zero

Uso:
  python finetune_nemotron_reranker_v2.py \\
    --train_file  training_data_nemotron_ft/train_groups.jsonl \\
    --dev_file    training_data_nemotron_ft/dev_groups.jsonl \\
    --output_dir  models/nemotron-rerank-1b-retrix

  # Full fine-tuning senza LoRA:
  python finetune_nemotron_reranker_v2.py \\
    --train_file  training_data_nemotron_ft/train_groups.jsonl \\
    --dev_file    training_data_nemotron_ft/dev_groups.jsonl \\
    --output_dir  models/nemotron-rerank-1b-retrix \\
    --no_lora --lr 5e-6
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


# ── Dataset ────────────────────────────────────────────────────────────────────

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
    Tokenizza on-the-fly per risparmiare memoria.
    """

    def __init__(self, groups: list[dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs: list[tuple[str, str, float]] = []

        for g in groups:
            query = g["query"]
            for doc in g["pos"]:
                self.pairs.append((query, doc, 1.0))
            for doc in g["neg"]:
                self.pairs.append((query, doc, 0.0))

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
            padding=False,       # il DataCollator farà il padding dinamico
            return_tensors=None, # ritorna liste Python, non tensori
        )
        enc["labels"] = label
        return enc


def compute_pos_weight(dataset: PairDataset) -> float:
    n_pos = sum(1 for _, _, label in dataset.pairs if label >= 1.0)
    n_neg = sum(1 for _, _, label in dataset.pairs if label < 1.0)
    ratio = n_neg / max(1, n_pos)
    log.info(f"pos_weight: {ratio:.2f}  ({n_neg:,} neg / {n_pos:,} pos)")
    return ratio


# ── Evaluator ─────────────────────────────────────────────────────────────────

def build_eval_samples(
    groups: list[dict],
    max_queries: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Costruisce samples per la valutazione reranking dai gruppi dev."""
    samples = []
    for g in groups:
        positives = set(g["pos"])
        documents = g["pos"] + g["neg"]
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
    """
    Calcola i logit di Nemotron per una lista di (query, doc) in batch.
    Ritorna una lista di float (logit grezzi, non sigmoid-izzati).
    """
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

            # Nemotron ritorna SequenceClassifierOutput con .logits
            out = model(**enc)
            logits = out.logits  # shape: (batch, 1) o (batch,)
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


# ── NemotronTrainer ────────────────────────────────────────────────────────────

class NemotronTrainer(Trainer):
    """
    Trainer custom con BCEWithLogitsLoss + pos_weight dinamico.
    Gestisce correttamente i logit di shape (batch, 1).
    """

    def __init__(self, *args, pos_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_value = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.dim() > 1:
            logits = logits.squeeze(-1)

        pw = torch.tensor(
            self.pos_weight_value,
            dtype=logits.dtype,
            device=logits.device,
        )
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels)
        return (loss, outputs) if return_outputs else loss


class RerankingEvalCallback(TrainerCallback):
    """
    Callback che esegue la valutazione reranking ad ogni on_evaluate.
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


# ── LoRA ───────────────────────────────────────────────────────────────────────

def apply_lora(model, lora_rank: int, lora_alpha: int):
    """
    Applica LoRA al modello HuggingFace nativo.
    task_type=SEQ_CLS è corretto perché lavoriamo con
    AutoModelForSequenceClassification direttamente (non via CrossEncoder).
    """
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # necessario per GC con PEFT
    return model


# ── DataCollator ───────────────────────────────────────────────────────────────

@dataclass
class PairCollator:
    """
    DataCollator con padding dinamico e conversione label in float tensor.
    """
    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict]) -> dict:
        # Estrai e rimuovi le label temporaneamente
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning llama-nemotron-rerank-1b-v2 con HF Trainer nativo"
    )
    parser.add_argument("--train_file",       required=True)
    parser.add_argument("--dev_file",         required=True)
    parser.add_argument("--output_dir",       default="models/nemotron-rerank-1b-retrix")
    parser.add_argument("--model_name",       default="nvidia/llama-nemotron-rerank-1b-v2")
    parser.add_argument("--no_lora",          action="store_true")
    parser.add_argument("--lora_rank",        type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=int,   default=32)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--grad_accum",       type=int,   default=4)
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--lr",               type=float, default=None)
    parser.add_argument("--max_length",       type=int,   default=512)
    parser.add_argument("--warmup_ratio",     type=float, default=0.1)
    parser.add_argument("--eval_steps",       type=int,   default=500)
    parser.add_argument("--save_steps",       type=int,   default=500)
    parser.add_argument("--max_eval_queries", type=int,   default=500)
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    use_lora = not args.no_lora
    if args.lr is None:
        args.lr = 2e-4 if use_lora else 5e-6

    log.info(f"Modalità: {'LoRA' if use_lora else 'Full fine-tuning'} | LR={args.lr}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 1. Tokenizer ───────────────────────────────────────────────────────────
    log.info(f"Caricamento tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token impostato a eos_token")

    # ── 2. Modello ─────────────────────────────────────────────────────────────
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

    # ── 3. LoRA ────────────────────────────────────────────────────────────────
    if use_lora:
        log.info(f"Applicazione LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
        model = apply_lora(model, args.lora_rank, args.lora_alpha)
    else:
        log.info("Full fine-tuning (no LoRA)")
        try:
            model.gradient_checkpointing_enable()
            log.info("gradient_checkpointing abilitato")
        except Exception as e:
            log.warning(f"gradient_checkpointing non abilitabile: {e}")

    # ── 4. Dataset ─────────────────────────────────────────────────────────────
    log.info("Caricamento dataset...")
    train_groups  = load_jsonl(args.train_file)
    dev_groups    = load_jsonl(args.dev_file)
    train_dataset = PairDataset(train_groups, tokenizer, args.max_length)
    dev_dataset   = PairDataset(dev_groups,   tokenizer, args.max_length)
    pos_weight    = compute_pos_weight(train_dataset)

    # ── 5. Eval samples ────────────────────────────────────────────────────────
    eval_samples = build_eval_samples(
        dev_groups, max_queries=args.max_eval_queries, seed=args.seed
    )

    # ── 6. Baseline ────────────────────────────────────────────────────────────
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

    # ── 7. Training args ───────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=True,
        dataloader_num_workers=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=0,
        logging_strategy="steps",
        save_only_model=True,
        logging_steps=100,
        logging_first_step=True,
        seed=args.seed,
        run_name="nemotron-rerank-1b-retrix",
        weight_decay=0.01,
        gradient_checkpointing=not use_lora,
        load_best_model_at_end=False,
        report_to="none",
    )

    # ── 8. Trainer ─────────────────────────────────────────────────────────────
    trainer = NemotronTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=PairCollator(tokenizer=tokenizer),
        pos_weight=pos_weight,
        callbacks=[
            RerankingEvalCallback(
                eval_samples=eval_samples,
                tokenizer=tokenizer,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
            )
        ],
    )

    log.info("Avvio training...")
    trainer.train()

    # ── 9. Valutazione finale ─────────────────────────────────────────────────
    log.info("Valutazione finale:")
    final_metrics = evaluate_reranking(
        model=model,
        tokenizer=tokenizer,
        eval_samples=eval_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        prefix="final_",
    )

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump({**baseline_metrics, **final_metrics}, f, indent=2)

    # ── 10. Salvataggio ───────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if use_lora:
        log.info("Merging LoRA adapter nel modello base...")
        try:
            model = model.merge_and_unload()
            log.info("✓ Merge completato")
        except Exception as e:
            log.error(f"Merge fallito: {e}. Salvo con adapter separati.")

    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"✓ Modello salvato in {final_dir}")

    log.info("\nPer usarlo a inference:")
    log.info("  from transformers import AutoTokenizer, AutoModelForSequenceClassification")
    log.info(f"  tokenizer = AutoTokenizer.from_pretrained('{final_dir}', trust_remote_code=True)")
    log.info(f"  model = AutoModelForSequenceClassification.from_pretrained('{final_dir}', trust_remote_code=True)")


if __name__ == "__main__":
    main()