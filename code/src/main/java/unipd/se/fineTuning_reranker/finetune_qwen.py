"""
finetune_qwen3_reranker.py — Fine-tuning Qwen3-Reranker con LoRA
=================================================================
Addestra Qwen3-Reranker-{0.6B|4B} (AutoModelForCausalLM) con LoRA
usando la stessa loss yes/no del meccanismo di inference.

Legge:
  - data/finetune_reranker/train.jsonl
  - data/finetune_reranker/dev.jsonl

Scrive:
  - models/ft_qwen3_reranker_{run_name}/  (checkpoints + adapter)
  - results/finetune_eval_{run_name}.json (metriche dev per ogni eval step)

Uso rapido (0.6B per iterare veloce):
  python finetune_qwen3_reranker.py

Uso per 4B (stesso codice, più batch_size ridotto):
  python finetune_qwen3_reranker.py \\
    --model Qwen/Qwen3-Reranker-4B \\
    --batch_size 2 \\
    --grad_accum 16 \\
    --lora_r 16 \\
    --run_name ft_4B_r16

Uso multi-GPU con torchrun:
  torchrun --nproc_per_node=2 finetune_qwen3_reranker.py \\
    --model Qwen/Qwen3-Reranker-4B \\
    --batch_size 4

Note architetturali:
  - 0.6B e 4B hanno IDENTICO meccanismo yes/no → stesso codice
  - Cambia solo: MODEL_NAME, lora_r, batch_size, grad_accum
  - La TASK_INSTRUCTION qui deve essere IDENTICA a quella in CausalReranker.py
  - LoRA è consigliato anche per 0.6B: converge più veloce, meno overfitting
"""

import json
import os
import argparse
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

# Modello — cambia SOLO questo per passare da 0.6B a 4B
parser.add_argument("--model",       default="Qwen/Qwen3-Reranker-0.6B",
                    help="HuggingFace model ID. "
                         "Per 4B: 'Qwen/Qwen3-Reranker-4B'")

# Dati
parser.add_argument("--train_data",  default="../../../../../../../data/finetune_reranker/train.jsonl")
parser.add_argument("--dev_data",    default="../../../../../../../data/finetune_reranker/dev.jsonl")
parser.add_argument("--output_dir",  default=None,
                    help="Directory checkpoint (default: models/ft_qwen3_<run_name>)")
parser.add_argument("--results_dir", default="../../../../../../../results",
                    help="Directory dove salvare i file JSON delle metriche")
parser.add_argument("--run_name",    default=None,
                    help="Nome run (default: derivato dal modello)")

# Training
parser.add_argument("--epochs",      type=int,   default=3)
parser.add_argument("--batch_size",  type=int,   default=8,
                    help="Per device. 0.6B: 8-16 | 4B: 2-4")
parser.add_argument("--grad_accum",  type=int,   default=4,
                    help="Gradient accumulation steps. "
                         "Effective batch = batch_size * grad_accum. "
                         "0.6B: 4 | 4B: 8-16")
parser.add_argument("--lr",          type=float, default=1e-4,
                    help="Learning rate per LoRA (default: 1e-4)")
parser.add_argument("--warmup_ratio", type=float, default=0.10)
parser.add_argument("--max_length",  type=int,   default=512,
                    help="Max token length per coppia query+doc")
parser.add_argument("--n_hard_neg",  type=int,   default=7,
                    help="Negativi per training step. "
                         "NON deve superare il n_hard_neg usato in prepare_reranker_data.py")
parser.add_argument("--seed",        type=int,   default=42)

# LoRA
parser.add_argument("--lora_r",      type=int,   default=8,
                    help="Rank LoRA. 0.6B: 8 | 4B: 16")
parser.add_argument("--lora_alpha",  type=int,   default=16,
                    help="Alpha LoRA (default: 2*lora_r)")
parser.add_argument("--lora_dropout", type=float, default=0.05)

# Eval e checkpointing
parser.add_argument("--eval_steps",  type=int,   default=200,
                    help="Valuta sul dev set ogni N optimizer steps")
parser.add_argument("--save_steps",  type=int,   default=200,
                    help="Salva checkpoint ogni N optimizer steps")
parser.add_argument("--patience",    type=int,   default=5,
                    help="Early stopping patience (in eval steps senza miglioramento)")

# Precision e ottimizzazioni
parser.add_argument("--fp16",        action="store_true", default=True)
parser.add_argument("--bf16",        action="store_true", default=False,
                    help="Usa bfloat16 invece di fp16 (solo Ampere+)")
parser.add_argument("--no_fp16",     action="store_true", default=False,
                    help="Disabilita fp16/bf16 (solo CPU o debug)")

args = parser.parse_args()

# Deriva run_name e output_dir se non specificati
if args.run_name is None:
    model_short = args.model.split("/")[-1]
    args.run_name = f"ft_{model_short}_r{args.lora_r}"

if args.output_dir is None:
    args.output_dir = f"models/{args.run_name}"

# fp16 vs bf16 vs full precision
if args.no_fp16:
    USE_FP16 = False
    USE_BF16 = False
    DTYPE = torch.float32
elif args.bf16:
    USE_FP16 = False
    USE_BF16 = True
    DTYPE = torch.bfloat16
else:
    USE_FP16 = True
    USE_BF16 = False
    DTYPE = torch.float16


# ─────────────────────────────────────────────────────────────
# TASK_INSTRUCTION — DEVE ESSERE IDENTICA A CausalReranker.py
# ─────────────────────────────────────────────────────────────
# ATTENZIONE: se cambi questa stringa, cambiala anche in CausalReranker.py.
# Qualsiasi discrepanza tra training e inference degrada le metriche.
TASK_INSTRUCTION = (
    "Given a short user query (like a tweet) and a formal academic paper, "
    "judge how relevant the paper is to the query."
    "Return a higher score for relevant papers, and a lower score for irrelevant papers."
)

# Prompt format Qwen3-Reranker (identico a CausalReranker.py)
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Return an higher score for relevant papers and a lower score for irrelevant papers."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(query: str, doc: str) -> str:
    return (
        f"<Instruct>: {TASK_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


# ─────────────────────────────────────────────────────────────
# Riproducibilità
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class RerankerDataset(Dataset):
    """
    Ogni esempio è una lista di coppie (query, doc) con label:
      - label=1: (query, positive)
      - label=0: (query, negative) × n_hard_neg

    Strategia di sampling durante training:
      - Usa 1 positivo + n_hard_neg negativi per ogni esempio
      - I negativi vengono campionati casualmente dagli hard negs disponibili
        (permette augmentation quando n_hard_neg_available > n_hard_neg_used)
    """

    def __init__(self, path: str, n_hard_neg: int, shuffle_negs: bool = True):
        self.examples  = []
        self.n_hard_neg = n_hard_neg
        self.shuffle_negs = shuffle_negs

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                if ex.get("negatives"):
                    self.examples.append(ex)

        print(f"  Dataset loaded: {len(self.examples)} esempi da {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query    = ex["query"]
        positive = ex["positive"]
        all_negs = ex["negatives"]

        # Campiona n_hard_neg negativi (con eventuale ripetizione se non bastano)
        if len(all_negs) >= self.n_hard_neg:
            negs = random.sample(all_negs, self.n_hard_neg) if self.shuffle_negs \
                else all_negs[:self.n_hard_neg]
        else:
            # Pochi negativi: usa tutti + ripeti
            negs = all_negs + random.choices(all_negs, k=self.n_hard_neg - len(all_negs))

        # Ritorna lista piatta: [(query, positive, 1), (query, neg1, 0), ...]
        pairs  = [(query, positive)] + [(query, neg) for neg in negs]
        labels = [1] + [0] * len(negs)
        return pairs, labels


def collate_fn(batch, tokenizer, prefix_ids, suffix_ids, max_length):
    """
    Trasforma una batch di (pairs, labels) in tensori pronti per il modello.

    Ogni coppia viene formattata con PREFIX + content + SUFFIX (come in inference).
    Il padding è LEFT-padding (richiesto dai modelli causali per la posizione finale).
    """
    all_formatted = []
    all_labels    = []

    for pairs, labels in batch:
        for (query, doc), label in zip(pairs, labels):
            all_formatted.append(format_pair(query, doc))
            all_labels.append(label)

    # content_max_length: esclude overhead di prefix/suffix
    content_max = max_length - len(prefix_ids) - len(suffix_ids)

    # Tokenizza il contenuto senza special tokens (li aggiungiamo manualmente)
    tokenized = tokenizer(
        all_formatted,
        padding=False,
        truncation=True,
        max_length=content_max,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    # Assembla: prefix + content + suffix per ogni sequenza
    for i, ids in enumerate(tokenized["input_ids"]):
        tokenized["input_ids"][i] = prefix_ids + ids + suffix_ids

    # Left-padding del batch
    padded = tokenizer.pad(
        tokenized,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
    )

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    return padded, labels_tensor


# ─────────────────────────────────────────────────────────────
# Loss — meccanismo yes/no ufficiale Qwen3-Reranker
# ─────────────────────────────────────────────────────────────
def compute_loss(logits, labels, token_true_id, token_false_id):
    """
    Loss che imita esattamente il meccanismo di scoring in inference:

      last_logit = logits[:, -1, :]        # logit dell'ultimo token
      pair       = [logit_no, logit_yes]
      log_probs  = log_softmax(pair)
      score      = exp(log_probs[yes])     # = P(yes)

    Per il training usiamo cross-entropy sulle 2 classi [no, yes]:
      - label=1 (rilevante)   → classe 1 (yes)
      - label=0 (irrilevante) → classe 0 (no)

    Questo è IDENTICO alla loss usata nel paper Qwen3 e nel training ufficiale.
    """
    # Estrai logit dell'ultimo token generato
    last_logits = logits[:, -1, :]                         # [B, vocab_size]

    true_logits  = last_logits[:, token_true_id]           # [B]
    false_logits = last_logits[:, token_false_id]          # [B]

    # Stack: [B, 2] → [no_logit, yes_logit]
    pair_logits = torch.stack([false_logits, true_logits], dim=1)

    # Cross entropy: labels=1 → ottimizza yes, labels=0 → ottimizza no
    loss = F.cross_entropy(pair_logits, labels)
    return loss


# ─────────────────────────────────────────────────────────────
# Eval su dev set
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dev_loader, tokenizer, token_true_id, token_false_id, device):
    """
    Calcola loss media e accuracy binaria sul dev set.

    Accuracy = percentuale di coppie correttamente classificate:
      - (query, positive) → P(yes) > 0.5
      - (query, negative) → P(yes) < 0.5

    Nota: questa non è nDCG@10 ma è un proxy veloce e correlato.
    Per la valutazione finale usa Evaluator.java sul test set.
    """
    model.eval()
    total_loss  = 0.0
    total_acc   = 0
    total_count = 0

    for padded, labels in dev_loader:
        for k in padded:
            padded[k] = padded[k].to(device)
        labels = labels.to(device)

        logits = model(**padded).logits

        loss = compute_loss(logits, labels, token_true_id, token_false_id)
        total_loss += loss.item() * labels.size(0)

        # Calcola accuratezza
        last_logits  = logits[:, -1, :]
        true_logits  = last_logits[:, token_true_id]
        false_logits = last_logits[:, token_false_id]
        pair_logits  = torch.stack([false_logits, true_logits], dim=1)
        preds        = pair_logits.argmax(dim=1)   # 0=no, 1=yes
        total_acc   += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_acc  / max(total_count, 1)

    model.train()
    return {"loss": avg_loss, "accuracy": accuracy, "n_examples": total_count}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {args.model}")
    print(f"  Run name:    {args.run_name}")
    print(f"  Device:      {device}")
    print(f"  LoRA rank:   {args.lora_r}")
    print(f"  Dtype:       {DTYPE}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size} × grad_accum {args.grad_accum} "
          f"= effective {args.batch_size * args.grad_accum}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",   # CRITICO per modelli causali
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Token IDs per yes/no — fissi per vocabolario Qwen3
    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    print(f"  Token IDs → yes: {token_true_id}, no: {token_false_id}")

    if token_true_id == tokenizer.unk_token_id or token_false_id == tokenizer.unk_token_id:
        raise ValueError(
            "Token 'yes' o 'no' non nel vocabolario. "
            "Controlla che il tokenizer sia corretto per Qwen3-Reranker."
        )

    # Pre-tokenizza prefix e suffix (costanti)
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    print(f"  Prefix: {len(prefix_ids)} tokens | Suffix: {len(suffix_ids)} tokens")

    # ── Modello + LoRA ──────────────────────────────────────────
    print(f"\nLoading model: {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=DTYPE,
        device_map="auto" if device == "cuda" else None,
    )

    # Gradient checkpointing — fondamentale per 4B, utile anche per 0.6B
    model.gradient_checkpointing_enable()

    lora_alpha = args.lora_alpha if args.lora_alpha > 0 else 2 * args.lora_r

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=lora_alpha,
        # Moduli target: attention + proiezione output (standard per Qwen)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset e DataLoader ────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = RerankerDataset(args.train_data, n_hard_neg=args.n_hard_neg, shuffle_negs=True)
    dev_dataset   = RerankerDataset(args.dev_data,   n_hard_neg=args.n_hard_neg, shuffle_negs=False)

    _collate = lambda batch: collate_fn(batch, tokenizer, prefix_ids, suffix_ids, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * 2,  # dev: niente grad, possiamo raddoppiare
        shuffle=False,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # ── Optimizer e Scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps   = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    print(f"\nTraining steps: {total_steps} | Warmup: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Scaler fp16 ─────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    # ── Training loop ───────────────────────────────────────────
    best_dev_loss     = float("inf")
    best_dev_accuracy = 0.0
    patience_counter  = 0
    global_step       = 0
    eval_history      = []

    # Salva la configurazione del run
    run_config = {
        "model":        args.model,
        "run_name":     args.run_name,
        "lora_r":       args.lora_r,
        "lora_alpha":   lora_alpha,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "grad_accum":   args.grad_accum,
        "lr":           args.lr,
        "n_hard_neg":   args.n_hard_neg,
        "max_length":   args.max_length,
        "seed":         args.seed,
        "task_instruction": TASK_INSTRUCTION,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nInizio training: {args.epochs} epoch × {len(train_loader)} steps/epoch")
    print(f"Eval ogni {args.eval_steps} optimizer steps | Patience: {args.patience}\n")

    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for step, (padded, labels) in enumerate(pbar, 1):
            # Sposta su GPU
            for k in padded:
                padded[k] = padded[k].to(device)
            labels = labels.to(device)

            # Forward pass con autocast
            with torch.cuda.amp.autocast(enabled=USE_FP16 or USE_BF16,
                                         dtype=DTYPE if (USE_FP16 or USE_BF16) else torch.float32):
                logits = model(**padded).logits
                loss   = compute_loss(logits, labels, token_true_id, token_false_id)
                loss   = loss / args.grad_accum  # normalizza per accumulazione

            # Backward
            scaler.scale(loss).backward()

            epoch_loss  += loss.item() * args.grad_accum
            epoch_steps += 1

            # Optimizer step ogni grad_accum mini-batch
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / epoch_steps
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                    "step": global_step,
                })

                # ── Eval sul dev set ────────────────────────────
                if global_step % args.eval_steps == 0:
                    print(f"\n[Step {global_step}] Evaluating on dev set...")
                    dev_metrics = evaluate(
                        model, dev_loader, tokenizer,
                        token_true_id, token_false_id, device
                    )
                    dev_metrics["step"]  = global_step
                    dev_metrics["epoch"] = epoch
                    eval_history.append(dev_metrics)

                    print(f"  Dev loss:     {dev_metrics['loss']:.4f}")
                    print(f"  Dev accuracy: {dev_metrics['accuracy']:.4f} "
                          f"({dev_metrics['n_examples']} pairs)")

                    # Salva eval history
                    eval_path = os.path.join(
                        args.results_dir, f"finetune_eval_{args.run_name}.json"
                    )
                    with open(eval_path, "w") as f:
                        json.dump(eval_history, f, indent=2)

                    # Best model per accuracy sul dev
                    is_best = dev_metrics["accuracy"] > best_dev_accuracy
                    if is_best:
                        best_dev_accuracy = dev_metrics["accuracy"]
                        best_dev_loss     = dev_metrics["loss"]
                        patience_counter  = 0

                        best_path = os.path.join(args.output_dir, "best")
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        print(f"  ✓ Nuovo best model salvato → {best_path} "
                              f"(accuracy={best_dev_accuracy:.4f})")
                    else:
                        patience_counter += 1
                        print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

                    # Early stopping
                    if patience_counter >= args.patience:
                        print(f"\nEarly stopping a step {global_step} "
                              f"(patience={args.patience} esaurita).")
                        print(f"Best dev accuracy: {best_dev_accuracy:.4f}")
                        break

                # ── Checkpoint periodico ────────────────────────
                if global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    print(f"  Checkpoint salvato → {ckpt_path}")

        else:
            # Fine epoch normale (no early stopping)
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"\nEpoch {epoch} completata. Loss media: {avg_epoch_loss:.4f}")
            continue

        # Se siamo usciti dal loop step per early stopping, esci anche dall'epoch loop
        break

    # ── Salva modello finale ─────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModello finale salvato → {final_path}")
    print(f"Best checkpoint    → {args.output_dir}/best")
    print(f"Best dev accuracy  → {best_dev_accuracy:.4f}")
    print(f"Eval history       → {args.results_dir}/finetune_eval_{args.run_name}.json")

    print(f"\n{'='*60}")
    print("PROSSIMI PASSI:")
    print(f"  1. Valuta il best checkpoint sul test set con evaluate_reranker.py")
    print(f"  2. Per usarlo in inference, aggiorna CausalReranker.py:")
    print(f"     MODEL_NAME = '{args.output_dir}/best'")
    print(f"     (o usa --model {args.output_dir}/best)")
    print(f"  3. Per passare a 4B: --model Qwen/Qwen3-Reranker-4B --lora_r 16 \\")
    print(f"                        --batch_size 2 --grad_accum 16")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()