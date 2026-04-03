"""
evaluate_reranker.py — Valuta un checkpoint fine-tuned in modalità inference reale
====================================================================================
Carica il checkpoint LoRA fine-tuned, esegue il re-ranking sul test set
(o su qualsiasi BM25 run) con lo STESSO meccanismo yes/no di CausalReranker.py,
e scrive l'output in formato compatibile con Evaluator.java.

Questo script è pensato per:
  1. Confrontare il checkpoint fine-tuned vs zero-shot sullo stesso test set
  2. Produrre il file JSON da passare al Java evaluator per le metriche finali
  3. Debug rapido: con --max_queries 200 puoi testare in pochi minuti

Uso:
  # Valuta il best checkpoint sul test set
  python evaluate_reranker.py \\
    --model models/ft_qwen3_reranker_0.6B_r8/best \\
    --queries data/expanded_queries_bge_large.json \\
    --papers  data/collection_data.json \\
    --bm25    results/bm25_results.json \\
    --test_split data/finetune_reranker/test.jsonl \\
    --output results/reranked_finetuned_test.json

  # Confronto zero-shot (stesso script, modello base)
  python evaluate_reranker.py \\
    --model Qwen/Qwen3-Reranker-0.6B \\
    --test_split data/finetune_reranker/test.jsonl \\
    --output results/reranked_zeroshot_test.json

  # Rerank su tutte le query (per submission finale con 4B fine-tuned)
  python evaluate_reranker.py \\
    --model models/ft_qwen3_reranker_4B_r16/best \\
    --output results/reranked_finetuned_4B_final.json
"""

import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",       required=True,
                    help="Path al checkpoint fine-tuned (es. models/ft_.../best) "
                         "oppure modello base HuggingFace (es. Qwen/Qwen3-Reranker-0.6B)")
parser.add_argument("--base_model",  default=None,
                    help="Modello base per LoRA adapter. "
                         "Se --model è un path locale con adapter_config.json, "
                         "viene rilevato automaticamente. "
                         "Altrimenti specifica es. Qwen/Qwen3-Reranker-0.6B")
parser.add_argument("--queries",     default="../../../../../../data/expanded_queries_bge_large.json")
parser.add_argument("--papers",      default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",        default="../../../../../../../results/bm25_results.json")
parser.add_argument("--test_split",  default=None,
                    help="Se specificato, valuta SOLO sulle query del test set. "
                         "Passa data/finetune_reranker/test.jsonl per un confronto fair.")
parser.add_argument("--output",      required=True,
                    help="Path output JSON (formato: {qid -> [pubkey, ...]})")
parser.add_argument("--top_k",       type=int, default=100)
parser.add_argument("--batch",       type=int, default=16)
parser.add_argument("--max_length",  type=int, default=512)
parser.add_argument("--max_queries", type=int, default=None,
                    help="Limita il numero di query (debug rapido)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────
# TASK_INSTRUCTION — IDENTICA a CausalReranker.py e finetune
# ─────────────────────────────────────────────────────────────
TASK_INSTRUCTION = (
    "Given a short user query (like a tweet) and a formal academic paper, "
    "judge how relevant the paper is to the query."
    "Return a higher score for relevant papers, and a lower score for irrelevant papers."
)
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
# Carica modello (auto-detection LoRA vs base)
# ─────────────────────────────────────────────────────────────
def load_model(model_path: str, base_model: str, device: str):
    """
    Carica automaticamente:
    - Se model_path è una directory con adapter_config.json → LoRA checkpoint
    - Altrimenti → modello base HuggingFace (zero-shot)
    """
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.isfile(adapter_config_path)

    if is_lora:
        # Leggi il base model dall'adapter_config se non specificato
        if base_model is None:
            with open(adapter_config_path) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path", None)
            if base_model is None:
                raise ValueError(
                    "base_model non trovato in adapter_config.json. "
                    "Specifica --base_model manualmente."
                )
        print(f"  Modello LoRA rilevato.")
        print(f"  Base model: {base_model}")
        print(f"  Adapter:    {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()  # fonde LoRA per inference più veloce
    else:
        print(f"  Modello base (zero-shot): {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Scoring (identico a CausalReranker.py)
# ─────────────────────────────────────────────────────────────
def make_score_fn(model, tokenizer, prefix_ids, suffix_ids, max_length,
                  token_true_id, token_false_id):
    content_max = max_length - len(prefix_ids) - len(suffix_ids)

    def score_pairs(pairs):
        formatted = [format_pair(q, d) for q, d in pairs]
        inputs = tokenizer(
            formatted,
            padding=False,
            truncation=True,
            max_length=content_max,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_ids + ids + suffix_ids
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt",
                               max_length=max_length)
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits

        last = logits[:, -1, :]
        stacked = torch.stack([last[:, token_false_id], last[:, token_true_id]], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()

    return score_pairs


def predict_scores_safe(score_fn, pairs, batch_size):
    """Identico a CausalReranker.py: gestione OOM con dimezzamento batch."""
    all_scores    = []
    current_batch = batch_size
    i = 0
    while i < len(pairs):
        batch = pairs[i: i + current_batch]
        try:
            all_scores.extend(score_fn(batch))
            i += current_batch
            current_batch = batch_size
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if current_batch <= 1:
                all_scores.extend([0.0] * len(batch))
                i += current_batch
                current_batch = batch_size
            else:
                current_batch = max(1, current_batch // 2)
    return all_scores


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ── Carica dati ──────────────────────────────────────────
    print("Loading data...")
    with open(args.queries, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)
    with open(args.papers, "r", encoding="utf-8") as f:
        papers_raw = json.load(f)
    with open(args.bm25, "r", encoding="utf-8") as f:
        bm25_results = json.load(f)

    paper_texts = {
        str(p["pubkey"]): (p.get("title","") + ". " + p.get("abstract","")).strip()
        for p in papers_raw
    }
    query_texts = {
        str(q["index"]): q.get("original", q.get("text", ""))
        for q in queries_raw
    }

    # Filtra per test split se specificato
    if args.test_split:
        print(f"Filtering to test split: {args.test_split}")
        test_qids = set()
        with open(args.test_split) as f:
            for line in f:
                ex = json.loads(line)
                test_qids.add(str(ex["qid"]))
        bm25_results = {qid: v for qid, v in bm25_results.items() if qid in test_qids}
        print(f"  Query nel test split: {len(bm25_results)}")

    # Limita per debug
    if args.max_queries:
        items = list(bm25_results.items())[:args.max_queries]
        bm25_results = dict(items)
        print(f"  Limitato a {args.max_queries} query (debug mode)")

    # ── Carica modello ────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(args.model, args.base_model, device)

    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)

    score_fn = make_score_fn(
        model, tokenizer, prefix_ids, suffix_ids, args.max_length,
        token_true_id, token_false_id
    )

    # ── Sanity check ──────────────────────────────────────────
    test_pairs = [
        ("neural network classification",
         "A deep learning model for image classification using CNNs"),
        ("neural network classification",
         "Ancient Roman history and the emperors of the first century BC"),
    ]
    scores = score_fn(test_pairs)
    print(f"\nSanity check → relevant={scores[0]:.4f} | irrelevant={scores[1]:.4f} | "
          f"Δ={abs(scores[0]-scores[1]):.4f}")
    if abs(scores[0] - scores[1]) < 0.05:
        print("WARNING: Δ molto piccolo. Controlla il modello.")

    # ── Re-ranking ────────────────────────────────────────────
    print(f"\nRe-ranking {len(bm25_results)} query (top_k={args.top_k})...")
    reranked = {}

    for qid, candidates in tqdm(bm25_results.items()):
        qt = query_texts.get(str(qid), "")
        if not qt:
            reranked[qid] = candidates
            continue

        valid = [(pk, paper_texts[str(pk)])
                 for pk in candidates[:args.top_k] if str(pk) in paper_texts]
        if not valid:
            reranked[qid] = candidates
            continue

        pairs = [(qt, doc) for _, doc in valid]
        pubkeys = [pk for pk, _ in valid]

        scores = predict_scores_safe(score_fn, pairs, args.batch)
        ranked = sorted(zip(pubkeys, scores), key=lambda x: -x[1])
        reranked[qid] = [pk for pk, _ in ranked]

    # ── Salva output ──────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reranked, f, indent=2, ensure_ascii=False)

    print(f"\nOutput salvato → {args.output}")
    print(f"Coverage: {len(reranked)}/{len(bm25_results)} query rerankate")
    print("\nOra esegui il Java evaluator puntando a questo file.")


if __name__ == "__main__":
    main()