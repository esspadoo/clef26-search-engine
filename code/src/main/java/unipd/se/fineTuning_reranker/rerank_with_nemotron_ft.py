"""
rerank_with_nemotron_ft.py

Inference con llama-nemotron-rerank-1b-v2 fine-tuned.

Carica il modello con AutoModelForSequenceClassification + AutoTokenizer
direttamente — coerente con come finetune_nemotron_reranker_v2.py salva
il modello (model.save_pretrained + tokenizer.save_pretrained).

NON usa CrossEncoder di sentence-transformers perché il modello salvato
è un HuggingFace nativo, non un CrossEncoder wrappato.

IMPORTANTE:
  - trust_remote_code=True obbligatorio (architettura bidirectional custom)
  - Nessun prefisso sulle query — solo clean_text(), identico al training
  - use_sigmoid=True (default): porta i logit raw in [0,1] senza
    cambiare l'ordine relativo (sigmoid è monotona crescente)

Uso:
  python rerank_with_nemotron_ft.py \
    --model_dir    models/nemotron-rerank-1b-retrix/final \
    --nemotron_run reranked_results_nemotron_topk400.json \
    --collection   collection_data.json \
    --queries      en_train.json \
    --output       reranked_results_nemotron_ft_top100.json \
    --rerank_depth 100 \
    --batch_size   64
"""

import json
import logging
import emoji
import re
import gc
import argparse
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Preprocessing ──────────────────────────────────────────────────────────────
# IDENTICO a prepare_data_nemotron_finetune.py — nessun prefisso.

def clean_text(text: str) -> str:
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


def doc_text(item: dict, max_chars: int) -> str:
    t = (item.get("title")    or "").strip()
    a = (item.get("abstract") or "").strip()
    return f"{t}. {a}".strip(". ")[:max_chars]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_collection(path: str, max_chars: int) -> dict[str, str]:
    """Entrambi i corpus usano 'pubkey' come chiave."""
    data   = json.load(open(path, encoding="utf-8"))
    corpus = {}
    for item in data:
        if "pubkey" in item and item["pubkey"] is not None:
            key = str(item["pubkey"])
        else:
            continue
        text = doc_text(item, max_chars)
        if text:
            corpus[key] = text
    log.info(f"Corpus: {len(corpus):,} documenti  ← {path}")
    return corpus


def load_queries(path: str) -> dict[str, str]:
    data    = json.load(open(path, encoding="utf-8"))
    queries = {}
    for item in data:
        qid          = str(item["index"])
        queries[qid] = clean_text(item["text"])
    log.info(f"Queries: {len(queries):,}  ← {path}")
    return queries


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    raw = json.load(open(path, encoding="utf-8"))
    run = {str(k): [str(d) for d in v] for k, v in raw.items()}
    log.info(f"Nemotron run: {len(run):,} queries  ← {path}")
    return run


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_pairs(
        model,
        tokenizer,
        query:       str,
        docs:        list[str],
        batch_size:  int,
        max_length:  int,
        device:      torch.device,
        use_sigmoid: bool,
) -> list[float]:
    """
    Calcola gli score per una query contro una lista di documenti.
    Gestisce OOM dimezzando il batch_size automaticamente.
    """
    all_scores = []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        current_bs = len(batch_docs)

        while current_bs >= 1:
            try:
                enc = tokenizer(
                    [query] * current_bs,
                    batch_docs[:current_bs],
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    )
                enc = {k: v.to(device) for k, v in enc.items()}

                with torch.no_grad():
                    out    = model(**enc)
                    logits = out.logits  # shape: (batch, 1)
                    if logits.dim() > 1:
                        logits = logits.squeeze(-1)
                    if use_sigmoid:
                        logits = torch.sigmoid(logits)
                    scores = logits.float().cpu().tolist()

                # Se il batch era stato ridotto per OOM, processa il resto
                all_scores.extend(scores)
                if current_bs < len(batch_docs):
                    # Ricorsione sul resto del batch originale
                    remaining = batch_docs[current_bs:]
                    rest = score_pairs(
                        model, tokenizer, query, remaining,
                        current_bs, max_length, device, use_sigmoid
                    )
                    all_scores.extend(rest)
                break

            except torch.cuda.OutOfMemoryError:
                log.warning(f"OOM con batch_size={current_bs}, provo {current_bs // 2}...")
                torch.cuda.empty_cache()
                gc.collect()
                current_bs //= 2

        else:
            log.error("OOM anche con batch_size=1. Restituisco 0.5.")
            all_scores.extend([0.5] * len(batch_docs))

    return all_scores


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(results: dict, nemotron_run: dict, rerank_depth: int):
    if not results:
        log.error("SANITY CHECK FAILED: output vuoto!")
        return
    coverage = len(results) / max(1, len(nemotron_run))
    avg_len  = sum(len(v) for v in results.values()) / len(results)
    log.info(f"Sanity check: {len(results):,} queries ({coverage:.1%} coverage), "
             f"avg {avg_len:.1f} doc/query (atteso ≤{rerank_depth})")
    sample_qid = next(iter(results))
    log.info(f"  Sample qid={sample_qid} top-3: {results[sample_qid][:3]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reranking con llama-nemotron-rerank-1b-v2 fine-tuned (HF nativo)"
    )
    parser.add_argument("--model_dir",     required=True,
                        help="Path a models/nemotron-rerank-1b-retrix/final")
    parser.add_argument("--nemotron_run",  required=True)
    parser.add_argument("--collection",    required=True)
    parser.add_argument("--queries",       required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--rerank_depth",  type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--max_length",    type=int, default=512)
    parser.add_argument("--max_doc_chars", type=int, default=4000)
    parser.add_argument("--no_sigmoid",    action="store_true",
                        help="Usa logit raw invece di sigmoid (default: sigmoid attiva)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Carica tokenizer e modello ─────────────────────────────────────────────
    log.info(f"Caricamento da {args.model_dir} (trust_remote_code=True)...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    log.info(f"Modello caricato. Parametri: {sum(p.numel() for p in model.parameters()):,}")

    # ── Carica dati ────────────────────────────────────────────────────────────
    corpus       = load_collection(args.collection, args.max_doc_chars)
    queries      = load_queries(args.queries)
    nemotron_run = load_nemotron_run(args.nemotron_run)

    use_sigmoid = not args.no_sigmoid
    log.info(f"use_sigmoid={use_sigmoid}")

    # ── Reranking ──────────────────────────────────────────────────────────────
    results      = {}
    n_missing_q  = 0
    n_empty_docs = 0

    for qid, nemotron_docs in tqdm(nemotron_run.items(), desc="Reranking"):
        if qid not in queries:
            n_missing_q += 1
            continue

        query         = queries[qid]
        candidate_ids = nemotron_docs[:args.rerank_depth]
        valid_ids     = [d for d in candidate_ids if d in corpus]
        valid_docs    = [corpus[d] for d in valid_ids]

        if not valid_ids:
            n_empty_docs += 1
            continue

        scores = score_pairs(
            model, tokenizer, query, valid_docs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            use_sigmoid=use_sigmoid,
        )

        ranked       = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [doc_id for doc_id, _ in ranked]

    if n_missing_q:
        log.warning(f"Query mancanti nel file queries: {n_missing_q}")
    if n_empty_docs:
        log.warning(f"Query senza documenti nel corpus: {n_empty_docs}")

    sanity_check(results, nemotron_run, args.rerank_depth)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"✓ Output: {out_path}  ({len(results):,} queries)")


if __name__ == "__main__":
    main()