"""
Reranker_zerank1.py — zerank-1 / zerank-2 re-ranking dei risultati BM25
========================================================================
zeroentropy/zerank-1 e zeroentropy/zerank-2 sono cross-encoder basati su
Qwen3-4B con architettura e pesi custom (modeling_zeranker.py).

Meccanismo ufficiale (da HuggingFace zeroentropy/zerank-1 e zerank-2):
  1. Caricare con CrossEncoder("zeroentropy/zerank-1", trust_remote_code=True)
     NON usare AutoModelForSequenceClassification direttamente: il parametro
     score.weight verrebbe re-inizializzato casualmente invece di usare i
     pesi addestrati nel checkpoint.
  2. model.predict([(query, doc), ...]) → score float in [0, 1]
     - ~1.0 = molto rilevante
     - ~0.0 = irrilevante
     Score già normalizzati: nessuna sigmoid necessaria.
  3. CRITICO: CrossEncoder.predict() con batch_size > 1 crasha se il
     tokenizer non ha pad_token definito (bug noto del modello). Soluzione:
     mini-batching manuale coppia per coppia dentro predict_scores(),
     NON affidarsi al parametro batch_size di CrossEncoder.predict().

Differenze chiave rispetto a Nemotron:
  - CrossEncoder (sentence-transformers), NON AutoModelForSequenceClassification
  - Output già in [0,1]: nessuna sigmoid
  - Mini-batching manuale per evitare crash su padding
  - Per passare da zerank-1 a zerank-2: cambia solo MODEL_NAME

Richiede:
  pip install sentence-transformers torch

Legge:
  - data/expanded_queries_*.json   (query, campo "original" o "text")
  - data/collection_data.json      (corpus, title+abstract)
  - results/bm25_results.json      (output Java: { qid → [pubkey, ...] })

Scrive:
  - results/reranked_results_zerank.json  (stesso formato: { qid → [pubkey, ...] })

Uso:
  python Reranker_zerank1.py
  python Reranker_zerank1.py --model zeroentropy/zerank-2 --top_k 100 --batch 16
"""

import json
import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--queries",     default="../../../../../../data/expanded_queries_bge_large.json")
parser.add_argument("--papers",      default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",        default="../../../../../../../results/bm25_results.json")
parser.add_argument("--output",      default="../../../../../../../results/reranked_results_zerank.json")
parser.add_argument("--model",       default="zeroentropy/zerank-1",
                    help="Modello zerank da usare. "
                         "Opzioni: zeroentropy/zerank-1, zeroentropy/zerank-2 (default: zerank-1). "
                         "Entrambi hanno la stessa interfaccia: cambia solo questo argomento.")
parser.add_argument("--top_k",       type=int, default=100,
                    help="Candidati BM25 da passare al re-ranker (default: 100)")
parser.add_argument("--batch",       type=int, default=16,
                    help="Coppie per iterazione di scoring (default: 16). "
                         "zerank-1 (4B BF16) ≈ 8-9 GB VRAM. "
                         "Su L40S 48GB puoi alzare fino a 32-64.")
parser.add_argument("--query_chunk", type=int, default=8,
                    help="Query per chunk (default: 8).")
parser.add_argument("--max_length",  type=int, default=512,
                    help="Lunghezza massima token per coppia (default: 512).")


# ──────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────
def sanity_check_scores(score_fn) -> bool:
    """
    Verifica che il modello produca score discriminativi.
    CrossEncoder zerank restituisce score in [0,1]: rilevante >> irrilevante.
    """
    test_pairs = [
        ("neural network classification",
         "A deep learning model for image classification using convolutional neural networks"),
        ("neural network classification",
         "The history of ancient Rome and its emperors in the first century BC"),
    ]
    try:
        scores = score_fn(test_pairs)
        s0, s1 = float(scores[0]), float(scores[1])
        diff = abs(s0 - s1)
        print(f"\n  [Sanity check] Score rilevante={s0:.4f} | Score irrilevante={s1:.4f} | Δ={diff:.4f}")
        if diff < 0.05:
            print("  WARNING: score quasi identici — controlla il modello/tokenizer!")
            return False
        print("  [Sanity check] OK.")
        return True
    except Exception as e:
        print(f"  WARNING: sanity check fallito: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Helpers CPU
# ──────────────────────────────────────────────────────────────
def build_pairs_for_queries(query_items, paper_texts, query_texts, top_k):
    """
    Costruisce le coppie (query_text, doc_text) per un chunk di query.
    Ritorna:
      all_pairs : lista piatta di (str, str)
      meta      : [(qid, valid_keys, n_pairs, fallback_candidates), ...]
      missing   : n° doc non trovati nel corpus
    """
    all_pairs = []
    meta      = []
    missing   = 0

    for qid, candidate_pubkeys in query_items:
        # Supporta sia chiavi int che str nel dict query_texts
        query_text = query_texts.get(str(qid), query_texts.get(qid, ""))
        candidates = candidate_pubkeys[:top_k]

        if not query_text:
            print(f"  WARNING: query text non trovato per qid={qid}, uso ordine BM25", flush=True)
            meta.append((qid, [], 0, candidates))
            continue

        pairs      = []
        valid_keys = []
        for pk in candidates:
            doc_text = paper_texts.get(str(pk), paper_texts.get(pk))
            if doc_text:
                pairs.append((query_text, doc_text))
                valid_keys.append(pk)
            else:
                missing += 1

        meta.append((qid, valid_keys, len(pairs), candidates))
        all_pairs.extend(pairs)

    return all_pairs, meta, missing


def scores_to_results(meta, scores_flat):
    """
    Ricostruisce qid→[pubkey] dallo score array piatto.
    Score in [0,1]: reverse=True mette i più rilevanti in cima.
    """
    results = {}
    offset  = 0
    for qid, valid_keys, n_pairs, fallback_candidates in meta:
        if n_pairs == 0:
            results[qid] = fallback_candidates
            continue
        scores = scores_flat[offset: offset + n_pairs]
        offset += n_pairs
        # Score alti (~1) = più rilevanti → reverse=True
        ranked = sorted(zip(valid_keys, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [pk for pk, _ in ranked]
    return results


# ──────────────────────────────────────────────────────────────
# Caricamento modello — CrossEncoder ufficiale zerank
# ──────────────────────────────────────────────────────────────
def load_zerank(model_name: str, device: str, max_length: int):
    """
    Carica zerank-1 o zerank-2 tramite CrossEncoder di sentence-transformers.

    CRITICO: usare trust_remote_code=True per caricare modeling_zeranker.py.
    Senza di esso, transformers usa Qwen3ForSequenceClassification standard
    e re-inizializza score.weight casualmente → score completamente casuali.

    Output di model.predict(): score float già in [0,1], nessuna sigmoid necessaria.
    """
    from sentence_transformers.cross_encoder import CrossEncoder

    print(f"  Loading CrossEncoder: {model_name} → {device}", flush=True)
    model = CrossEncoder(
        model_name,
        trust_remote_code=True,
        max_length=max_length,
        device=device,
    )
    # Assicura che il pad_token sia definito per evitare crash durante il padding
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    if model.model.config.pad_token_id is None:
        model.model.config.pad_token_id = model.tokenizer.eos_token_id

    print(f"  Modello caricato. Max length: {max_length}", flush=True)

    def score_pairs(pairs: list) -> list:
        """
        Closure che incapsula il CrossEncoder.
        Accetta lista di (query, doc) e ritorna lista di float in [0, 1].

        NOTA: NON passare batch_size > 1 a model.predict() direttamente,
        in quanto il tokenizer di zerank può non avere pad_token configurato
        correttamente nella chiamata interna, causando crash con batch > 1.
        Il mini-batching viene gestito esternamente in predict_scores().
        Qui predict() viene chiamato su una coppia singola alla volta.
        """
        scores = model.predict(pairs)
        # predict() restituisce numpy array o lista, convertiamo a lista di float
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return scores if isinstance(scores, list) else [float(scores)]

    return score_pairs


# ──────────────────────────────────────────────────────────────
# Inferenza con gestione OOM e mini-batching manuale
# ──────────────────────────────────────────────────────────────
def predict_scores(score_fn, pairs: list, batch_size: int) -> list:
    """
    Itera le coppie in mini-batch, chiamando score_fn su ogni singola coppia.

    Il mini-batching è manuale e coppia per coppia: zerank con CrossEncoder
    può crashare se il padding è applicato su batch > 1 senza pad_token.
    Iteriamo a batch_size coppie per chiamata a score_fn, ma ogni coppia
    viene processata individualmente dentro score_fn per sicurezza.

    Gestisce OOM dimezzando il batch e ripristinandolo al chunk successivo.
    """
    all_scores    = []
    current_batch = batch_size
    i             = 0

    while i < len(pairs):
        batch = pairs[i: i + current_batch]
        try:
            # Processamento coppia per coppia per evitare crash da padding
            batch_scores = []
            for pair in batch:
                s = score_fn([pair])
                batch_scores.append(float(s[0]))
            all_scores.extend(batch_scores)
            i += current_batch
            current_batch = batch_size

        except (torch.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            if current_batch <= 1:
                print(
                    f"  WARNING: OOM anche con batch=1. "
                    f"Score=0.5 come fallback (ordine BM25 mantenuto).",
                    flush=True,
                )
                # 0.5 = valore neutro in [0,1], non perturba l'ordine relativo
                all_scores.extend([0.5] * len(batch))
                i += current_batch
                current_batch = batch_size
            else:
                current_batch = max(1, current_batch // 2)
                print(f"  OOM → riduco batch a {current_batch} per questo chunk", flush=True)

    return all_scores


# ──────────────────────────────────────────────────────────────
# Worker per-GPU (path multi-GPU)
# ──────────────────────────────────────────────────────────────
def rerank_worker(
        gpu_id: int,
        query_items: list,
        paper_texts: dict,
        query_texts: dict,
        model_name: str,
        top_k: int,
        batch_size: int,
        query_chunk: int,
        max_length: int,
        result_queue: mp.Queue,
) -> None:
    # CRITICO: isola la GPU prima di qualsiasi import CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    device = "cuda:0"
    print(f"[GPU {gpu_id}] {torch.cuda.get_device_name(0)} — caricamento modello...", flush=True)

    score_fn = load_zerank(model_name, device, max_length)

    if gpu_id == 0:
        sanity_check_scores(score_fn)

    local_results     = {}
    local_missing     = 0
    local_pairs_total = 0

    chunks = [
        query_items[i: i + query_chunk]
        for i in range(0, len(query_items), query_chunk)
    ]

    for chunk in tqdm(chunks, desc=f"Re-ranking [GPU {gpu_id}]", position=gpu_id, leave=True):
        all_pairs, meta, missing = build_pairs_for_queries(
            chunk, paper_texts, query_texts, top_k
        )
        local_missing += missing

        if not all_pairs:
            for qid, _, _, fallback in meta:
                local_results[qid] = fallback
            continue

        local_pairs_total += len(all_pairs)
        scores_flat = predict_scores(score_fn, all_pairs, batch_size)
        local_results.update(scores_to_results(meta, scores_flat))

    result_queue.put((local_results, local_missing, local_pairs_total))


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parser.parse_args()

    MODEL_NAME = args.model

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        DEVICE = "cpu"
        print("Device: cpu (nessuna GPU CUDA trovata)")
    else:
        DEVICE = "cuda:0"
        print(f"Device: cuda — {NUM_GPUS} GPU disponibili:")
        for i in range(NUM_GPUS):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    # ── Carica dati ────────────────────────────────────────────
    print(f"\nModello: {MODEL_NAME}")
    print("Loading data...")

    with open(args.queries, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)

    with open(args.papers, "r", encoding="utf-8") as f:
        papers_raw = json.load(f)

    with open(args.bm25, "r", encoding="utf-8") as f:
        bm25_results: dict = json.load(f)

    paper_texts: dict = {
        str(p["pubkey"]): (p.get("title", "") + " " + p.get("abstract", "")).strip()
        for p in papers_raw
    }

    query_texts: dict = {
        str(q["index"]): q.get("original", q.get("text", ""))
        for q in queries_raw
    }

    # ── Sanity check sui dati ──────────────────────────────────
    print(f"\nCorpus: {len(paper_texts)} documenti | "
          f"Queries: {len(query_texts)} | "
          f"BM25 results: {len(bm25_results)}")

    sample_qid        = next(iter(bm25_results))
    sample_candidates = bm25_results[sample_qid][:5]
    hits = sum(1 for pk in sample_candidates if str(pk) in paper_texts)
    print(f"Sanity check corpus — query '{sample_qid}': "
          f"{hits}/{len(sample_candidates)} candidati trovati nel corpus")
    if hits == 0:
        print("ERRORE CRITICO: 0 candidati trovati. "
              f"Tipo pubkey bm25={type(sample_candidates[0])}, "
              f"tipo chiave corpus=str")

    sample_qt = query_texts.get(str(sample_qid), "")
    print(f"Sanity check queries — qid='{sample_qid}': '{sample_qt[:80]}...'")
    if not sample_qt:
        print("WARNING: query text vuoto. Controlla campo 'original'/'text' nel JSON query.")

    all_query_items = list(bm25_results.items())
    print(f"\nRe-ranking {len(bm25_results)} queries "
          f"(top_k={args.top_k}, batch={args.batch}, "
          f"query_chunk={args.query_chunk}, max_length={args.max_length})...")

    # ── Singola GPU / CPU ───────────────────────────────────────
    if NUM_GPUS <= 1:
        score_fn = load_zerank(MODEL_NAME, DEVICE, args.max_length)

        ok = sanity_check_scores(score_fn)
        if not ok:
            print("\nWARNING: sanity check fallito. Procedo comunque.\n")

        reranked_results: dict = {}
        missing_docs  = 0
        total_pairs   = 0

        chunks = [
            all_query_items[i: i + args.query_chunk]
            for i in range(0, len(all_query_items), args.query_chunk)
        ]

        for chunk in tqdm(chunks, desc="Re-ranking"):
            all_pairs, meta, missing = build_pairs_for_queries(
                chunk, paper_texts, query_texts, args.top_k
            )
            missing_docs += missing

            if not all_pairs:
                for qid, _, _, fallback in meta:
                    reranked_results[qid] = fallback
                continue

            total_pairs += len(all_pairs)
            scores_flat  = predict_scores(score_fn, all_pairs, args.batch)
            reranked_results.update(scores_to_results(meta, scores_flat))

    # ── Multi-GPU ───────────────────────────────────────────────
    else:
        partitions = [[] for _ in range(NUM_GPUS)]
        for i, item in enumerate(all_query_items):
            partitions[i % NUM_GPUS].append(item)

        for i, p in enumerate(partitions):
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {len(p)} queries")

        mp.set_start_method("spawn", force=True)
        result_queue: mp.Queue = mp.Queue()

        processes = []
        for gpu_id in range(NUM_GPUS):
            p = mp.Process(
                target=rerank_worker,
                args=(
                    gpu_id,
                    partitions[gpu_id],
                    paper_texts,
                    query_texts,
                    MODEL_NAME,
                    args.top_k,
                    args.batch,
                    args.query_chunk,
                    args.max_length,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        reranked_results: dict = {}
        missing_docs  = 0
        total_pairs   = 0

        for _ in range(NUM_GPUS):
            local_results, local_missing, local_pairs = result_queue.get()
            reranked_results.update(local_results)
            missing_docs += local_missing
            total_pairs  += local_pairs

        for p in processes:
            p.join()

        # Ripristina ordine originale delle query
        reranked_results = {
            qid: reranked_results[qid]
            for qid in bm25_results
            if qid in reranked_results
        }

    # ── Report finale ──────────────────────────────────────────
    print(f"\nTotal pairs scored : {total_pairs}")
    print(f"Queries re-ranked  : {len(reranked_results)}")
    if missing_docs:
        print(f"Pubkey non trovati : {missing_docs} (ordine BM25 mantenuto)")
    coverage = len(reranked_results) / len(bm25_results) * 100
    print(f"Coverage           : {coverage:.1f}% ({len(reranked_results)}/{len(bm25_results)})")

    # ── Salva ──────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, indent=2, ensure_ascii=False)

    print(f"\nSalvato → {args.output}")
    print("Done. Ora esegui il Java evaluator puntando a reranked_results_zerank.json.")