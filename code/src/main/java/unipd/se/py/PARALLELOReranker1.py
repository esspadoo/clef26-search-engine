"""
Reranker.py — Cross-Encoder re-ranking dei risultati BM25

Legge:
  - data/expanded_queries_4.json   (query originali, per avere il testo originale)
  - data/collection_data.json      (corpus, per recuperare titolo+abstract dei candidati)
  - results/bm25_results.json      (output di Main.java: { qid → [pubkey, ...] })

Scrive:
  - results/reranked_results.json  (stesso formato: { qid → [pubkey, ...] })

Uso:
  python BESTReranker.py
  # oppure con argomenti:
  python BESTReranker.py --queries data/expanded_queries_4.json \
                     --papers  data/collection_data.json \
                     --bm25    results/bm25_results.json \
                     --output  results/reranked_results.json \
                     --top_k   1000
"""

import json
import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ──────────────────────────────────────────────────────────────
# CLI — fuori da __main__ per essere disponibile ai worker spawn
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--queries", default="../../../../../../data/expanded_queries_bge_large.json")
parser.add_argument("--papers",  default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",    default="../../../../../../../results/bm25_results.json")
parser.add_argument("--output",  default="../../../../../../../results/reranked_results.json")
parser.add_argument("--top_k",   type=int, default=100,
                    help="Quanti candidati BM25 passare al re-ranker (default: 100)")
parser.add_argument("--batch",   type=int, default=512,
                    help="Coppie per chiamata GPU. Con 24GB VRAM e MiniLM, 512-1024 satura bene.")
parser.add_argument("--query_chunk", type=int, default=32,
                    help="Quante query raggruppare in un unico batch GPU (default: 32). "
                         "Con top_k=1000 ogni chunk = ~32000 coppie divise in batch da --batch.")

MODEL_NAME = "nvidia/llama-nemotron-rerank-1b-v2"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def build_pairs_for_queries(query_items, paper_texts, query_texts, top_k):
    """
    Pre-costruisce tutte le coppie (query_text, doc_text) per un chunk di query.
    Eseguito sulla CPU mentre la GPU lavora sul chunk precedente.
    Restituisce:
      - all_pairs   : lista piatta di (query_text, doc_text)
      - meta        : [(qid, valid_keys, n_pairs), ...] per ricostruire i risultati
      - missing     : contatore documenti non trovati
    """
    all_pairs = []
    meta = []
    missing = 0

    for qid, candidate_pubkeys in query_items:
        query_text = query_texts.get(qid, "")
        candidates = candidate_pubkeys[:top_k]

        if not query_text:
            meta.append((qid, [], 0, candidates))   # fallback: mantieni ordine BM25
            continue

        pairs = []
        valid_keys = []
        for pk in candidates:
            doc_text = paper_texts.get(str(pk))
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
    Ricostruisce il dizionario qid→[pubkey] a partire dai meta e dallo score array piatto.
    """
    results = {}
    offset = 0
    for qid, valid_keys, n_pairs, fallback_candidates in meta:
        if n_pairs == 0:
            results[qid] = fallback_candidates
            continue
        scores = scores_flat[offset: offset + n_pairs]
        offset += n_pairs
        ranked = sorted(zip(valid_keys, scores), key=lambda x: -x[1])
        results[qid] = [pk for pk, _ in ranked]
    return results


# ──────────────────────────────────────────────────────────────
# Worker per-GPU
# Strategia: invece di chiamare reranker.predict() una volta per
# query (overhead CPU enorme), raggruppa query_chunk query in un
# unico batch piatto e fa UNA sola chiamata GPU per chunk.
# Questo riduce il numero di chiamate GPU da N_queries a
# N_queries/query_chunk, saturando molto meglio la VRAM.
# ──────────────────────────────────────────────────────────────
def rerank_worker(
        gpu_id: int,
        query_items: list,
        paper_texts: dict,
        query_texts: dict,
        top_k: int,
        batch_size: int,
        query_chunk: int,
        result_queue: mp.Queue,
) -> None:
    # Isola questa GPU PRIMA di inizializzare CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda:0"
    print(f"[GPU {gpu_id}] Initializing on {torch.cuda.get_device_name(0)}", flush=True)

    reranker = CrossEncoder(MODEL_NAME, device=device, max_length=512, trust_remote_code=True)

    # Abilita fp16: dimezza la VRAM usata per i tensori di attivazione,
    # permettendo batch più grandi e throughput più alto su Ampere (3090)
    reranker.model.half()

    local_results: dict = {}
    local_missing = 0
    local_pairs_total = 0

    # Suddivide le query in chunk da query_chunk elementi ciascuno
    chunks = [
        query_items[i: i + query_chunk]
        for i in range(0, len(query_items), query_chunk)
    ]

    for chunk in tqdm(chunks, desc=f"Re-ranking [GPU {gpu_id}]", position=gpu_id, leave=True):
        # Costruisce tutte le coppie del chunk in un'unica lista piatta
        all_pairs, meta, missing = build_pairs_for_queries(
            chunk, paper_texts, query_texts, top_k
        )
        local_missing += missing

        if not all_pairs:
            # Nessuna coppia valida nel chunk: fallback BM25 per tutte le query
            for qid, _, _, fallback in meta:
                local_results[qid] = fallback
            continue

        local_pairs_total += len(all_pairs)

        # UNA sola chiamata GPU per tutto il chunk (invece di query_chunk chiamate)
        scores_flat = reranker.predict(
            all_pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        # Ricostruisce i risultati per-query dallo score array piatto
        chunk_results = scores_to_results(meta, scores_flat)
        local_results.update(chunk_results)

    result_queue.put((local_results, local_missing, local_pairs_total))


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parser.parse_args()

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        DEVICE = "cpu"
        print("Device: cpu (no CUDA GPUs found)")
    else:
        DEVICE = "cuda"
        print(f"Device: cuda — {NUM_GPUS} GPU(s) available:")
        for i in range(NUM_GPUS):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    # ── Carica dati ────────────────────────────────────────────
    print("Loading data...")

    with open(args.queries, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)

    with open(args.papers, "r", encoding="utf-8") as f:
        papers_raw = json.load(f)

    with open(args.bm25, "r", encoding="utf-8") as f:
        bm25_results: dict = json.load(f)

    print("Building paper index (normalizing pubkeys to str)...")
    paper_texts: dict = {
        str(p["pubkey"]): (p.get("title", "") + " " + p.get("abstract", "")).strip()
        for p in papers_raw
    }

    query_texts: dict = {
        str(q["index"]): q.get("original", q.get("text", ""))
        for q in queries_raw
    }

    # Diagnostica
    sample_bm25_key = next(iter(bm25_results))
    sample_candidates = bm25_results[sample_bm25_key][:3]
    hits = sum(1 for pk in sample_candidates if str(pk) in paper_texts)
    print(f"Sanity check — query '{sample_bm25_key}': {hits}/{len(sample_candidates)} candidates found in paper_texts")
    if hits == 0:
        print("WARNING: 0 candidates found. Check pubkey types.")

    all_query_items = list(bm25_results.items())
    print(f"\nRe-ranking {len(bm25_results)} queries "
          f"(top_k={args.top_k}, batch={args.batch}, query_chunk={args.query_chunk})...")

    # ── Singola GPU / CPU ───────────────────────────────────────
    if NUM_GPUS <= 1:
        from sentence_transformers import CrossEncoder

        print(f"Loading cross-encoder: {MODEL_NAME} ...")
        reranker = CrossEncoder(MODEL_NAME, device=DEVICE, max_length=512, trust_remote_code=True)
        if DEVICE == "cuda":
            reranker.model.half()

        reranked_results: dict = {}
        missing_docs = 0
        total_pairs = 0

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
            scores_flat = reranker.predict(
                all_pairs, batch_size=args.batch, show_progress_bar=False
            )
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
                    args.top_k,
                    args.batch,
                    args.query_chunk,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        reranked_results: dict = {}
        missing_docs = 0
        total_pairs = 0

        for _ in range(NUM_GPUS):
            local_results, local_missing, local_pairs = result_queue.get()
            reranked_results.update(local_results)
            missing_docs += local_missing
            total_pairs  += local_pairs

        for p in processes:
            p.join()

        reranked_results = {
            qid: reranked_results[qid]
            for qid in bm25_results
            if qid in reranked_results
        }

    print(f"\nTotal pairs scored: {total_pairs}")
    if missing_docs > 0:
        print(f"  Warning: {missing_docs} candidate pubkeys not found in collection (skipped)")

    # ── Salva ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(reranked_results)} re-ranked queries → {args.output}")
    print("Done. Now run the Java evaluator pointing to reranked_results.json.")