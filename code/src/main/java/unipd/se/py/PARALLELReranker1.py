"""
Reranker.py — Cross-Encoder re-ranking dei risultati BM25
Funziona con più GPU CUDA in parallelo

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
                     --top_k   100
"""

import json
import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CLI — fuori da __main__ per essere disponibile ai worker spawn
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--queries", default="../../../../../../data/expanded_queries_4.json")
parser.add_argument("--papers",  default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",    default="../../../../../../../results/bm25_results.json")
parser.add_argument("--output",  default="../../../../../../../results/reranked_results.json")
parser.add_argument("--top_k",   type=int, default=100,
                    help="Quanti candidati BM25 passare al re-ranker (default: 100)")
parser.add_argument("--batch",   type=int, default=2048,
                    help="Batch size per il cross-encoder (default: 2048, aumentato per saturare la VRAM)")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ──────────────────────────────────────────────────────────────
# Worker per-GPU
# CRITICO: os.environ["CUDA_VISIBLE_DEVICES"] viene impostato
# come PRIMA cosa nel worker, prima di qualsiasi import torch/
# sentence_transformers. Questo garantisce che ogni processo
# veda solo la sua GPU e non possa accidentalmente usare cuda:0
# anche se gpu_id != 0. È l'unico modo affidabile per isolare
# i processi CUDA con start_method="spawn".
# ──────────────────────────────────────────────────────────────
def rerank_worker(
        gpu_id: int,
        query_items: list,
        paper_texts: dict,
        query_texts: dict,
        top_k: int,
        batch_size: int,
        result_queue: mp.Queue,
) -> None:
    # Isola questa GPU PRIMA di inizializzare CUDA
    # Dopo questo, "cuda:0" in questo processo == gpu_id fisico
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import ritardato: devono avvenire DOPO aver impostato CUDA_VISIBLE_DEVICES
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda:0"   # sempre 0 nel contesto isolato di questo processo
    print(f"[GPU {gpu_id}] Initializing on {torch.cuda.get_device_name(0)}", flush=True)

    reranker = CrossEncoder(MODEL_NAME, device=device, max_length=512)

    local_results: dict = {}
    local_missing = 0
    local_pairs_total = 0

    for qid, candidate_pubkeys in tqdm(
            query_items,
            desc=f"Re-ranking [GPU {gpu_id}]",
            position=gpu_id,
            leave=True,
    ):
        query_text = query_texts.get(qid, "")
        if not query_text:
            local_results[qid] = candidate_pubkeys[:top_k]
            continue

        candidates = candidate_pubkeys[:top_k]
        pairs = []
        valid_keys = []

        for pk in candidates:
            doc_text = paper_texts.get(str(pk))
            if doc_text:
                pairs.append((query_text, doc_text))
                valid_keys.append(pk)
            else:
                local_missing += 1

        if not pairs:
            local_results[qid] = candidates
            continue

        local_pairs_total += len(pairs)
        scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        ranked = sorted(zip(valid_keys, scores), key=lambda x: -x[1])
        local_results[qid] = [pk for pk, _ in ranked]

    result_queue.put((local_results, local_missing, local_pairs_total))


# ──────────────────────────────────────────────────────────────
# Entry point — tutto ciò che lancia processi figli sta qui.
# Con start_method="spawn" questo guard è obbligatorio.
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
    print(f"\nRe-ranking {len(bm25_results)} queries (top_k={args.top_k}, batch={args.batch})...")

    # ── Singola GPU / CPU ───────────────────────────────────────
    if NUM_GPUS <= 1:
        from sentence_transformers import CrossEncoder

        print(f"Loading cross-encoder: {MODEL_NAME} ...")
        reranker = CrossEncoder(MODEL_NAME, device=DEVICE, max_length=512)

        reranked_results: dict = {}
        missing_docs = 0
        total_pairs = 0

        for qid, candidate_pubkeys in tqdm(all_query_items, desc="Re-ranking"):
            query_text = query_texts.get(qid, "")
            if not query_text:
                reranked_results[qid] = candidate_pubkeys[:args.top_k]
                continue

            candidates = candidate_pubkeys[:args.top_k]
            pairs = []
            valid_keys = []

            for pk in candidates:
                doc_text = paper_texts.get(str(pk))
                if doc_text:
                    pairs.append((query_text, doc_text))
                    valid_keys.append(pk)
                else:
                    missing_docs += 1

            if not pairs:
                reranked_results[qid] = candidates
                continue

            total_pairs += len(pairs)
            scores = reranker.predict(pairs, batch_size=args.batch, show_progress_bar=False)

            ranked = sorted(zip(valid_keys, scores), key=lambda x: -x[1])
            reranked_results[qid] = [pk for pk, _ in ranked]

    # ── Multi-GPU ───────────────────────────────────────────────
    else:
        # Round-robin per bilanciare query di lunghezze diverse
        partitions = [[] for _ in range(NUM_GPUS)]
        for i, item in enumerate(all_query_items):
            partitions[i % NUM_GPUS].append(item)

        for i, p in enumerate(partitions):
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {len(p)} queries")

        # "spawn" è obbligatorio con CUDA
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

        # Ripristina ordine originale
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