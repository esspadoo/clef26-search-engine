"""
prepare_data_nemotron_finetune_v2.py

Prepara il dataset per fine-tuning di nvidia/llama-nemotron-rerank-1b-v2.

═══════════════════════════════════════════════════════════════════
CAMBIAMENTI RISPETTO A v1
═══════════════════════════════════════════════════════════════════

[FIX 1] SOFT NEGATIVES aggiunti (campo "neg" separato da "hard_neg")
  In v1 il dataset aveva solo hard negatives (top-k di Nemotron base).
  Avere solo hard negatives crea un training instabile: la loss parte
  già da casi difficilissimi senza gradiente "facile" che la ancori.
  Ora il dataset ha due tipi di negativi:
    - "hard_neg": rank [neg_offset .. neg_offset+num_hard_neg] del run Nemotron
      → i documenti che il modello sbaglia già, massimo segnale
    - "neg":      rank [soft_neg_offset .. soft_neg_offset+num_soft_neg]
      → documenti a metà classifica, abbastanza distanti dal positivo
        ma non triviali. Stabilizzano il training nelle prime iterazioni.
  Il finetune_nemotron_reranker_v3.py già gestisce entrambi i campi
  (hard_neg con oversampling x2, neg con peso normale).

[FIX 2] neg_offset di default spostato: 0 → 1
  In v1 il primo hard negative era rank=1 (il documento immediatamente
  dopo il positivo). Se il positivo non è nel run (raro ma possibile),
  rank=0 potrebbe essere un altro documento rilevante non annotato.
  Partire da offset=1 aggiunge un margine di sicurezza.
  ATTENZIONE: se usi nemotron_run con il positivo già escluso, lascia
  neg_offset=0.

[FIX 3] soft_neg_offset e num_soft_neg configurabili
  Default: soft_neg_offset=10, num_soft_neg=5
  → Prende i documenti in posizione 10-15 del ranking di Nemotron base.
  Questi sono abbastanza lontani dal positivo da essere quasi sicuramente
  non rilevanti, ma non così lontani da essere triviali (es. topic
  completamente diverso). La posizione 10-15 è empiricamente il
  "punto medio" più pulito per CT26 con corpus ~10k documenti.

[FIX 4] Verifica domain shift 2025 vs 2026
  Aggiunta analisi statistica della lunghezza query e overlap del
  vocabolario tra i due anni. Se il cosine overlap < 0.5, viene
  emesso un warning esplicito che suggerisce di escludere i dati 2025.
  Questo riflette la preoccupazione discussa: i dati 2025 potrebbero
  introdurre distribuzione distorta se il topic è significativamente
  diverso.

[FIX 5] Statistiche di qualità del dataset più dettagliate
  stats.json ora include:
    - overlap rate: quante query hanno il positivo nel run Nemotron
      (se < 0.9 il run BM25 upstream ha un problema di recall)
    - hard_neg_avg_rank: rank medio degli hard negatives selezionati
      (se vicino a 1.0 il modello base è già molto calibrato)
    - domain_shift_warning: flag booleano
    - per_year breakdown delle statistiche

[FIX 6] Shuffle stratificato per anno nel train set
  In v1 il shuffle finale mescolava tutto casualmente. Con molti più
  dati 2025 (15699) che 2026 (13480), nelle prime batch il modello
  vedeva prevalentemente dati 2025. Ora si usa uno shuffle interleaved
  che garantisce proporzione costante 2025/2026 in ogni finestra di batch.

[INVARIATO] Formato output compatibile con finetune_nemotron_reranker_v3.py
  I campi "pos", "neg", "hard_neg" sono già gestiti dal trainer v3.
  Nessuna modifica necessaria al trainer per usare questo dataset.

═══════════════════════════════════════════════════════════════════

USO minimo (solo 2026):
  python prepare_data_nemotron_finetune_v2.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400.json

USO completo con 2025:
  python3 prepare_data_nemotron_finetune_v2.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400_entrain2026.json \
    --collection_2025     collection_data2025.json \
    --queries_2025        subtask4b_query_tweets_train.json \
                          subtask4b_query_tweets_dev.json \
                          subtask4b_query_tweets_test_gold.json \
    --nemotron_runs_2025  reranked_results_nemotron_topk140_tweets_train2025_BiencoderTopk1000.json \
                          reranked_results_nemotron_topk140_tweets_dev2025_BiencoderTopk1000.json \
                          reranked_results_nemotron_topk140_tweets_test_gold2025_BiencoderTopk1000.json

USO con solo dati 2026 (raccomandato se domain shift confermato):
  python3 prepare_data_nemotron_finetune_v2.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400_entrain2026.json \
    --no_2025
"""

import json
import re
import random
import logging
import argparse
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Pulizia allineata con il pipeline Java/Lucene.
    Nessun prefisso — Nemotron non ne usa.
    """
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


# ══════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════

def load_queries_and_gt(path: str) -> tuple[dict[str, str], dict[str, str]]:
    data = json.load(open(path, encoding="utf-8"))
    queries, gt = {}, {}
    for item in data:
        qid           = str(item["index"])
        queries[qid]  = clean_text(item["text"])
        gt[qid]       = str(item["pubkey"])
    log.info(f"    {Path(path).name}: {len(queries):,} query")
    return queries, gt


def load_corpus(path: str, max_chars: int, label: str = "") -> dict[str, str]:
    """
    Carica corpus JSON con campo "pubkey".
    Compatibile con collection_data.json (pubkey int)
    e collection_data2025.json (pubkey str).
    """
    data = json.load(open(path, encoding="utf-8"))
    c = {}
    for d in data:
        key  = str(d["pubkey"])
        text = _doc_text(d, max_chars)
        if text:
            c[key] = text
    log.info(f"  Corpus {label} ({Path(path).name}): {len(c):,} documenti")
    return c


def _doc_text(item: dict, max_chars: int) -> str:
    t = (item.get("title")    or "").strip()
    a = (item.get("abstract") or "").strip()
    return f"{t}. {a}".strip(". ")[:max_chars]


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    raw   = json.load(open(path, encoding="utf-8"))
    run   = {str(k): [str(d) for d in v] for k, v in raw.items()}
    total = sum(len(v) for v in run.values())
    log.info(f"    Nemotron run {Path(path).name}: {len(run):,} query, {total:,} doc")
    return run


# ══════════════════════════════════════════════════════════════════
# Analisi domain shift
# ══════════════════════════════════════════════════════════════════

def _tokenize_simple(text: str) -> set[str]:
    """Tokenizzazione grezza per analisi vocabolario — non per training."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def analyze_domain_shift(
        queries_2026: dict[str, str],
        queries_2025: dict[str, str],
        threshold: float = 0.5,
) -> bool:
    """
    [FIX 4] Stima il domain shift tra le query 2025 e 2026 tramite
    Jaccard overlap del vocabolario.

    Jaccard = |V_2025 ∩ V_2026| / |V_2025 ∪ V_2026|

    Se Jaccard < threshold (default 0.5) → warning: i dati 2025
    potrebbero introdurre distribuzione distorta nel training.

    Nota: il threshold 0.5 è conservativo. Per corpus biomedici sullo
    stesso dominio (COVID) ci aspettiamo Jaccard > 0.6. Valori < 0.5
    indicano un cambio di topic sostanziale.
    """
    vocab_2026 = set()
    for q in queries_2026.values():
        vocab_2026.update(_tokenize_simple(q))

    vocab_2025 = set()
    for q in queries_2025.values():
        vocab_2025.update(_tokenize_simple(q))

    intersection = len(vocab_2026 & vocab_2025)
    union        = len(vocab_2026 | vocab_2025)
    jaccard      = intersection / union if union > 0 else 0.0

    avg_len_2026 = sum(len(q.split()) for q in queries_2026.values()) / max(1, len(queries_2026))
    avg_len_2025 = sum(len(q.split()) for q in queries_2025.values()) / max(1, len(queries_2025))

    log.info("  ── Analisi domain shift 2025 vs 2026 ──")
    log.info(f"    Vocabolario 2026: {len(vocab_2026):,} token unici")
    log.info(f"    Vocabolario 2025: {len(vocab_2025):,} token unici")
    log.info(f"    Jaccard overlap:  {jaccard:.3f}  (soglia={threshold})")
    log.info(f"    Lunghezza media query — 2026: {avg_len_2026:.1f} tok | 2025: {avg_len_2025:.1f} tok")

    if jaccard < threshold:
        log.warning(
            f"  ⚠  DOMAIN SHIFT RILEVATO (Jaccard={jaccard:.3f} < {threshold}). "
            "I dati 2025 potrebbero degradare le performance sul dev 2026. "
            "Considera --no_2025 se il fine-tuning peggiora il baseline."
        )
        return True
    else:
        log.info(f"  ✓ Nessun domain shift significativo (Jaccard={jaccard:.3f} ≥ {threshold})")
        return False


# ══════════════════════════════════════════════════════════════════
# Costruzione gruppi
# ══════════════════════════════════════════════════════════════════

def build_groups(
        queries:         dict[str, str],
        corpus:          dict[str, str],
        gt:              dict[str, str],
        nemotron_run:    dict[str, list[str]],
        num_hard_neg:    int,
        neg_offset:      int,
        num_soft_neg:    int,
        soft_neg_offset: int,
        tag:             str,
) -> tuple[list[dict], dict]:
    """
    Costruisce gruppi di training nel formato:
      {
        "query":    str,
        "pos":      [str],        # 1 documento positivo (ground truth)
        "hard_neg": [str, ...],   # top-k errori di Nemotron base  [FIX 1]
        "neg":      [str, ...],   # soft negatives a metà classifica [FIX 1]
        "_qid":     str,          # metadata, rimosso al salvataggio
        "_pos_id":  str,
      }

    [FIX 1] PERCHÉ DUE LISTE SEPARATE:
    Il trainer v3 legge "hard_neg" e li oversampla (x2) perché sono
    i casi più informativi. I "neg" soft vengono inclusi con peso
    normale per stabilizzare il gradiente nelle prime iterazioni.
    Avere solo hard negatives equivale ad allenarsi sempre su esempi
    al limite della capacità del modello — buono a regime, instabile
    all'inizio.

    [FIX 2] neg_offset=1 di default:
    Il documento a rank=0 nel run (dopo aver escluso il positivo) è
    il più simile alla query secondo Nemotron. In rari casi potrebbe
    essere un documento rilevante non annotato (falso negativo nel
    ground truth). Partire da offset=1 aggiunge un margine.

    [FIX 3] soft_neg_offset=10:
    Rank 10-15 sono abbastanza lontani dal positivo da essere quasi
    certamente non rilevanti, ma non così lontani da essere triviali.
    Per un corpus di ~10k documenti con 400 candidati BM25, rank 10-15
    è il "punto medio" empiricamente più pulito.
    """
    groups        = []
    stats         = Counter()
    rank_sum_hard = 0
    rank_count    = 0

    for qid, query_text in queries.items():
        pos_id = gt.get(qid)
        if pos_id is None:
            stats["no_gt"] += 1
            continue
        if pos_id not in corpus:
            stats["pos_missing"] += 1
            continue
        if qid not in nemotron_run:
            stats["no_run"] += 1
            continue

        # Candidati negativi: tutti i doc nel run escluso il positivo
        candidates = [
            d for d in nemotron_run[qid]
            if d != pos_id and d in corpus
        ]

        if len(candidates) < num_hard_neg + 2:
            stats["few_negs"] += 1
            continue

        # [FIX 1 + FIX 2] Hard negatives: top-k con offset
        hard_start = neg_offset
        hard_end   = neg_offset + num_hard_neg
        hard_negs  = candidates[hard_start:hard_end]

        # Fallback se offset supera i candidati disponibili
        if len(hard_negs) < num_hard_neg:
            hard_negs = candidates[:num_hard_neg]

        # [FIX 1 + FIX 3] Soft negatives: posizioni a metà classifica
        soft_start = soft_neg_offset
        soft_end   = soft_neg_offset + num_soft_neg
        soft_negs  = candidates[soft_start:soft_end]

        # Se non ci sono abbastanza candidati per i soft neg, skippa i soft
        # (non è un errore fatale — meglio avere meno soft che zero gruppi)
        if len(soft_negs) == 0:
            stats["no_soft_neg"] += 1
            soft_negs = []

        # Verifica che hard e soft non si sovrappongano
        # (può capitare se neg_offset + num_hard_neg > soft_neg_offset)
        hard_set = set(hard_negs)
        soft_negs = [d for d in soft_negs if d not in hard_set]

        # Statistiche rank degli hard negatives per FIX 5
        for d in hard_negs:
            try:
                rank = nemotron_run[qid].index(d)
                rank_sum_hard += rank
                rank_count    += 1
            except ValueError:
                pass

        groups.append({
            "query":    query_text,
            "pos":      [corpus[pos_id]],
            "hard_neg": [corpus[d] for d in hard_negs],
            "neg":      [corpus[d] for d in soft_negs],
            "_qid":     f"{tag}_{qid}",
            "_pos_id":  pos_id,
        })
        stats["ok"] += 1

    # Statistiche qualità [FIX 5]
    overlap_rate      = stats["ok"] / max(1, len(queries))
    hard_neg_avg_rank = rank_sum_hard / max(1, rank_count)

    log.info(
        f"  [{tag}] {stats['ok']:,} gruppi "
        f"({num_hard_neg} hard + {num_soft_neg} soft neg) | "
        f"overlap_rate={overlap_rate:.2%} | "
        f"hard_neg_avg_rank={hard_neg_avg_rank:.1f} | "
        f"skip: no_gt={stats['no_gt']} pos_missing={stats['pos_missing']} "
        f"no_run={stats['no_run']} few_negs={stats['few_negs']} "
        f"no_soft={stats['no_soft_neg']}"
    )

    if overlap_rate < 0.85:
        log.warning(
            f"  ⚠  overlap_rate={overlap_rate:.2%} < 85% per [{tag}]. "
            "Controlla che il run Nemotron copra la maggior parte delle query."
        )
    if hard_neg_avg_rank < 2.0:
        log.warning(
            f"  ⚠  hard_neg_avg_rank={hard_neg_avg_rank:.1f} molto basso. "
            "Il positivo potrebbe essere già escluso dal run — verifica neg_offset."
        )

    quality_stats = {
        "groups":             stats["ok"],
        "overlap_rate":       round(overlap_rate, 4),
        "hard_neg_avg_rank":  round(hard_neg_avg_rank, 2),
        "skip_no_gt":         stats["no_gt"],
        "skip_pos_missing":   stats["pos_missing"],
        "skip_no_run":        stats["no_run"],
        "skip_few_negs":      stats["few_negs"],
        "skip_no_soft":       stats["no_soft_neg"],
    }
    return groups, quality_stats


# ══════════════════════════════════════════════════════════════════
# Split train/dev
# ══════════════════════════════════════════════════════════════════

def split_2026_by_query(
        groups_2026: list[dict],
        dev_ratio:   float,
        seed:        int,
) -> tuple[list[dict], list[dict]]:
    """
    Split stratificato per query ID.
    Tutte le righe della stessa query vanno nello stesso split
    (train o dev) — evita data leakage.
    """
    by_qid = defaultdict(list)
    for g in groups_2026:
        by_qid[g["_qid"]].append(g)

    qids = list(by_qid.keys())
    random.Random(seed).shuffle(qids)

    n_dev      = max(10, int(len(qids) * dev_ratio))
    dev_qids   = set(qids[:n_dev])
    train_qids = set(qids[n_dev:])

    train = [g for qid in train_qids for g in by_qid[qid]]
    dev   = [g for qid in dev_qids   for g in by_qid[qid]]

    log.info(f"  Split 2026: {len(train):,} train / {len(dev):,} dev gruppi")
    return train, dev


def interleaved_shuffle(
        groups_2026: list[dict],
        groups_2025: list[dict],
        seed:        int,
) -> list[dict]:
    """
    [FIX 6] Shuffle interleaved che mantiene proporzione 2026/2025 costante.

    Invece di concatenare e mescolare casualmente (che porta a batch
    iniziali dominati dal gruppo più grande), interleava i due dataset
    in proporzione, poi shuffla localmente in finestre di dimensione
    fissa. Questo garantisce che ogni finestra di batch_size esempi
    abbia una rappresentazione bilanciata dei due anni.

    Esempio con ratio 1:1.16 (13480 vs 15699 come nel tuo caso):
      [2026, 2025, 2026, 2025, 2026, 2025, 2026, 2026, ...]
    invece di:
      [2026, 2026, ..., 2026, 2025, 2025, ..., 2025]
    """
    rng = random.Random(seed)

    g26 = groups_2026[:]
    g25 = groups_2025[:]
    rng.shuffle(g26)
    rng.shuffle(g25)

    if not g25:
        return g26

    # Calcola ratio: per ogni N documenti 2026, quanti 2025 inserire
    ratio = len(g25) / max(1, len(g26))  # es. 15699/13480 ≈ 1.16

    result  = []
    idx_25  = 0
    acc_25  = 0.0

    for item_26 in g26:
        result.append(item_26)
        acc_25 += ratio
        while acc_25 >= 1.0 and idx_25 < len(g25):
            result.append(g25[idx_25])
            idx_25  += 1
            acc_25  -= 1.0

    # Eventuali 2025 rimanenti in coda
    result.extend(g25[idx_25:])

    # Shuffle locale in finestre di 128 per non distruggere il bilanciamento
    window = 128
    for start in range(0, len(result), window):
        chunk = result[start : start + window]
        rng.shuffle(chunk)
        result[start : start + window] = chunk

    return result


# ══════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════

def save_jsonl(groups: list[dict], path: Path):
    """Salva i gruppi rimuovendo i campi metadata (prefisso '_')."""
    with open(path, "w", encoding="utf-8") as f:
        for g in groups:
            out = {k: v for k, v in g.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    log.info(f"  → {path.name}: {len(groups):,} gruppi salvati")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Prepara dati per fine-tuning llama-nemotron-rerank-1b-v2 v2"
    )

    g26 = p.add_argument_group("Corpus 2026 (obbligatorio)")
    g26.add_argument("--collection_2026",   required=True)
    g26.add_argument("--queries_2026",      required=True)
    g26.add_argument("--nemotron_run_2026", required=True)

    g25 = p.add_argument_group("Corpus 2025 (opzionale)")
    g25.add_argument("--collection_2025",    default=None)
    g25.add_argument("--queries_2025",       nargs="+", default=[])
    g25.add_argument("--nemotron_runs_2025", nargs="+", default=[])
    g25.add_argument(
        "--no_2025",
        action="store_true",
        help="Ignora completamente i dati 2025 (raccomandato se domain shift confermato)",
    )

    p.add_argument("--out_dir",          default="training_data_nemotron_ft_v2")
    p.add_argument(
        "--num_hard_neg",
        type=int, default=5,
        help="Numero hard negatives (top-k errori Nemotron base). Default: 5",
    )
    p.add_argument(
        "--neg_offset",
        type=int, default=1,                    # [FIX 2] era 0
        help="Offset iniziale per gli hard negatives nel run. Default: 1",
    )
    p.add_argument(
        "--num_soft_neg",
        type=int, default=5,                    # [FIX 1]
        help="Numero soft negatives (metà classifica). Default: 5",
    )
    p.add_argument(
        "--soft_neg_offset",
        type=int, default=10,                   # [FIX 3]
        help="Posizione di partenza per i soft negatives nel run. Default: 10",
    )
    p.add_argument("--dev_ratio",       type=float, default=0.1)
    p.add_argument("--max_doc_chars",   type=int,   default=4000)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument(
        "--domain_shift_threshold",
        type=float, default=0.5,
        help="Soglia Jaccard per warning domain shift. Default: 0.5",
    )
    args = p.parse_args()

    # Validazione argomenti 2025
    if not args.no_2025 and args.queries_2025 and args.nemotron_runs_2025:
        if len(args.queries_2025) != len(args.nemotron_runs_2025):
            p.error(
                f"--queries_2025 ha {len(args.queries_2025)} file ma "
                f"--nemotron_runs_2025 ne ha {len(args.nemotron_runs_2025)}. "
                "Devono corrispondere 1:1."
            )

    # Validazione offset: hard e soft non devono sovrapporsi
    hard_end = args.neg_offset + args.num_hard_neg
    if hard_end > args.soft_neg_offset:
        log.warning(
            f"  ⚠  hard negatives [{args.neg_offset}:{hard_end}] e "
            f"soft negatives [{args.soft_neg_offset}:...] si sovrappongono. "
            f"Considera di aumentare --soft_neg_offset a {hard_end + 2} o superiore."
        )

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_quality_stats  = {}
    domain_shift_found = False

    # ── 1. Corpus e query 2026 ────────────────────────────────────
    log.info("══ Corpus 2026 ══════════════════════════════════════════════")
    corpus_2026           = load_corpus(args.collection_2026, args.max_doc_chars, "2026")
    queries_2026, gt_2026 = load_queries_and_gt(args.queries_2026)
    nemotron_2026         = load_nemotron_run(args.nemotron_run_2026)

    groups_2026, qstats_2026 = build_groups(
        queries      = queries_2026,
        corpus       = corpus_2026,
        gt           = gt_2026,
        nemotron_run = nemotron_2026,
        num_hard_neg    = args.num_hard_neg,
        neg_offset      = args.neg_offset,
        num_soft_neg    = args.num_soft_neg,
        soft_neg_offset = args.soft_neg_offset,
        tag             = "2026",
    )
    all_quality_stats["2026"] = qstats_2026

    train_2026, dev_2026 = split_2026_by_query(groups_2026, args.dev_ratio, args.seed)

    # ── 2. Corpus e query 2025 (opzionale) ───────────────────────
    train_2025 = []
    n25        = 0

    use_2025 = (
            not args.no_2025
            and args.collection_2025
            and args.queries_2025
            and args.nemotron_runs_2025
    )

    if args.no_2025:
        log.info("══ Dati 2025 SKIPPATI (--no_2025) ══════════════════════════")
    elif use_2025:
        log.info("══ Corpus 2025 ══════════════════════════════════════════════")
        corpus_2025 = load_corpus(args.collection_2025, args.max_doc_chars, "2025")

        # Raccogli tutte le query 2025 per analisi domain shift
        all_queries_2025: dict[str, str] = {}
        all_groups_2025_per_file: list[list[dict]] = []

        for q_path, n_path in zip(args.queries_2025, args.nemotron_runs_2025):
            stem = Path(q_path).stem
            log.info(f"  Dataset: {stem}")
            queries_q, gt_q = load_queries_and_gt(q_path)
            all_queries_2025.update(queries_q)
            nemotron_q = load_nemotron_run(n_path)
            groups_q, qstats_q = build_groups(
                queries      = queries_q,
                corpus       = corpus_2025,
                gt           = gt_q,
                nemotron_run = nemotron_q,
                num_hard_neg    = args.num_hard_neg,
                neg_offset      = args.neg_offset,
                num_soft_neg    = args.num_soft_neg,
                soft_neg_offset = args.soft_neg_offset,
                tag             = f"2025_{stem}",
            )
            all_quality_stats[f"2025_{stem}"] = qstats_q
            all_groups_2025_per_file.append(groups_q)

        # [FIX 4] Analisi domain shift
        domain_shift_found = analyze_domain_shift(
            queries_2026 = queries_2026,
            queries_2025 = all_queries_2025,
            threshold    = args.domain_shift_threshold,
        )

        for groups_q in all_groups_2025_per_file:
            train_2025.extend(groups_q)
        n25 = len(train_2025)

    elif not args.no_2025 and args.queries_2025:
        log.warning("⚠  --nemotron_runs_2025 non fornito: dati 2025 SKIPPATI.")

    # ── 3. Composizione train finale ─────────────────────────────
    # [FIX 6] Shuffle interleaved invece di concatenazione + shuffle random
    final_train = interleaved_shuffle(train_2026, train_2025, args.seed)
    final_dev   = dev_2026

    if not final_train:
        log.error("Nessun gruppo costruito. Controlla i file di input.")
        return

    # ── 4. Salvataggio ────────────────────────────────────────────
    log.info("══ Composizione finale ══════════════════════════════════════")
    log.info(f"  Training: {len(final_train):,} gruppi  (2026={len(train_2026):,} + 2025={n25:,})")
    log.info(f"  Dev (solo 2026): {len(final_dev):,} gruppi")

    save_jsonl(final_train, out_dir / "train_groups.jsonl")
    save_jsonl(final_dev,   out_dir / "dev_groups.jsonl")

    # ── 5. Stats.json arricchito [FIX 5] ─────────────────────────
    tgs = 1 + args.num_hard_neg + args.num_soft_neg
    stats = {
        "train_groups":           len(final_train),
        "dev_groups":             len(final_dev),
        "train_from_2026":        len(train_2026),
        "train_from_2025":        n25,
        "dev_from_2026_only":     True,
        "num_hard_neg":           args.num_hard_neg,
        "num_soft_neg":           args.num_soft_neg,          # [FIX 1]
        "neg_offset":             args.neg_offset,
        "soft_neg_offset":        args.soft_neg_offset,       # [FIX 3]
        "train_group_size_for_training": tgs,
        "domain_shift_warning":   domain_shift_found,         # [FIX 4]
        "quality_per_split":      all_quality_stats,          # [FIX 5]
        "note": (
            "Nemotron NON usa prefissi query. Solo pulizia testo. "
            f"train_group_size={tgs}. "
            f"hard_neg=errori modello base (rank {args.neg_offset}-{args.neg_offset+args.num_hard_neg}). "
            f"soft_neg=metà classifica (rank {args.soft_neg_offset}-{args.soft_neg_offset+args.num_soft_neg})."
        ),
    }
    stats_path = out_dir / "stats.json"
    json.dump(stats, open(stats_path, "w"), indent=2)
    log.info(f"  → stats.json salvato")

    # ── 6. Riepilogo finale ───────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("✓ Dataset pronto")
    log.info(f"  Output dir:          {out_dir}/")
    log.info(f"  train_groups.jsonl : {len(final_train):,} righe")
    log.info(f"  dev_groups.jsonl   : {len(final_dev):,} righe")
    log.info(f"  Hard neg per query : {args.num_hard_neg} (rank {args.neg_offset}-{args.neg_offset+args.num_hard_neg})")
    log.info(f"  Soft neg per query : {args.num_soft_neg} (rank {args.soft_neg_offset}-{args.soft_neg_offset+args.num_soft_neg})")
    if domain_shift_found:
        log.warning(
            "  ⚠  Domain shift rilevato — considera di rieseguire con --no_2025 "
            "e confrontare le metriche sul dev set."
        )
    log.info("═" * 60)
    log.info("\nProssimo step:")
    log.info(f"  python finetune_nemotron_reranker_v3.py \\")
    log.info(f"    --train_file {out_dir}/train_groups.jsonl \\")
    log.info(f"    --dev_file   {out_dir}/dev_groups.jsonl \\")
    log.info(f"    --output_dir models/nemotron-rerank-1b-retrix-v3")
    log.info(f"\n  # Oppure senza dati 2025 se domain shift confermato:")
    log.info(f"  python prepare_data_nemotron_finetune_v2.py \\")
    log.info(f"    --collection_2026   collection_data.json \\")
    log.info(f"    --queries_2026      en_train.json \\")
    log.info(f"    --nemotron_run_2026 reranked_results_nemotron_topk400.json \\")
    log.info(f"    --no_2025")


if __name__ == "__main__":
    main()