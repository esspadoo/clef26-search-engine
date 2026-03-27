"""
QueryExpansor — COMPATIBILE CON CUDA (NVIDIA GPU) MIGLIORE RISPETTO A "QueryExpansor" SU RECALL ALTE (VEDI "evaluation_resultsCUDA.json")
"""

import json
import re
import unicodedata
import spacy
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import os

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_BATCH_SIZE = 512
QUERY_BATCH_SIZE  = 64

# FIX 3: ridotti rispetto alla run3 (15, 20) → meno rumore BM25
TOP_K_DOCS        = 5    # stesso valore della run1 (migliore recall@1)
TOP_K_TERMS       = 15   # termini di espansione max

SPACY_BATCH_SIZE  = 128
DATA_BASE         = "../../../../../../data"

print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────
# Lucene stop words (da EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)
# Usate SOLO per filtrare i termini di espansione grezzi prima
# di aggiungerli alla query — Lucene li filtrerebbe comunque,
# ma è meglio rimuoverli subito per non sprecare spazio query.
# ──────────────────────────────────────────────────────────────
LUCENE_STOPS: set[str] = {
    "a","an","and","are","as","at","be","but","by","for","if","in","into",
    "is","it","no","not","of","on","or","such","that","the","their","then",
    "there","these","they","this","to","was","will","with",
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","them","themselves","what","which","who",
    "whom","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","nor","only","own","same","so","than","too","very",
    "s","t","can","just","don","should","now","d","ll","m","o","re","ve","y",
    "ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
}

# ──────────────────────────────────────────────────────────────
# Tokenizer leggero per estrarre token grezzi dal corpus
# (usato SOLO per la frequenza nell'expansion pool, non stemmato)
# ──────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:['\-][a-zA-Z0-9]+)*")

def raw_tokenize(text: str) -> list[str]:
    """
    Tokenizza il testo in token alfanumerici minuscoli,
    rimuovendo stopwords Lucene e token corti.
    I token sono nella forma GREZZA (non stemmata) — Lucene
    li stemmerà durante il parsing della query espansa.
    """
    tokens = _TOKEN_RE.findall(text)
    result: list[str] = []
    for t in tokens:
        t = t.lstrip("#").lower()
        if t.endswith("'s"):
            t = t[:-2]
        if t in LUCENE_STOPS:
            continue
        if len(t) < 3:  # più conservativo di run3 (>= 2) per ridurre rumore
            continue
        result.append(t)
    return result

# ──────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────
print("Loading spaCy (tagger only)...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

print(f"Loading SBERT on {DEVICE}...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
if DEVICE == "cuda":
    model.half()   # fp16 → +50-100% throughput su RTX 3090

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
print("Loading data...")
with open(f"{DATA_BASE}/collection_data.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
with open(f"{DATA_BASE}/en_train.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

corpus_texts: list[str] = [
    (p.get("title", "") + " " + p.get("abstract", "")).strip()
    for p in papers
]

# ──────────────────────────────────────────────────────────────
# Pre-tokenize corpus in forma GREZZA (no stemming)
# FIX 2: i token vengono mandati a Lucene non stemmati →
# Lucene applica MyEnglishAnalyzer → no double-stemming
# ──────────────────────────────────────────────────────────────
print("Pre-tokenizing corpus (raw, no stemming)...")
corpus_raw_tokens: list[list[str]] = [
    raw_tokenize(t) for t in tqdm(corpus_texts, desc="Tokenizing corpus")
]

# ──────────────────────────────────────────────────────────────
# Encode corpus (GPU, batch)
# ──────────────────────────────────────────────────────────────
print("Encoding corpus (GPU)...")
corpus_embeddings = model.encode(
    corpus_texts,
    batch_size=CORPUS_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
)

# ──────────────────────────────────────────────────────────────
# Query texts (originali, no normalizzazione)
# ──────────────────────────────────────────────────────────────
query_texts: list[str] = [q.get("text", "") for q in queries]

# ──────────────────────────────────────────────────────────────
# Encode queries (GPU, batch)
# ──────────────────────────────────────────────────────────────
print("Encoding queries (GPU)...")
query_embeddings = model.encode(
    query_texts,
    batch_size=QUERY_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
)

# ──────────────────────────────────────────────────────────────
# Semantic search — una sola chiamata su matrice (GPU)
# ──────────────────────────────────────────────────────────────
print("Semantic search (GPU)...")
all_hits = util.semantic_search(
    query_embeddings,
    corpus_embeddings,
    top_k=TOP_K_DOCS,
)

# ──────────────────────────────────────────────────────────────
# Keyword extraction dalla query originale via spaCy (batch)
# Stessa logica della run1: NOUN/PROPN lemmatizzati, forma leggibile
# ──────────────────────────────────────────────────────────────
print("Extracting keywords from queries (spaCy batch)...")
query_docs = list(nlp.pipe(query_texts, batch_size=SPACY_BATCH_SIZE, n_process=1))

def extract_query_keywords(doc: spacy.tokens.Doc) -> str:
    """
    Estrae NOUN/PROPN dalla query e li restituisce come stringa
    nella forma LEMMA (non stemmata Snowball) — questo è ciò che
    fa la run1 e che performa meglio.
    """
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 2
    ]
    return " ".join(dict.fromkeys(keywords))  # dedup preserving order

# ──────────────────────────────────────────────────────────────
# Assemble expanded queries
# ──────────────────────────────────────────────────────────────
print("Assembling expanded queries...")
expanded_queries: list[dict] = []

for i, q in enumerate(tqdm(queries, desc="Expanding")):
    original  = query_texts[i]
    query_doc = query_docs[i]

    # 1. Keywords dalla query: lemmi spaCy (NOUN/PROPN), forma leggibile
    keywords = extract_query_keywords(query_doc)

    # 2. Termini di espansione dai top-K documenti in forma GREZZA
    #    FIX 4: peso = score float (non round(score*10)) per distribuzione uniforme
    expansion_pool: list[str] = []
    for hit in all_hits[i]:
        cid   = hit["corpus_id"]
        score = hit["score"]
        # Aggiungiamo i token del documento tante volte quante 1..5 in base allo score
        # Usando ceil(score * 5) otteniamo pesi 1-5 invece di 1-10 → meno amplificazione
        weight = max(1, round(score * 5))
        expansion_pool.extend(corpus_raw_tokens[cid] * weight)

    # 3. Frequenza pesata → top-N termini grezzi
    counter = Counter(expansion_pool)
    expansion_terms_raw = [w for w, _ in counter.most_common(TOP_K_TERMS * 2)]

    # FIX 5: rimuovi dall'expansion termini già inclusi nelle keywords
    #         (come sottostringa case-insensitive) per massimizzare copertura
    keywords_set = set(keywords.lower().split())
    expansion_filtered = [
        t for t in expansion_terms_raw
        if t not in keywords_set
    ][:TOP_K_TERMS]

    expansion_str = " ".join(expansion_filtered)

    # FIX 1 + FIX 6: struttura della query espansa:
    #   original query (anchor forte BM25) + keywords lemmatizzate + expansion grezza
    # L'ordine mette i termini più importanti (quelli della query originale) per primi,
    # dando loro maggiore peso nell'IDF di BM25.
    expanded = " ".join(filter(None, [original, keywords, expansion_str])).strip()

    expanded_queries.append({
        "index":    q["index"],
        "original": original,
        "keywords": keywords,
        "expanded": expanded,
        "pubkey":   q.get("pubkey", ""),
    })

# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
output_path = f"{DATA_BASE}/expanded_queries_4.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(expanded_queries)} expanded queries → {output_path}")