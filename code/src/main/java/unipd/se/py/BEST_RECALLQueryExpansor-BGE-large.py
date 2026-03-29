"""
QueryExpansor — BGE-large-en-v1.5 (BAAI/bge-large-en-v1.5)
Uno dei migliori modelli generici per retrieval (top sul benchmark BEIR).
Richiede il prefisso "Represent this sentence for searching relevant passages: "
SOLO sulle query — il corpus non ha prefisso.
 
Rispetto a MiniLM: embedding 1024-dim (vs 384), addestrato con hard-negative mining
su dataset di retrieval → migliore separazione tra documenti rilevanti e non.

Output: data/expanded_queries_bge_large.json
Aggiorna Main.java: queriesPath = "code/data/expanded_queries_bge_large.json"
"""

import json
import re
import torch
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
# BGE-large ha embedding 1024-dim → più VRAM per batch.
# RTX 3090 (24GB) regge 256 in fp16, abbassa a 128 se OOM.
CORPUS_BATCH_SIZE = 256
QUERY_BATCH_SIZE  = 32
TOP_K_DOCS        = 5
TOP_K_TERMS       = 15
SPACY_BATCH_SIZE  = 128
DATA_BASE         = "../../../../../../data"

MODEL_NAME = "BAAI/bge-large-en-v1.5"

# BGE richiede questo prefisso SOLO sulle query (non sul corpus).
# Senza di esso le performance calano sensibilmente (indicato nella doc ufficiale).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")

# ──────────────────────────────────────────────────────────────
# Lucene stop words — identico a BESTQueryExpansorCUDA1.py
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

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:['\-][a-zA-Z0-9]+)*")

def raw_tokenize(text: str) -> list[str]:
    """
    Tokenizza in forma grezza (non stemmata) — uguale a BESTQueryExpansorCUDA1.
    Lucene applicherà MyEnglishAnalyzer (Snowball) durante il parsing.
    """
    tokens = _TOKEN_RE.findall(text)
    result: list[str] = []
    for t in tokens:
        t = t.lstrip("#").lower()
        if t.endswith("'s"):
            t = t[:-2]
        if t in LUCENE_STOPS:
            continue
        if len(t) < 3:
            continue
        result.append(t)
    return result

# ──────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────
print("Loading spaCy (tagger only)...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

print(f"Loading {MODEL_NAME} on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

if DEVICE == "cuda":
    try:
        model.half()
        print("fp16 enabled")
        print(f"VRAM stimata corpus encode: ~{(1024 * 4 * CORPUS_BATCH_SIZE) / 1024**2:.0f} MB/batch")
    except Exception as e:
        print(f"fp16 non disponibile ({e}), uso fp32")

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
query_texts: list[str] = [q.get("text", "") for q in queries]

# ──────────────────────────────────────────────────────────────
# Pre-tokenize corpus (grezza, identico a CUDA1)
# ──────────────────────────────────────────────────────────────
print("Pre-tokenizing corpus (raw, no stemming)...")
corpus_raw_tokens: list[list[str]] = [
    raw_tokenize(t) for t in tqdm(corpus_texts, desc="Tokenizing corpus")
]

# ──────────────────────────────────────────────────────────────
# Encode corpus — BGE NON usa prefisso sul corpus, solo sulle query
# ──────────────────────────────────────────────────────────────
print("Encoding corpus (GPU, no prefix)...")
corpus_embeddings = model.encode(
    corpus_texts,
    batch_size=CORPUS_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
    normalize_embeddings=True,   # necessario per BGE con cosine similarity
)

# ──────────────────────────────────────────────────────────────
# Encode queries — BGE RICHIEDE il prefisso sulle query
# Senza prefisso le metriche calano (documentato da BAAI).
# ──────────────────────────────────────────────────────────────
print(f"Encoding queries (GPU, with prefix: '{BGE_QUERY_PREFIX[:40]}...')...")
query_texts_prefixed = [BGE_QUERY_PREFIX + t for t in query_texts]
query_embeddings = model.encode(
    query_texts_prefixed,
    batch_size=QUERY_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
    normalize_embeddings=True,   # necessario per BGE
)

# ──────────────────────────────────────────────────────────────
# Semantic search (GPU, unica chiamata matriciale)
# Con normalize_embeddings=True, dot product == cosine similarity
# ──────────────────────────────────────────────────────────────
print("Semantic search (GPU)...")
all_hits = util.semantic_search(
    query_embeddings,
    corpus_embeddings,
    top_k=TOP_K_DOCS,
    score_function=util.dot_score,  # equivalente a cosine con embedding normalizzati
)

# ──────────────────────────────────────────────────────────────
# Keyword extraction via spaCy (batch) — identico a CUDA1
# ──────────────────────────────────────────────────────────────
print("Extracting keywords from queries (spaCy batch)...")
query_docs = list(nlp.pipe(query_texts, batch_size=SPACY_BATCH_SIZE, n_process=1))

def extract_query_keywords(doc: spacy.tokens.Doc) -> str:
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 2
    ]
    return " ".join(dict.fromkeys(keywords))

# ──────────────────────────────────────────────────────────────
# Assemble expanded queries (logica identica a BESTQueryExpansorCUDA1)
# ──────────────────────────────────────────────────────────────
print("Assembling expanded queries...")
expanded_queries: list[dict] = []

for i, q in enumerate(tqdm(queries, desc="Expanding")):
    original  = query_texts[i]
    query_doc = query_docs[i]

    keywords = extract_query_keywords(query_doc)

    expansion_pool: list[str] = []
    for hit in all_hits[i]:
        cid    = hit["corpus_id"]
        score  = hit["score"]
        weight = max(1, round(score * 5))
        expansion_pool.extend(corpus_raw_tokens[cid] * weight)

    counter = Counter(expansion_pool)
    expansion_terms_raw = [w for w, _ in counter.most_common(TOP_K_TERMS * 2)]

    keywords_set = set(keywords.lower().split())
    expansion_filtered = [
        t for t in expansion_terms_raw
        if t not in keywords_set
    ][:TOP_K_TERMS]

    expansion_str = " ".join(expansion_filtered)
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
output_path = f"{DATA_BASE}/expanded_queries_bge_large.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(expanded_queries)} expanded queries → {output_path}")
print("\nPer usare questo output in Main.java, imposta:")
print('  String queriesPath = "code/data/expanded_queries_bge_large.json";')