"""
QueryExpansor — BGE-large-en-v1.5 (BAAI/bge-large-en-v1.5)
Sincronizzato con MyEnglishAnalyzer_V2 (KStem via Snowball, stoplist custom, ASCII fold).

Pipeline replicata:
  StandardTokenizer → TrimFilter → PatternReplace(#/@) → ASCIIFoldingFilter
  → LowerCaseFilter → EnglishPossessiveFilter → StopFilter(custom) → KStemFilter

Output: data/expanded_queries_bge_large.json
Aggiorna Main.java: queriesPath = "code/data/expanded_queries_bge_large.json"
"""

import json
import re
import unicodedata
import torch
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from nltk.stem import SnowballStemmer

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_BATCH_SIZE = 256
QUERY_BATCH_SIZE  = 32
TOP_K_DOCS        = 5
TOP_K_TERMS       = 15
SPACY_BATCH_SIZE  = 128
DATA_BASE         = "../../../../../../data"
STOPLIST_PATH     = "../../../../../../../code/data/stoplist_en_ranksnl_large.txt"

MODEL_NAME       = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")

# ──────────────────────────────────────────────────────────────
# Stemmer — Snowball English ≈ KStem per l'inglese
# ──────────────────────────────────────────────────────────────
_stemmer = SnowballStemmer("english")

# ──────────────────────────────────────────────────────────────
# ASCII folding — replica ASCIIFoldingFilter di Lucene
# Applicata sia al testo che alle stopword per garantire coerenza.
# ──────────────────────────────────────────────────────────────
def normalize_ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# ──────────────────────────────────────────────────────────────
# Stopword — stoplist custom (ranksnl_large) + default Lucene
# Le stopword vengono normalizzate ASCII al caricamento per
# garantire il match dopo ASCIIFoldingFilter.
# ──────────────────────────────────────────────────────────────
_LUCENE_DEFAULT_STOPS: set[str] = {
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

def _load_stopwords(path: str) -> set[str]:
    """Carica la stoplist custom e la normalizza in ASCII, poi unisce
    alle stopword default di Lucene per coprire entrambi i set."""
    try:
        with open(path, encoding="utf-8") as f:
            custom = {normalize_ascii(line.strip().lower()) for line in f if line.strip()}
        print(f"Stoplist caricata: {len(custom)} voci da '{path}'")
    except FileNotFoundError:
        print(f"[WARN] Stoplist non trovata in '{path}', uso solo default Lucene.")
        custom = set()
    combined = custom | _LUCENE_DEFAULT_STOPS
    print(f"Stopwords totali (custom + Lucene default): {len(combined)}")
    return combined

STOPWORDS = _load_stopwords(STOPLIST_PATH)

# ──────────────────────────────────────────────────────────────
# Text cleaning — replica PatternReplaceFilter (@menzioni, #hashtag)
# Usata per pulire la query originale prima di passarla a Lucene.
# ──────────────────────────────────────────────────────────────
def clean_query(text: str) -> str:
    """Rimuove @menzioni e '#' dagli hashtag, normalizza spazi."""
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\B#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ──────────────────────────────────────────────────────────────
# Tokenizer — replica esatta della pipeline MyEnglishAnalyzer_V2:
#
#   1. Trim + lowercase
#   2. Rimozione @menzioni e #hashtag  (PatternReplaceFilter)
#   3. ASCIIFoldingFilter              (normalize_ascii)
#   4. EnglishPossessiveFilter         (rimozione "'s")
#   5. StopFilter                      (stoplist custom + Lucene default)
#   6. Lunghezza minima 3 char         (coerente con i filtri Lucene)
#   7. KStemFilter                     (approssimato con Snowball English)
#
# I token prodotti da questa funzione devono corrispondere 1:1
# ai token presenti nell'indice Lucene costruito con MyEnglishAnalyzer_V2.
# ──────────────────────────────────────────────────────────────
def raw_tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    if stopwords is None:
        stopwords = STOPWORDS

    # 1. Trim + lowercase
    text = text.strip().lower()

    # 2. Rimozione @menzioni e #hashtag
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\B#(\w+)", r"\1", text)

    # 3. ASCIIFoldingFilter
    text = normalize_ascii(text)

    # Tokenizzazione (StandardTokenizer: alfanumerici, gestisce apostrofi e trattini)
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)

    result: list[str] = []
    for t in tokens:
        # 4. EnglishPossessiveFilter — rimuove "'s" finale
        if t.endswith("'s"):
            t = t[:-2]

        # 5. StopFilter
        if t in stopwords:
            continue

        # 6. Lunghezza minima
        if len(t) < 3:
            continue

        # 7. KStemFilter (via Snowball English — stessa famiglia di algoritmi)
        result.append(_stemmer.stem(t))

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
# Pre-tokenize corpus — applica la stessa pipeline dell'analyzer
# I token qui prodotti devono matchare esattamente quelli nell'indice.
# ──────────────────────────────────────────────────────────────
print("Pre-tokenizing corpus (stemmed, analyzer-aligned)...")
corpus_raw_tokens: list[list[str]] = [
    raw_tokenize(t) for t in tqdm(corpus_texts, desc="Tokenizing corpus")
]

# ──────────────────────────────────────────────────────────────
# Encode corpus — BGE non usa prefisso sul corpus
# ──────────────────────────────────────────────────────────────
print("Encoding corpus (GPU, no prefix)...")
corpus_embeddings = model.encode(
    corpus_texts,
    batch_size=CORPUS_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
    normalize_embeddings=True,
)

# ──────────────────────────────────────────────────────────────
# Encode queries — BGE richiede il prefisso sulle query
# ──────────────────────────────────────────────────────────────
print(f"Encoding queries (GPU, with prefix: '{BGE_QUERY_PREFIX[:40]}...')...")
query_texts_prefixed = [BGE_QUERY_PREFIX + t for t in query_texts]
query_embeddings = model.encode(
    query_texts_prefixed,
    batch_size=QUERY_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
    normalize_embeddings=True,
)

# ──────────────────────────────────────────────────────────────
# Semantic search (GPU, unica chiamata matriciale)
# normalize_embeddings=True → dot_score == cosine similarity
# ──────────────────────────────────────────────────────────────
print("Semantic search (GPU)...")
all_hits = util.semantic_search(
    query_embeddings,
    corpus_embeddings,
    top_k=TOP_K_DOCS,
    score_function=util.dot_score,
)

# ──────────────────────────────────────────────────────────────
# Keyword extraction via spaCy (NOUN + PROPN, lemmatizzati)
# I lemmi di spaCy sono grezzi — verranno stemmati da Lucene
# al momento del parsing della query, come tutti gli altri termini.
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
# Assemble expanded queries
#
# Logica di pesatura:
#   weight = max(1, round(score * 5))
#   → un documento con similarity 0.9 contribuisce 4-5x più di uno a 0.5
#   → favorisce i documenti semanticamente più vicini alla query
# ──────────────────────────────────────────────────────────────
print("Assembling expanded queries...")
expanded_queries: list[dict] = []

for i, q in enumerate(tqdm(queries, desc="Expanding")):
    original  = query_texts[i]
    query_doc = query_docs[i]

    keywords = extract_query_keywords(query_doc)

    # Costruisce il pool di espansione dai top-K documenti semanticamente simili.
    # I token sono già stemmati da raw_tokenize → allineati all'indice Lucene.
    expansion_pool: list[str] = []
    for hit in all_hits[i]:
        cid    = hit["corpus_id"]
        score  = hit["score"]
        weight = max(1, round(score * 5))
        expansion_pool.extend(corpus_raw_tokens[cid] * weight)

    counter = Counter(expansion_pool)
    expansion_terms_raw = [w for w, _ in counter.most_common(TOP_K_TERMS * 2)]

    # Deduplica rispetto ai keyword già presenti
    keywords_set = set(keywords.lower().split())
    expansion_filtered = [
        t for t in expansion_terms_raw
        if t not in keywords_set
    ][:TOP_K_TERMS]

    expansion_str  = " ".join(expansion_filtered)
    original_clean = clean_query(original)
    expanded = " ".join(filter(None, [original_clean, keywords, expansion_str])).strip()

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