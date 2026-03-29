"""
QueryExpansor — SPECTER2 (allenai/specter2)
Modello specializzato su paper scientifici (citazioni + abstract).
Rispetto a BESTQueryExpansorCUDA1.py (all-MiniLM-L6-v2), SPECTER2
capisce meglio il dominio accademico → top-K docs più rilevanti
→ expansion pool più pulito → potenzialmente migliori metriche IR.

Output: data/expanded_queries_specter2.json
Aggiorna Main.java: queriesPath = "code/data/expanded_queries_specter2.json"


SPECTER2 è addestrato su un task di citation recommendation — data una paper, trova le paper che cita. Questo è diverso dal tuo task, che è query-to-paper retrieval: date query brevi in linguaggio naturale (spesso simili a tweet o frasi informali), trova i paper rilevanti.
Le tue query vengono da en_train.json e hanno probabilmente una distribuzione molto diversa dagli abstract scientifici su cui SPECTER2 è stato addestrato. MiniLM invece è addestrato su coppie di frasi generiche in linguaggio naturale — molto più vicino alla distribuzione delle tue query.
Gli altri fattori che contribuiscono
Pooling strategy: SPECTER2 usa CLS-token pooling, che funziona bene su testi lunghi e strutturati come abstract. Le tue query sono brevi → il CLS token cattura meno informazione utile rispetto al mean pooling di MiniLM.
Spazio embedding: SPECTER2 produce embedding in uno spazio ottimizzato per similarità tra paper, non per rilevanza query→documento. La distanza coseno tra una query breve e un abstract lungo in quello spazio è meno significativa.

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
CORPUS_BATCH_SIZE = 256   # SPECTER2 è più grande di MiniLM → batch più piccolo
QUERY_BATCH_SIZE  = 32    # idem
TOP_K_DOCS        = 5
TOP_K_TERMS       = 15
SPACY_BATCH_SIZE  = 128
DATA_BASE         = "../../../../../../data"

# ──────────────────────────────────────────────────────────────
# SPECTER2: prefisso query richiesto dal modello
# Il corpus NON ha prefisso (solo le query lo usano).
# https://huggingface.co/allenai/specter2
# ──────────────────────────────────────────────────────────────
QUERY_PREFIX = ""   # SPECTER2 non richiede prefisso esplicito (a differenza di E5/BGE)
# ma lo lasciamo configurabile per future varianti

MODEL_NAME   = "allenai/specter2_base"

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

# fp16 solo su CUDA e se il modello lo supporta
if DEVICE == "cuda":
    try:
        model.half()
        print("fp16 enabled")
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
# Encode corpus — SPECTER2 non usa prefisso sul corpus
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
# Encode queries — SPECTER2 non richiede prefisso esplicito
# (a differenza di E5/BGE che usano "query: " e "passage: ")
# ──────────────────────────────────────────────────────────────
print("Encoding queries (GPU)...")
query_texts_input = [QUERY_PREFIX + t for t in query_texts] if QUERY_PREFIX else query_texts
query_embeddings = model.encode(
    query_texts_input,
    batch_size=QUERY_BATCH_SIZE,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=DEVICE,
)

# ──────────────────────────────────────────────────────────────
# Semantic search (GPU, unica chiamata matriciale)
# ──────────────────────────────────────────────────────────────
print("Semantic search (GPU)...")
all_hits = util.semantic_search(
    query_embeddings,
    corpus_embeddings,
    top_k=TOP_K_DOCS,
)

# ──────────────────────────────────────────────────────────────
# Keyword extraction via spaCy (batch)
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
output_path = f"{DATA_BASE}/expanded_queries_specter2.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(expanded_queries)} expanded queries → {output_path}")
print("\nPer usare questo output in Main.java, imposta:")
print('  String queriesPath = "code/data/expanded_queries_specter2.json";')
