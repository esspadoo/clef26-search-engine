import json
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# -----------------------------
# 1. Normalization function
# -----------------------------
def normalize_query(text: str) -> str:
    """
    Normalize social-post style queries.
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"@\w+", " ", text)            # remove mentions
    text = re.sub(r"#", "", text)                # remove hashtags
    text = re.sub(r"[^a-z0-9\s]", " ", text)    # remove punctuation
    text = re.sub(r"\s+", " ", text)            # collapse spaces
    return text.strip()

# -----------------------------
# 2. Load papers
# -----------------------------
with open("data/collection_data.json", "r", encoding="utf-8") as f:
    papers = json.load(f)  # [{"pubkey": "...", "title": "...", "abstract": "..."}]

# Prepare corpus texts
corpus_texts = [p["title"] + " " + p["abstract"] for p in papers]

# -----------------------------
# 3. Load queries
# -----------------------------
with open("data/en_dev.json", "r", encoding="utf-8") as f:
    queries = json.load(f)  # [{"index": "...", "text": "...", "pubkey": "..."}]

# -----------------------------
# 4. Load SBERT model
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight multilingual model

# Precompute embeddings for corpus
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)

# -----------------------------
# 5. Expand queries
# -----------------------------
expanded_queries = []

for q in tqdm(queries, desc="Expanding queries"):
    original_text = q["text"]
    normalized = normalize_query(original_text)
    query_emb = model.encode(normalized, convert_to_tensor=True)

    # Semantic search: get top 5 similar papers
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]

    # Collect expansion terms from top hits
    expansion_terms = []
    for hit in hits:
        idx = hit['corpus_id']
        text = corpus_texts[idx]
        expansion_terms += text.split()

    # Keep top unique terms (simple)
    expansion_terms = list(dict.fromkeys(expansion_terms))[:15]

    expanded_text = normalized + " " + " ".join(expansion_terms)

    expanded_queries.append({
        "index": q["index"],
        "original": original_text,
        "normalized": normalized,
        "expanded": expanded_text,
        "pubkey": q.get("pubkey", "")
    })

# -----------------------------
# 6. Save expanded queries
# -----------------------------
with open("data/expanded_queries.json", "w", encoding="utf-8") as f:
    json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

print("Expanded queries saved to data/expanded_queries.json")