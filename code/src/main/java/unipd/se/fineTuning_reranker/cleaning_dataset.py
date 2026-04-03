import re
import pandas as pd

# =========================
# CLEAN FUNCTION per pulire query da immondizia
# =========================
def clean_tweet(text: str) -> str:
    # Rimuovi @mention (mantieni il testo dopo)
    text = re.sub(r'@\w+\s?', '', text)
    # Rimuovi URL
    text = re.sub(r'http\S+', '', text)
    # Rimuovi hashtag symbol ma mantieni il testo
    text = re.sub(r'#(\w+)', r'\1', text)
    # Rimuovi numerazioni di thread (1/, 2/, ecc.)
    text = re.sub(r'^\d+\/\s*', '', text, flags=re.MULTILINE)
    # Rimuovi emoji e caratteri speciali non ASCII
    text = text.encode('ascii', 'ignore').decode()
    # Collassa spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# LOAD DATA
# =========================
queries = pd.read_json("en_train.json")
papers = pd.read_json("collection_data.json")

# =========================
# CLEAN QUERIES
# =========================
queries["clean_text"] = queries["text"].apply(clean_tweet)

# rimuovi query vuote dopo cleaning
queries = queries[queries["clean_text"].str.len() > 0]

# =========================
# BUILD PAPER TEXT
# =========================
papers["full_text"] = papers["title"].fillna('') + ". " + papers["abstract"].fillna('')

# mapping pubkey -> documento
paper_dict = dict(zip(papers["pubkey"], papers["full_text"]))

# =========================
# BUILD POSITIVES (SAFE)
# =========================
dataset = []

missing_pubkeys = 0

for _, row in queries.iterrows():
    pubkey = row["pubkey"]

    # usa solo positivi sicuri
    if pubkey not in paper_dict:
        missing_pubkeys += 1
        continue

    query = row["clean_text"]
    positive_doc = paper_dict[pubkey]

    # filtro sicurezza: evita testi troppo corti
    if len(query) < 10 or len(positive_doc) < 50:
        continue

    dataset.append({
        "query": query,
        "positive": positive_doc,
        "pubkey": pubkey
    })

# =========================
# SAVE
# =========================
df = pd.DataFrame(dataset)
df.to_json("positives.json", orient="records", indent=2)

print("=== STATS ===")
print(f"Query iniziali: {len(queries)}")
print(f"Positivi validi: {len(df)}")
print(f"Pubkey mancanti: {missing_pubkeys}")