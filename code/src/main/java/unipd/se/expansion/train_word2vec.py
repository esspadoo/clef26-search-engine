import json
import re
from gensim.models import Word2Vec
from tqdm import tqdm

def load_stopwords(filepath):
    """Carica le stopwords da un file di testo, una per riga."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Legge, pulisce spazi e filtra righe vuote o commenti
            return set(line.strip().lower() for line in f if line.strip() and not line.startswith('#'))
    except FileNotFoundError:
        print(f"Attenzione: File stopwords non trovato in {filepath}. Procedo senza filtro.")
        return set()

def clean_medical_text(text):
    if not text:
        return []
    text = re.sub(r'\[\d+]|\(\d+\)|Google Scholar|Crossref|PubMed', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\-]', ' ', text)
    return text.split()

def run_pipeline(input_file, output_file, stopwords_file):
    # 1. Caricamento stopwords
    stopwords = load_stopwords(stopwords_file)
    print(f"Caricate {len(stopwords)} stopwords.")

    print(f"Caricamento dati da {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    for paper in tqdm(data, desc="Preprocessing"):
        content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        tokens = clean_medical_text(content)
        if tokens:
            sentences.append(tokens)

    print(f"Addestramento Word2Vec su {len(sentences)} documenti...")
    model = Word2Vec(
        sentences,
        vector_size=200,
        window=10,
        min_count=3,
        workers=4,
        epochs=10
    )

    print("Estrazione sinonimi con filtro stopwords...")
    expansion_dict = {}
    vocab = model.wv.index_to_key

    for word in tqdm(vocab, desc="Generating Dictionary"):
        # Filtro 1: Non espandiamo una parola se è una stopword
        if word in stopwords:
            continue

        similar = model.wv.most_similar(word, topn=10) # Ne prendiamo di più per poi filtrare

        # Filtro 2: Teniamo solo i termini simili che NON sono stopwords
        filtered_similar = [
            {"term": s[0], "score": round(s[1], 3)}
            for s in similar
            if s[0] not in stopwords
        ][:5] # Limitiamo ai top 5 dopo il filtraggio

        if filtered_similar:
            expansion_dict[word] = filtered_similar

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expansion_dict, f, indent=2)
    print(f"Dizionario salvato. Parole espanse: {len(expansion_dict)}")

if __name__ == "__main__":
    # Aggiorna i percorsi secondo la tua struttura
    DATA_PATH = '../../../../../../data/collection_data.json'
    STOPLIST_PATH = '../../../../../../data/stoplist_en_TEX.txt'
    OUTPUT_FILE = 'word2vec_expansion_no_stopwords.json'

    run_pipeline(DATA_PATH, OUTPUT_FILE, STOPLIST_PATH)