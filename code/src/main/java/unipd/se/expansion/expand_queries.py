import json
import re
from tqdm import tqdm

def load_stopwords(filepath):
    """Carica le stopwords dal file .txt fornito."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip() and not line.startswith('#'))
    except FileNotFoundError:
        print(f"Errore: File stoplist non trovato in {filepath}")
        return set()

def clean_query_text(text):
    if not text:
        return []
    # Pulizia coerente con il training: minuscolo e caratteri alfanumerici/trattini
    text = text.lower()
    text = re.sub(r'[^a-z0-9\-]', ' ', text)
    return text.split()

def expand_queries(queries_file, expansion_model_file, output_file, stopwords_file):
    # 1. Caricamento risorse
    stopwords = load_stopwords(stopwords_file)
    print(f"Caricate {len(stopwords)} stopwords.")

    print(f"Caricamento modello di espansione...")
    with open(expansion_model_file, 'r', encoding='utf-8') as f:
        expansion_lookup = json.load(f)

    print(f"Caricamento query da {queries_file}...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    expanded_queries = []

    # 2. Processo di espansione
    for query in tqdm(queries, desc="Expanding Queries"):
        original_text = query.get('text', '')
        tokens = clean_query_text(original_text)

        expansion_terms = set()

        for token in tokens:
            # Filtro: non cerchiamo espansioni per le stopwords presenti nella query
            if token in stopwords:
                continue

            if token in expansion_lookup:
                similars = expansion_lookup[token]
                for item in similars:
                    # Filtro di sicurezza: non aggiungiamo stopwords come termini di espansione
                    if item['term'] not in stopwords:
                        expansion_terms.add(item['term'])

        # Aggiungiamo il campo expansion
        query['expansion'] = " ".join(list(expansion_terms))
        expanded_queries.append(query)

    print(f"Salvataggio query espanse in {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_queries, f, indent=2)

if __name__ == "__main__":
    # Percorsi file
    QUERIES_IN = '../../../../../../data/en_train.json'
    MODEL_IN = 'word2vec_expansion_no_stopwords.json'
    QUERIES_OUT = '../../../../../../data/expanded_queries_TEX_no_stopwords.json'
    STOPLIST_PATH = '../../../../../../data/stoplist_en_TEX.txt'

    expand_queries(QUERIES_IN, MODEL_IN, QUERIES_OUT, STOPLIST_PATH)