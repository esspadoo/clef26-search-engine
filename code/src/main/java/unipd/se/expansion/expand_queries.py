import json
import re
from tqdm import tqdm

def clean_query_text(text):
    if not text:
        return []
    # Applichiamo la stessa pulizia usata per il training Word2Vec
    text = text.lower()
    text = re.sub(r'[^a-z0-9\-]', ' ', text)
    return text.split()

def expand_queries(queries_file, expansion_model_file, output_file):
    print(f"Caricamento modello di espansione...")
    with open(expansion_model_file, 'r', encoding='utf-8') as f:
        expansion_lookup = json.load(f)

    print(f"Caricamento query da {queries_file}...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    expanded_queries = []

    for query in tqdm(queries, desc="Expanding Queries"):
        original_text = query.get('text', '')
        tokens = clean_query_text(original_text)

        # Creiamo un set per evitare termini duplicati nell'espansione
        expansion_terms = set()

        for token in tokens:
            if token in expansion_lookup:
                # Recuperiamo i termini simili (già filtrati a top 5 nello script precedente)
                similars = expansion_lookup[token]
                for item in similars:
                    expansion_terms.add(item['term'])

        # Aggiungiamo il campo expansion come stringa (o lista, a seconda di come ti serve in Java)
        query['expansion'] = " ".join(list(expansion_terms))
        expanded_queries.append(query)

    print(f"Salvataggio query espanse in {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_queries, f, indent=2)

if __name__ == "__main__":
    # Assicurati che i percorsi siano corretti per la tua struttura cartelle
    expand_queries(
        '../../../../../../data/en_train.json',
        'word2vec_expansion.json',
        '../../../../../../data/expanded_queries_TEX.json'
    )