import json
import re
from gensim.models import Word2Vec
from tqdm import tqdm

def clean_medical_text(text):
    if not text:
        return []
    # Rimuove riferimenti bibliografici tipo [1, 2], (513), o Google Scholar
    text = re.sub(r'\[\d+]|\(\d+\)|Google Scholar|Crossref|PubMed', ' ', text)
    # Rimuove URL
    text = re.sub(r'http\S+', '', text)
    # Minuscolo e tiene solo caratteri alfanumerici (importante per sigle come SARS-CoV-2)
    text = text.lower()
    # Sostituisce la punteggiatura con spazi, ma mantiene i trattini interni alle parole
    text = re.sub(r'[^a-z0-9\-]', ' ', text)
    return text.split()

def run_pipeline(input_file, output_file):
    print(f"Caricamento dati da {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    for paper in tqdm(data, desc="Preprocessing"):
        # Uniamo titolo e abstract per il contesto
        content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        tokens = clean_medical_text(content)
        if tokens:
            sentences.append(tokens)

    print(f"Addestramento Word2Vec su {len(sentences)} documenti...")
    # Parametri ottimizzati per testi scientifici:
    # window=10: i paper hanno frasi lunghe e complesse
    # min_count=3: teniamo anche termini meno comuni se sono tecnici
    model = Word2Vec(
        sentences,
        vector_size=200,
        window=10,
        min_count=3,
        workers=4,
        epochs=10
    )

    print("Estrazione sinonimi e salvataggio...")
    expansion_dict = {}
    vocab = model.wv.index_to_key

    for word in tqdm(vocab, desc="Generating Dictionary"):
        # Salviamo i 5 termini più simili con la loro probabilità (score)
        # Lo score può servire in Java per pesare il boost
        similar = model.wv.most_similar(word, topn=5)
        expansion_dict[word] = [{"term": s[0], "score": round(s[1], 3)} for s in similar]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expansion_dict, f, indent=2)

if __name__ == "__main__":
    # Assicurati che il nome del file corrisponda al tuo JSON
    run_pipeline('../../../../../../data/collection_data.json', 'word2vec_expansion.json')