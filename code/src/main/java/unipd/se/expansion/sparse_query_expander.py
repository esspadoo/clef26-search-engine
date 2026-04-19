import json
import re
import sys
import nltk
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Risorse NLTK minime
nltk.download('punkt', quiet=True)

class SimpleQueryAnalyzer:
    def __init__(self, stopwords_file):
        # Pattern di pulizia (URL, @mentions, # symbol)
        self.url_pattern = re.compile(r"https?://\S+\s?")
        self.mention_pattern = re.compile(r"@\w+\s?")
        self.hashtag_symbol = re.compile(r"#")

        # Caricamento stopword personalizzate
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stop_words = set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            self.stop_words = set(stopwords.words('english'))

        # Tokenizer: alfanumerici e trattini
        self.tokenizer = RegexpTokenizer(r"[a-z0-9\-]+")

    def analyze(self, text):
        if not text:
            return []

        # 1. Pulizia stringa
        text = self.url_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)
        text = self.hashtag_symbol.sub("", text)

        # 2. Lowercase e Tokenizzazione
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)

        # 3. Solo rimozione Stopwords (NIENTE stemming)
        return [t for t in tokens if t not in self.stop_words]

def run_expansion_pipeline(input_path, model_path, output_path, stop_path, threshold, max_terms):
    analyzer = SimpleQueryAnalyzer(stop_path)

    print("Caricamento risorse...")
    with open(model_path, 'r', encoding='utf-8') as f:
        expansion_lookup = json.load(f)

    with open(input_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    final_results = []

    for q in tqdm(queries, desc="Processing"):
        original_text = q.get('original', '')

        # Token originali puliti (no stopword)
        original_tokens = analyzer.analyze(original_text)

        # Set per unire originali + espansioni
        combined_terms = set(original_tokens)

        # Aggiunta termini dal Word2Vec
        for token in original_tokens:
            if token in expansion_lookup:
                similars = [item['term'] for item in expansion_lookup[token][:max_terms]
                            if item['score'] >= threshold]
                combined_terms.update(similars)

        # Costruzione JSON
        output_obj = {
            "index": q.get("index"),
            "original": original_text,
            "expanded": q.get("expanded", ""),
            "sparse": " ".join(list(combined_terms)),
            "pubkey": q.get("pubkey")
        }

        final_results.append(output_obj)

    print(f"Salvataggio in {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    # Parametri da riga di comando (Threshold e Max Terms)
    THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    MAX_TERMS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    run_expansion_pipeline(
        input_path='../../../../../../data/expanded_queries_bge_large.json',
        model_path='word2vec_expansion.json',
        output_path='../../../../../../data/expanded_queries_en.json',
        stop_path='../../../../../../data/stoplist_en_TEX.txt',
        threshold=THRESHOLD,
        max_terms=MAX_TERMS
    )