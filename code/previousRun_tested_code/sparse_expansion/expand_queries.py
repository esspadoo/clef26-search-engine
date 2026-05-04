import json
import re
import sys
import nltk
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Scarica le risorse se necessario
nltk.download('punkt', quiet=True)

class MyEnglishAnalyzerNLTK:
    def __init__(self, stopwords_file):
        self.url_pattern = re.compile(r"https?://\S+\s?")
        self.mention_pattern = re.compile(r"@\w+\s?")
        self.hashtag_symbol = re.compile(r"#")
        self.stemmer = PorterStemmer()

        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stop_words = set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            self.stop_words = set(stopwords.words('english'))

        self.tokenizer = RegexpTokenizer(r"[a-z0-9\-]+")

    def analyze(self, text):
        if not text:
            return []

        # Pre-processing
        text = self.url_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)
        text = self.hashtag_symbol.sub("", text)

        text = text.lower()
        tokens = self.tokenizer.tokenize(text)

        cleaned_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                stemmed = self.stemmer.stem(token)
                cleaned_tokens.append(stemmed)

        return cleaned_tokens

def run_expansion_pipeline(input_path, model_path, output_path, stop_path, threshold=0.85, max_terms=3):
    analyzer = MyEnglishAnalyzerNLTK(stop_path)

    with open(model_path, 'r', encoding='utf-8') as f:
        expansion_lookup = json.load(f)

    with open(input_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    final_results = []

    for q in tqdm(queries, desc="Analisi ed Espansione"):
        original_text = q.get('original', '')

        # 1. Analisi (Stemming + Stopwords)
        analyzed_tokens = analyzer.analyze(original_text)

        # 2. Reperimento termini di espansione
        # Usiamo un set per contenere TUTTO (Originali + Espansi) senza duplicati
        combined_terms = set(analyzed_tokens)

        for token in analyzed_tokens:
            if token in expansion_lookup:
                similars = [item['term'] for item in expansion_lookup[token][:max_terms]
                            if item['score'] >= threshold]
                combined_terms.update(similars)

        # 3. Costruzione dell'oggetto JSON
        output_obj = {
            "index": q.get("index"),
            "original": original_text,
            "expanded": q.get("expanded", ""),
            "sparse": " ".join(list(combined_terms)), # Include sia originali che espansi
            "pubkey": q.get("pubkey")
        }

        final_results.append(output_obj)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    # Parametri da riga di comando o default basati su analisi SOTA
    THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    MAX_TERMS = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    run_expansion_pipeline(
        input_path='../../data/expanded_queries_bge_large.json',
        model_path='../../src/main/java/unipd/se/expansion/word2vec_expansion.json',
        output_path='../../data/Train_set/expanded_queries_en.json',
        stop_path='../stoplist_en_TEX.txt',
        threshold=THRESHOLD,
        max_terms=MAX_TERMS
    )