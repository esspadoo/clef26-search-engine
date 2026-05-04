#!/usr/bin/env python3
"""
translate_queries.py
Traduce query multilingue (FR/DE) in inglese usando EuroLLM-9B-Instruct-2512.
Usa HuggingFace pipeline per applicare correttamente il chat template,
più un layer di pulizia per rimuovere artefatti nel testo generato.

Licenza modello: Apache 2.0 — usabile senza restrizioni per CLEF.

Uso:
  python translate_queries.py --input fr_train.json --lang fr --output fr_train_en.json
  python translate_queries.py --input de_train.json --lang de --output de_train_en.json

Opzioni:
  --input          Path al JSON delle query sorgente
  --lang           Lingua sorgente: 'fr' oppure 'de'
  --output         Path del JSON di output (query tradotte in EN)
  --model          Modello HuggingFace (default: utter-project/EuroLLM-9B-Instruct-2512)
  --batch_size     Query per batch (default: 8, sicuro per 9B su 24-48GB VRAM)
  --max_new_tokens Token massimi per la traduzione (default: 256)
  --cache_dir      Directory cache HuggingFace
  --keep_original  Aggiunge 'text_original' con testo sorgente per verifica
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "utter-project/EuroLLM-9B-Instruct-2512"

LANG_NAMES = {
    "fr": "French",
    "de": "German",
}

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the user's text to English. "
    "Output ONLY the English translation. "
    "Do not include the original text, explanations, labels, or any other content."
)

# Pattern di pulizia da applicare in ordine sull'output grezzo del modello.
# Coprono tutti gli artefatti osservati nell'output di EuroLLM:
#   - "English: \n assistant\n..."
#   - "assistant\n..."
#   - "English:\n..."
#   - testo sorgente rimasto prima della traduzione
CLEANUP_PATTERNS = [
    # Rimuovi tutto fino a "assistant" (incluso) se presente
    (r"(?s)^.*?\bassistant\b\s*", ""),
    # Rimuovi prefissi tipo "English:", "EN:", "Translation:", "Traduction:" ecc.
    (r"(?i)^(english|en|translation|traduction|übersetzung)\s*:\s*", ""),
    # Rimuovi il prompt che il modello ha ripetuto verbatim
    (r"(?s)^.*?nothing else\.\s*", ""),
    # Rimuovi righe che contengono ancora testo nella lingua sorgente
    # (heuristic: righe con caratteri tedeschi/francesi tipici dopo la traduzione)
    # — non applicato automaticamente, troppo aggressivo; gestito dal layer sopra
]


def clean_translation(raw: str) -> str:
    """
    Pulisce l'output grezzo del modello rimuovendo artefatti del prompt
    e testo residuo nella lingua sorgente.
    """
    text = raw.strip()
    for pattern, replacement in CLEANUP_PATTERNS:
        text = re.sub(pattern, replacement, text)
        text = text.strip()
    # Rimuovi eventuali righe vuote iniziali
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text.strip()


def build_messages(text: str, src_lang: str) -> list[dict]:
    """Costruisce la lista di messaggi nel formato ChatML per EuroLLM."""
    lang_name = LANG_NAMES[src_lang]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Translate this {lang_name} text to English. "
                f"Output only the English translation, nothing else.\n\n"
                f"{text}"
            ),
        },
    ]


def load_pipeline(model_name: str, cache_dir: str | None, device: str):
    log.info(f"Caricamento pipeline: {model_name}")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "cache_dir": cache_dir,
        },
        device_map="auto",
    )
    # Necessario per batch generation con decoder-only
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {torch.cuda.get_device_name(0)} ({mem:.0f}GB VRAM)")
    log.info("Pipeline caricata | bfloat16 | device_map=auto")
    return pipe


def translate_all(
    queries: list[dict],
    src_lang: str,
    pipe,
    batch_size: int,
    max_new_tokens: int,
    keep_original: bool,
) -> list[dict]:
    total = len(queries)
    log.info(
        f"Inizio traduzione: {total} query | batch_size={batch_size} | "
        f"{LANG_NAMES[src_lang]} -> English"
    )

    # Prepara tutti i messaggi
    all_messages = [build_messages(q["text"], src_lang) for q in queries]

    translated_texts = []
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch_messages = all_messages[i : i + batch_size]

        outputs = pipe(
            batch_messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=pipe.tokenizer.pad_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
            batch_size=batch_size,
        )

        for out in outputs:
            # pipeline restituisce lista di dict; l'ultimo messaggio è la risposta
            raw = out[0]["generated_text"]
            # generated_text include l'intero storico messaggi come lista o stringa
            if isinstance(raw, list):
                # Formato lista: l'ultimo elemento è il messaggio dell'assistant
                assistant_content = raw[-1]["content"]
            else:
                # Formato stringa: estrai solo la parte dopo l'ultimo "assistant"
                assistant_content = raw

            cleaned = clean_translation(assistant_content)
            translated_texts.append(cleaned)

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        speed = done / elapsed
        eta = (total - done) / speed if speed > 0 else 0
        log.info(f"  {done}/{total} | {speed:.1f} q/s | ETA {eta:.0f}s")

    elapsed_total = time.time() - t0
    log.info(
        f"Completato: {total} query in {elapsed_total:.1f}s "
        f"({total / elapsed_total:.1f} q/s)"
    )

    output = []
    for q, translated in zip(queries, translated_texts):
        entry = {"index": q["index"]}
        if keep_original:
            entry["text_original"] = q["text"]
        entry["text"] = translated
        entry["pubkey"] = q["pubkey"]
        output.append(entry)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Traduce query FR/DE -> EN con EuroLLM-9B-Instruct (Apache 2.0)"
    )
    parser.add_argument("--input",    required=True,  help="Path JSON query sorgente")
    parser.add_argument("--lang",     required=True,  choices=["fr", "de"], help="Lingua sorgente")
    parser.add_argument("--output",   required=True,  help="Path JSON output (EN)")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Query per batch (default: 8 — sicuro per 9B bfloat16 su 24GB VRAM)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Token massimi generati per traduzione (default: 256)",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--keep_original", action="store_true",
        help="Aggiunge 'text_original' per verifica qualità",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        log.warning("CUDA non disponibile — inferenza su CPU sara' molto lenta")

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"File non trovato: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        queries = json.load(f)
    log.info(f"Caricate {len(queries)} query da {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.model, args.cache_dir, device)

    translated = translate_all(
        queries, args.lang, pipe,
        args.batch_size, args.max_new_tokens, args.keep_original,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    log.info(f"Output salvato: {output_path}")

    # Verifica qualitativa su un campione
    log.info("--- Campione di traduzioni (prime 5) ---")
    for i in range(min(5, len(queries))):
        log.info(f"[idx={queries[i]['index']}]")
        if args.keep_original:
            log.info(f"  SRC: {queries[i]['text'][:120]}")
        log.info(f"  TGT: {translated[i]['text'][:120]}")


if __name__ == "__main__":
    main()
