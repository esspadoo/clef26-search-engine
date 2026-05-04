import subprocess
import re
import random
import csv
import os

# --- CONFIGURAZIONE PERCORSI ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
TARGET_CLASSES = os.path.join(PROJECT_ROOT, "target", "classes")
LIB_DIR = os.path.join(PROJECT_ROOT, "lib", "*")
OUTPUT_LOG = os.path.join(os.path.dirname(__file__), "tuning_results.csv")

PYTHON_EXPAND_SCRIPT = os.path.join(os.path.dirname(__file__), "sparse_query_expander.py")

JAVA_MAIN_CLASS = "unipd.se.Main"
NUM_TENTATIVI = 200
SEP = ";" if os.name == "nt" else ":"

def run_trial(threshold, max_terms, w1, w2, w3):
    print(f"\n[TEST] Thr: {threshold} | Max: {max_terms} | Weights: [{w1}, {w2}, {w3}]")

    try:
        subprocess.run(["python", PYTHON_EXPAND_SCRIPT, str(threshold), str(max_terms)], check=True)

        classpath = f"{TARGET_CLASSES}{SEP}{LIB_DIR}"

        cmd = (
            f'java -cp "{classpath}" {JAVA_MAIN_CLASS} '
            f'_ _ _ {w1} {w2} {w3}'
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            shell=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            print(f"  [!] Errore Java (Exit {result.returncode})")
            print(f"  [STDERR]: {result.stderr.strip()}")
            return 0.0

        match = re.search(r"Recall@100:\s+([0-9]+[.,][0-9]+)", result.stdout)

        if match:
            score = float(match.group(1).replace(',', '.'))
            print(f"  -> Recall@100: {score}")
            return score
        else:
            print("  [!] Recall@100 non trovato. Verificare l'output Java.")
            return 0.0

    except Exception as e:
        print(f"  [!] Errore: {e}")
        return 0.0

def main():
    best_score = -1
    best_params = {}
    seen_configs = set() # Per tenere traccia dei trial unici

    os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)

    file_exists = os.path.isfile(OUTPUT_LOG)

    # Carica trial passati se il file esiste per evitare duplicati tra diverse esecuzioni
    if file_exists:
        with open(OUTPUT_LOG, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Salta header
            for row in reader:
                if len(row) >= 5:
                    # Salva come tupla (thr, max, w1, w2, w3) convertiti correttamente
                    seen_configs.add((float(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])))

    with open(OUTPUT_LOG, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["threshold", "max_terms", "w1", "w2", "w3", "recall100"])

    for i in range(NUM_TENTATIVI):
        print(f"\n--- Trial {i+1}/{NUM_TENTATIVI} ---")

        # Loop per generare parametri unici
        while True:
            t = round(random.uniform(0.6, 1), 2)
            m = random.randint(1, 10)
            w1 = round(random.uniform(0, 20), 2)
            w2 = round(random.uniform(0, 20), 2)
            w3 = round(random.uniform(0, 20), 2)

            config = (t, m, w1, w2, w3)
            if config not in seen_configs:
                seen_configs.add(config)
                break
            else:
                print("  [INFO] Configurazione già testata, rigenerazione...")

        score = run_trial(t, m, w1, w2, w3)

        with open(OUTPUT_LOG, "a", newline='') as f:
            csv.writer(f).writerow([t, m, w1, w2, w3, score])

        if score > best_score:
            best_score = score
            best_params = {"thr": t, "max": m, "w1": w1, "w2": w2, "w3": w3}
            print(f"NUOVO RECORD: {best_score}")

    print("\n" + "="*40)
    print(f"MIGLIOR RISULTATO: {best_score}")
    print(f"PARAMETRI: {best_params}")
    print("="*40)

if __name__ == "__main__":
    main()