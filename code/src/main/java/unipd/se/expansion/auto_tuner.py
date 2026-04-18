import subprocess
import re
import random
import csv
import os

# --- CONFIGURAZIONE PERCORSI ---
# Poiché lo script è in src/main/java/unipd/se/expansion/, 
# dobbiamo salire di 5 livelli per arrivare alla radice del progetto.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
TARGET_CLASSES = os.path.join(PROJECT_ROOT, "target", "classes")
LIB_DIR = os.path.join(PROJECT_ROOT, "lib", "*")
OUTPUT_LOG = os.path.join(PROJECT_ROOT, "results", "tuning_results.csv")

# Script di espansione (nella stessa cartella di questo script)
PYTHON_EXPAND_SCRIPT = os.path.join(os.path.dirname(__file__), "expand_queries.py")

JAVA_MAIN_CLASS = "unipd.se.Main"
NUM_TENTATIVI = 200
SEP = ";" if os.name == "nt" else ":"

def run_trial(threshold, max_terms, w1, w2, w3):
    print(f"\n[TEST] Thr: {threshold} | Max: {max_terms} | Weights: [{w1}, {w2}, {w3}]")

    try:
        # 1. Fase Python: Espansione Query
        subprocess.run(["python", PYTHON_EXPAND_SCRIPT, str(threshold), str(max_terms)], check=True)

        # 2. Fase Java: Costruzione Classpath
        # Usiamo i percorsi assoluti calcolati sopra per evitare errori di "Class Not Found"
        classpath = f"{TARGET_CLASSES}{SEP}{LIB_DIR}"

        # Comando Java
        cmd = (
            f'java -cp "{classpath}" {JAVA_MAIN_CLASS} '
            f'_ _ _ {w1} {w2} {w3}'
        )

        # Esecuzione dalla ROOT del progetto per coerenza con i percorsi dei dati nel tuo Main
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            shell=True,
            cwd=PROJECT_ROOT # Forza l'esecuzione dalla radice del progetto
        )

        if result.returncode != 0:
            print(f"  [!] Errore Java (Exit {result.returncode})")
            print(f"  [STDERR]: {result.stderr.strip()}")
            return 0.0

        # 3. Estrazione Recall@100
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

    # Assicurati che la cartella results esista
    os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)

    file_exists = os.path.isfile(OUTPUT_LOG)
    with open(OUTPUT_LOG, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["threshold", "max_terms", "w1", "w2", "w3", "recall100"])

    for i in range(NUM_TENTATIVI):
        print(f"\n--- Trial {i+1}/{NUM_TENTATIVI} ---")

        t = round(random.uniform(0.85, 1), 3)
        m = random.randint(0, 5)
        w1 = round(random.uniform(12, 18), 3)
        w2 = round(random.uniform(15, 25), 3)
        w3 = round(random.uniform(0, 5), 3)

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