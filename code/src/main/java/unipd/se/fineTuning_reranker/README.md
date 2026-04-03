# Reranker Fine-Tuning — Qwen3-Reranker-{0.6B → 4B}

Workflow completo per il fine-tuning del reranker generativo Qwen3.
Il codice è scritto per essere **identico** su 0.6B e 4B — cambia solo `--model` e i parametri di batch.

---

## File

| File | Scopo |
|---|---|
| `prepare_reranker_data.py` | Costruisce train/dev/test con hard negatives da BM25 |
| `finetune_qwen3_reranker.py` | Training con LoRA + eval loop |
| `evaluate_reranker.py` | Inference con checkpoint fine-tuned → JSON per Java evaluator |

---

## Workflow rapido

### Step 1 — Prepara il dataset

```bash
python prepare_reranker_data.py \
  --queries data/en_train.json \
  --papers  data/collection_data.json \
  --bm25    results/bm25_results.json \
  --output  data/finetune_reranker \
  --n_hard_neg 15 \
  --bm25_pool  100
```

Output in `data/finetune_reranker/`:
- `train.jsonl` (~11.980 esempi)
- `dev.jsonl`   (~1.498 esempi)  ← usato per early stopping
- `test.jsonl`  (~1.499 esempi)  ← **NON toccare fino alla valutazione finale**

### Step 2 — Fine-tuning su 0.6B (iterazioni rapide)

```bash
python finetune_qwen3_reranker.py \
  --model       Qwen/Qwen3-Reranker-0.6B \
  --batch_size  8 \
  --grad_accum  4 \
  --lora_r      8 \
  --epochs      3 \
  --eval_steps  200 \
  --run_name    ft_06B_r8_hn15
```

Checkpoint salvati in `models/ft_06B_r8_hn15/`.

### Step 3 — Valuta il best checkpoint

```bash
# Valuta solo sul test set (confronto fair)
python evaluate_reranker.py \
  --model       models/ft_06B_r8_hn15/best \
  --test_split  data/finetune_reranker/test.jsonl \
  --output      results/reranked_finetuned_06B_test.json

# Poi passa al Java evaluator
java -cp ... unipd.se.Main --reranked results/reranked_finetuned_06B_test.json
```

### Step 4 — Confronto zero-shot vs fine-tuned (stesso test set)

```bash
python evaluate_reranker.py \
  --model      Qwen/Qwen3-Reranker-0.6B \
  --test_split data/finetune_reranker/test.jsonl \
  --output     results/reranked_zeroshot_06B_test.json
```

---

## Passaggio da 0.6B a 4B

**Cambia solo questi parametri** — tutto il resto del codice è identico:

```bash
python finetune_qwen3_reranker.py \
  --model       Qwen/Qwen3-Reranker-4B \
  --batch_size  2 \
  --grad_accum  16 \
  --lora_r      16 \
  --epochs      2 \
  --eval_steps  200 \
  --run_name    ft_4B_r16_hn15
```

---

## Parametri consigliati per RTX 3090 (24GB)

### 0.6B
| Parametro | Valore |
|---|---|
| `--batch_size` | 8–16 |
| `--grad_accum` | 4 |
| `--lora_r` | 8 |
| `--epochs` | 3 |
| `--n_hard_neg` | 7–15 |

### 4B
| Parametro | Valore |
|---|---|
| `--batch_size` | 2–4 |
| `--grad_accum` | 8–16 |
| `--lora_r` | 16 |
| `--epochs` | 2 |
| `--n_hard_neg` | 7–15 |

---

## Note critiche

### TASK_INSTRUCTION deve essere IDENTICA in tutti i file
`prepare_reranker_data.py` non usa TASK_INSTRUCTION (la aggiunge solo durante training),
ma `finetune_qwen3_reranker.py`, `evaluate_reranker.py` e `CausalReranker.py`
DEVONO avere la stessa stringa. Se la cambi, cambiala ovunque.

### Non valutare sul train set
Il `test.jsonl` viene separato prima del training e non deve essere toccato.
I risultati riportati nella relazione devono provenire dal test set.

### Il best checkpoint è selezionato per accuracy sul dev
L'accuracy dev è un proxy veloce (non nDCG@10). Per la valutazione finale
usa sempre `Evaluator.java` con il test set per le metriche ufficiali.

### LoRA merge in inference
`evaluate_reranker.py` fa automaticamente `merge_and_unload()` per fondere
l'adapter nel modello base prima dell'inference. Questo elimina la latenza
extra dell'adapter e rende il modello identico in velocità a quello base.

---

## Dipendenze

```bash
pip install transformers>=4.51.0 peft>=0.9.0 torch>=2.0.0 tqdm
```