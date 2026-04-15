"""
merge_lora_and_export.py

Fonde i LoRA adapter nel modello base e salva un modello merged completo.

Utile per:
  1. Deployment più veloce (no overhead PEFT a inference time)
  2. Convertire a formati quantizzati (GGUF, AWQ, GPTQ)
  3. Condividere il modello finale senza dipendenze PEFT

Uso:
  python merge_lora_and_export.py \
    --lora_checkpoint ./checkpoints/modernbert-reranker/best_checkpoint \
    --output_dir      ./checkpoints/modernbert-reranker-merged
"""

import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def merge_and_save(lora_checkpoint: str, output_dir: str):
    lora_path = Path(lora_checkpoint)
    out_path  = Path(output_dir)

    if not (lora_path / "adapter_config.json").exists():
        raise ValueError(f"{lora_path} non sembra un checkpoint PEFT valido.")

    log.info("Caricamento config PEFT...")
    peft_cfg = PeftConfig.from_pretrained(str(lora_path))

    log.info(f"Caricamento modello base: {peft_cfg.base_model_name_or_path}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_cfg.base_model_name_or_path,
        num_labels=1,
        torch_dtype=torch.float32,   # float32 per il merge (più stabile)
        ignore_mismatched_sizes=True,
    )

    log.info("Caricamento adapter LoRA...")
    model = PeftModel.from_pretrained(base_model, str(lora_path))

    log.info("Fusione LoRA nel modello base...")
    model = model.merge_and_unload()

    log.info(f"Salvataggio modello merged in {out_path}...")
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(out_path))

    # Verifica rapida
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"✓ Modello merged salvato. Parametri totali: {n_params/1e6:.1f}M")
    log.info(f"  Path: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter e salva modello completo")
    parser.add_argument("--lora_checkpoint", required=True)
    parser.add_argument("--output_dir",      required=True)
    args = parser.parse_args()

    merge_and_save(args.lora_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()