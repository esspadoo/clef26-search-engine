#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

QUERY_FILE="code/data/Dev_set/en_dev.json"
TOP100_RUN="runs/reranked_results_nemotron_topk100BASELINE_DEV.json"
TOP1000_RUN="runs/reranked_results_nemotron_topk1000BASELINE_DEV.json"
LORA_RUN="runs/reranked_results_nemotronFTAarsen20252026_Lora_enDEVtopk1000_onbiEncoderTopk1000.json"

STATS_DIR="results/statistics"
TMP_HOME="$ROOT_DIR/.tmp-stats-home"

TOP100_JSON="$STATS_DIR/nemotron_top100_baseline_en_dev.json"
TOP1000_JSON="$STATS_DIR/nemotron_top1000_baseline_en_dev.json"
LORA_JSON="$STATS_DIR/nemotron_lora_top1000_en_dev.json"

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
}

check_python_dependencies() {
    python3 -c 'import importlib.util, sys
modules = ["pandas", "matplotlib", "statsmodels", "scipy"]
missing = [module for module in modules if importlib.util.find_spec(module) is None]
if missing:
    print("Missing Python dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install them with: python3 -m pip install " + " ".join(missing), file=sys.stderr)
    raise SystemExit(1)
'
}

run_statistical_evaluator() {
    local ranking_file="$1"
    local output_json="$2"
    local system_name="$3"

    echo "[java] Generating $output_json"
    mvn -q exec:java \
        -Dexec.mainClass=unipd.se.StatisticalEvaluator \
        -Dexec.args="$QUERY_FILE $ranking_file $output_json $system_name"
}

run_metric_analysis() {
    local metric="$1"
    local outdir="$2"
    local title="$3"

    echo "[python] Running $metric analysis"
    HOME="$TMP_HOME" \
    MPLBACKEND=Agg \
    MPLCONFIGDIR="$TMP_HOME/.config/matplotlib" \
    XDG_CACHE_HOME="$TMP_HOME/.cache" \
    python3 code/py/anova_tukey_boxplot.py \
        --inputs \
        "$TOP100_JSON" \
        "$TOP1000_JSON" \
        "$LORA_JSON" \
        --metric "$metric" \
        --outdir "$outdir" \
        --title "$title"
}

require_file "$QUERY_FILE"
require_file "$TOP100_RUN"
require_file "$TOP1000_RUN"
require_file "$LORA_RUN"

mkdir -p "$STATS_DIR" "$TMP_HOME/.cache" "$TMP_HOME/.config/matplotlib"

echo "[check] Verifying Python dependencies"
check_python_dependencies

echo "[build] Compiling Java sources"
mvn -q -DskipTests compile

run_statistical_evaluator "$TOP100_RUN" "$TOP100_JSON" "nemotron_top100_baseline_en_dev"
run_statistical_evaluator "$TOP1000_RUN" "$TOP1000_JSON" "nemotron_top1000_baseline_en_dev"
run_statistical_evaluator "$LORA_RUN" "$LORA_JSON" "nemotron_lora_top1000_en_dev"

run_metric_analysis "mrr5" "$STATS_DIR/en_dev_mrr5" "English DEV - MRR@5"
run_metric_analysis "ndcg10" "$STATS_DIR/en_dev_ndcg10" "English DEV - nDCG@10"
run_metric_analysis "ap" "$STATS_DIR/en_dev_ap" "English DEV - AP"

echo "[done] Statistical validation artifacts written under $STATS_DIR"
