#!/usr/bin/env bash
# One command, end to end: environment check -> self-test -> every pipeline stage -> evaluation.
#
# Who: Person 1 only. Where: inside a container on the VM (IBM's Jupyter container during the day,
# our surgedesk container for the final run), from the repo root:
#   bash run_all.sh                 # full run (stages are cached; re-running skips finished work)
#   bash run_all.sh --from rank     # resume from a stage
#   bash run_all.sh --stub          # stub pipeline on fixtures only (Checkpoint 1)
#   FORCE=1 bash run_all.sh         # recompute everything
#
# Stages (python -m surgedesk.pipeline --stage <name>), in order:
#   data       HumAID TSVs -> outputs/<event>/reports.jsonl for every event
#   embed      MiniLM embeddings for every event -> embeddings.npy
#   classifier train category classifier on train events -> models/classifier.joblib
#   tag        tag every event -> tagged.jsonl
#   ranker     group + snapshot train events, train XGBoost -> models/ranker.json
#   rank       group + snapshot + rank the demo event -> incidents.jsonl
#   brief      Granite briefings for the demo snapshot's top N -> briefings.jsonl
#   eval       every metric -> outputs/<event>/summary.json, docs/RESULTS.md, plots
# The benchmark is separate (needs an idle machine): bash run_benchmark.sh
set -euo pipefail
cd "$(dirname "$0")"

STAGES=(data embed classifier tag ranker rank brief eval)
FROM="data"
STUB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --stub) STUB=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

log() { echo; echo "[$(date -u +%H:%M:%S)] $*"; }

log "Environment check"
python -m surgedesk.envcheck

log "Self-test"
python -m surgedesk.selftest

if [ "$STUB" = "1" ]; then
  log "Stub pipeline on fixtures (outputs/_stub/)"
  python -m surgedesk.pipeline --stub --event fixture_event
  log "Stub run finished. Open notebooks/04_demo.ipynb with STUB = True to see the cards."
  exit 0
fi

started=0
for s in "${STAGES[@]}"; do
  [ "$s" = "$FROM" ] && started=1
  [ "$started" = 1 ] || continue
  log "Stage: $s"
  t0=$(date +%s)
  python -m surgedesk.pipeline --stage "$s" $FORCE_FLAG
  log "Stage $s finished in $(( $(date +%s) - t0 ))s"
done
[ "$started" = 1 ] || { echo "unknown stage for --from: $FROM (choose: ${STAGES[*]})"; exit 2; }

log "Executing notebooks/05_eval_benchmark.ipynb (saved with outputs into the mounted repo)"
NB_OUT="${SURGEDESK_ROOT:-$PWD}/notebooks"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir "$NB_OUT" --output 05_eval_benchmark.ipynb notebooks/05_eval_benchmark.ipynb

log "ALL DONE. Results: outputs/${SURGEDESK_EVENT:-hurricane_harvey_2017}/summary.json and docs/RESULTS.md"
log "Demo: open notebooks/04_demo.ipynb in JupyterLab and Run All."
