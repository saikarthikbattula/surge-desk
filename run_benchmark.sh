#!/usr/bin/env bash
# Benchmark on LinuxONE. Run ONLY when no other notebook kernels are running.
# Who: Person 1, inside our surgedesk container on the VM, after run_all.sh finished:
#   bash run_benchmark.sh
# Writes outputs/<event>/benchmark_stream.csv, benchmark_granite.csv, benchmark_embed.csv (if NNPA stretch done)
# and PNG charts, all captioned "measured on IBM LinuxONE (s390x)".
set -euo pipefail
cd "$(dirname "$0")"
export SURGEDESK_THREADS=3
echo "Make sure every other Jupyter kernel is shut down (Kernel > Shut Down All Kernels)."
python -m surgedesk.envcheck
python -m surgedesk.pipeline --stage benchmark
echo "Re-executing notebooks/05_eval_benchmark.ipynb so the benchmark charts appear in it"
NB_OUT="${SURGEDESK_ROOT:-$PWD}/notebooks"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir "$NB_OUT" --output 05_eval_benchmark.ipynb notebooks/05_eval_benchmark.ipynb
echo "Benchmark done."
