#!/usr/bin/env bash
# Convert IBM Granite 3.3 2B Instruct to GGUF ourselves (big-endian for LinuxONE + little-endian for laptops).
# Who: Person 3. Where: a laptop (macOS/Linux, or WSL on Windows) with Python 3.10+ and ~15 GB free disk.
#   bash scripts/convert_granite.sh
# Then copy the BE file to the VM (command printed at the end).
#
# If the VM later fails to load the file with an "unknown architecture/metadata" error, re-run with
#   LLAMA_CPP_REF=<commit>  bash scripts/convert_granite.sh
# where <commit> is the llama.cpp commit vendored by llama-cpp-python 0.3.36
# (see the vendor/llama.cpp submodule in the llama-cpp-python GitHub repo at tag v0.3.36).
set -euo pipefail
WORK="${WORK:-$HOME/granite_convert}"
MODEL_ID="ibm-granite/granite-3.3-2b-instruct"
OUT_BE="granite-3.3-2b-instruct-be.q8_0.gguf"
OUT_LE="granite-3.3-2b-instruct-le.q8_0.gguf"
mkdir -p "$WORK" && cd "$WORK"

if [ ! -d llama.cpp ]; then git clone https://github.com/ggml-org/llama.cpp; fi
if [ -n "${LLAMA_CPP_REF:-}" ]; then (cd llama.cpp && git fetch --all && git checkout "$LLAMA_CPP_REF"); fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
if [ -f llama.cpp/requirements/requirements-convert_hf_to_gguf.txt ]; then
  pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
else
  pip install -q -r llama.cpp/requirements.txt
fi
pip install -q -U huggingface_hub

echo "== Downloading $MODEL_ID (~5 GB) =="
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_dir='granite-3.3-2b-instruct')"

echo "== Converting: big-endian (LinuxONE) =="
python3 llama.cpp/convert_hf_to_gguf.py granite-3.3-2b-instruct --outtype q8_0 --bigendian --outfile "$OUT_BE"
echo "== Converting: little-endian (laptops) =="
python3 llama.cpp/convert_hf_to_gguf.py granite-3.3-2b-instruct --outtype q8_0 --outfile "$OUT_LE"

python3 - <<PY
import struct
for name, want in [("$OUT_BE", ">"), ("$OUT_LE", "<")]:
    with open(name, "rb") as f:
        magic = f.read(4); raw = f.read(4)
    v_le, v_be = struct.unpack("<I", raw)[0], struct.unpack(">I", raw)[0]
    endian = "big" if v_be < 100 else "little"
    print(f"{name}: magic={magic} version={min(v_le, v_be)} endian={endian}")
    assert magic == b"GGUF"
    assert endian == ("big" if want == ">" else "little"), f"{name} has the wrong byte order"
PY
ls -lh "$OUT_BE" "$OUT_LE"
cat <<MSG

Done. Next:
  1. Copy the big-endian file to the VM:
     scp -i <key>.pem "$WORK/$OUT_BE" linux1@<VM_IP>:~/shared/surgedesk/models/granite/
  2. Give the little-endian file to the Agent 3 laptop (put it in that repo's models/granite/
     and set SURGEDESK_GRANITE=models/granite/$OUT_LE when developing there).
MSG
