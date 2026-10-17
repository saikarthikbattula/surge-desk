#!/usr/bin/env bash
# Download all public data onto the LinuxONE VM.
# Who: Person 2 (or Person 1). Where: VM host shell (ssh), from the repo root ~/shared/surgedesk
#   bash scripts/download_data.sh
# Safe to re-run: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW/humaid" "$RAW/gazetteer" "$RAW/svi" "$RAW/kaggle_disaster_tweets" \
         data/labels models/minilm models/granite outputs

unzip_py() { python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"; }

echo "== 1/3 HumAID Set 1 (~48K labeled tweets) =="
if [ ! -f "$RAW/humaid/.done" ]; then
  wget -q --show-progress -O "$RAW/humaid/set1.tar.gz" \
    https://crisisnlp.qcri.org/data/humaid/HumAID_data_events_set1_47K.tar.gz
  tar -xzf "$RAW/humaid/set1.tar.gz" -C "$RAW/humaid"
  touch "$RAW/humaid/.done"
fi
echo "Event files found:"
find "$RAW/humaid" -name "*.tsv" | sort
echo "Events:"
find "$RAW/humaid" -name "*_train.tsv" -exec basename {} _train.tsv \; | sort

echo "== 2/3 Census Gazetteer (places + counties) =="
for kind in place counties; do
  if ls "$RAW/gazetteer"/*_Gaz_${kind}_national.txt >/dev/null 2>&1; then echo "$kind: already present"; continue; fi
  ok=0
  for year in 2023 2024 2022; do
    url="https://www2.census.gov/geo/docs/maps-data/data/gazetteer/${year}_Gazetteer/${year}_Gaz_${kind}_national.zip"
    if wget -q -O "/tmp/gaz_${kind}.zip" "$url"; then
      unzip_py "/tmp/gaz_${kind}.zip" "$RAW/gazetteer"; echo "$kind: $year ok"; ok=1; break
    fi
  done
  [ "$ok" = 1 ] || echo "ACTION NEEDED: download the Census Gazetteer ${kind} national file by hand into $RAW/gazetteer"
done
ls -la "$RAW/gazetteer"

echo "== 3/3 CDC SVI (county level) =="
if [ -f "$RAW/svi/svi_counties.csv" ]; then
  echo "SVI present: $(head -1 "$RAW/svi/svi_counties.csv" | cut -c1-120)"
else
  cat <<'MSG'
ACTION NEEDED (manual, 3 minutes):
  1. On a laptop, open the CDC/ATSDR SVI "Data & Documentation Download" page.
  2. Choose: Year = 2022 (or latest), Geography = United States, Geography Type = Counties, File Type = CSV.
  3. Rename the file to svi_counties.csv and copy it to the VM:
       scp -i <key>.pem svi_counties.csv linux1@<VM_IP>:~/shared/surgedesk/data/raw/svi/
MSG
fi

echo
echo "Next: in a JupyterLab terminal (inside the container) run:"
echo "  cd /home/jovyan/shared/surgedesk && python scripts/download_minilm.py"
