# Person 3 handoff: Our own Granite model

You drive **Agent 3** (`handoffs/agent3_brief.md`). You own converting IBM Granite 3.3 2B Instruct ourselves into a big-endian GGUF for LinuxONE (the organizers want our own model, not the demo lab's), getting it onto the VM, and judging the quality of Granite's output.

## 1. Your timeline (Central)
| Time | Do |
|---|---|
| 1:00 PM | Clone the repo; start Agent 3 on branch `a3-brief` with: "Read AGENTS.md and handoffs/agent3_brief.md, then implement it. Use a fake LLM until I give you the model file." Start the conversion (section 2) right away; it's mostly download time |
| ~2:30 PM | Conversion done. Give the little-endian file to Agent 3's laptop setup (section 3). Start the `scp` of the big-endian file to the VM |
| 4:30 PM | Checkpoint 1 |
| 5:30 PM | Agent 3's `--granite` test passes on your laptop |
| 6:00 PM | VM load test with Person 1 (section 4) |
| 8:30 PM | Checkpoint 2: agree `TOP_N_BRIEF` with Person 1 from the measured speed |
| 8:30 PM to 12:30 AM | Review briefings; tune prompts if needed (section 5) |
| 12:30 to 3:00 AM | Extraction eval on hand labels; collect 5 failure cases (section 6) |
| 3:00 to 6:00 AM | Sleep |
| 6:00 to 7:30 AM | Edit the video with Person 2 |

## 2. Convert Granite (1:00 PM, your laptop)
Needs Python 3.10+, git, ~15 GB free disk. On Windows, use WSL.
```bash
cd <repo root>
bash scripts/convert_granite.sh
```
The script clones llama.cpp, installs its converter requirements in a private venv, downloads `ibm-granite/granite-3.3-2b-instruct` (~5 GB), and writes two Q8_0 files in `~/granite_convert/`:
- `granite-3.3-2b-instruct-be.q8_0.gguf`: **big-endian**, for LinuxONE
- `granite-3.3-2b-instruct-le.q8_0.gguf`: little-endian, for laptops
It checks the byte order of both and fails loudly if either is wrong.

Pitch line this earns: "We converted IBM Granite 3.3 2B ourselves to a big-endian GGUF for LinuxONE."

## 3. Distribute the files (~2:30 PM)
- **VM** (start early, it's ~2.7 GB): `scp -i <key>.pem ~/granite_convert/granite-3.3-2b-instruct-be.q8_0.gguf linux1@148.100.112.180:~/shared/surgedesk/models/granite/`
- **Agent 3's laptop** (yours): put the LE file in `<repo>/models/granite/` and develop with `export SURGEDESK_GRANITE=models/granite/granite-3.3-2b-instruct-le.q8_0.gguf`; `pip install llama-cpp-python==0.3.36` on your laptop only.
- If `scp` is slow, use `rsync -avP` so it can resume.

## 4. VM load test (~6:00 PM, with Person 1)
Person 1 runs `notebooks/03_brief.ipynb` cells 1 to 4. Check:
- [ ] Loads without "GGUF file version ... extremely large" (that error = little-endian file).
- [ ] No "unknown model architecture" / metadata error. If you get one: the converter is newer than the llama.cpp bundled with llama-cpp-python 0.3.36. Find that version's `vendor/llama.cpp` submodule commit in the llama-cpp-python GitHub repo (tag v0.3.36), then `LLAMA_CPP_REF=<commit> bash scripts/convert_granite.sh` and re-copy.
- [ ] Test prompt answers in plain English, not gibberish.
- [ ] Verbose log screenshot taken (any zDNN/NNPA lines highlighted).
- [ ] Post: load seconds, seconds per extraction call, tokens/sec, RSS.

## 5. Quality review (8:30 PM to 12:30 AM)
Read every briefing for the demo snapshot in `03_brief.ipynb` output:
- Are grounded fields correct? Are rejected sentences correctly rejected?
- If too many sentences get rejected or fields are empty, Agent 3 may tune the **prompts only** (wording, examples). Never loosen `grounding.py` checks to make numbers look better.
- If acceptance is still poor at midnight, that's fine: the template fallback is honest. Note it as a limitation.

## 6. Evaluation and failure cases (12:30 to 3:00 AM)
- After `hand_labels.csv` exists, Person 1 runs the eval stage (it calls `extract_for_eval`). Review per-field match rates in `docs/RESULTS.md`.
- Collect **5 concrete failure cases** with screenshots for the limitations slide: e.g. a wrong merge, a missed place, an ungrounded field, a rejected but true sentence, a misranked incident. Put them in the shared folder with one line each. Give them to Person 4 by 3:00 AM.
