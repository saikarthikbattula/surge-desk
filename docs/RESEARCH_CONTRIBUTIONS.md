# Contribution record

The original SurgeDesk prototype and its multi-author history are preserved on
`a4-integration` at `6d20725`. They are not claimed as solo work.

The improvement implementation is local commit `ed6c48b`, authored under the
repository's configured identity `Ish0212`, on `surgedesk-month-improvements`.
Inspect it with `git show --stat ed6c48b` or inspect individual changes with
`git show ed6c48b -- PATH`.

| New work | Reviewable implementation |
| --- | --- |
| Split provenance, leakage review, fixed workloads | `surgedesk/research_data.py`, public split specification, preparation script |
| Local classifier comparison and selection | `surgedesk/research.py`, `research_sweep.py`, computed results and experiment history |
| Cache validity and recovery | `surgedesk/artifacts.py`, `pipeline_state.py`, embedding/tagging integration, regression tests |
| Location diagnostics and uncertainty | Gazetteer/matcher additions, masked review exports, annotation guide and backlog |
| Delivery and evidence | Fixture CI, fresh-kernel notebooks, synthetic demo/recording, computed case study and validation records |

The work was implemented with Codex assistance. Development review annotations
are explicitly agent-assisted, not independent human validation. Runtime/data
and model artifacts stay local; public evidence contains aggregate metrics and
text-free workload metadata.

Use the measured, qualified resume suggestions in `RESEARCH_CASE_STUDY.md`.
Missing-person recall regressed despite the overall macro-F1 improvement and
must not be omitted when discussing tradeoffs. No hosted PR, public release,
human usability result, or new LinuxONE measurement is claimed by this record.

Local closeout: `680d6fa` records standalone clean-checkout verification. The
local research release is tagged `surgedesk-month-v1` on the improvement branch;
inspect its full authored history with
`git log --oneline 6d20725..surgedesk-month-v1`. The release includes automated
desktop visual review, with human usability and LinuxONE limitations retained.
