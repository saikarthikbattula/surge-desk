# Local validation record

Generated from local command outputs. This is not a hosted CI or LinuxONE result.

- Self-test: 133 passed, 0 failed, 0 skipped; foundation errors: 0.
- Synthetic smoke: passed; repeated predictions and quality metrics match.
- Both research notebooks executed from fresh kernels; source checksums recorded in RESEARCH_VALIDATION.json.
- Public split specification reconstructed the identical ordered workloads without exposing report text or review notes.
- Candidate Git paths excluded data/, models/, and outputs/; aggregate public JSON contained no report-text, individual gold-label, or annotation fields.

Chrome Guest opened the task-specific local demo without a profile-sync prompt.
Automated visual inspection verified the synthetic banner, incident cards,
ranking reasons, location warnings, evidence table, and an expanded disclosure
showing four Houston fixture reports. This is a static desktop visual check,
not human usability testing or a UI screen recording. No account or sync
settings were changed. The user confirmed that no consenting testers are
available. Hosted CI, human usability results, and LinuxONE validation are not
claimed.

## Clean checkout verification

A fresh local clone of `fb7fb4c` passed in a standalone Python 3.14.0 Windows
virtual environment with system packages disabled and no private data or model
files. Self-test: 132 passed, 0 failed, 1 skipped; foundation errors: 0. The
skipped test requires local MiniLM weights. Dependency consistency, import
isolation, the synthetic smoke command, and both fresh-kernel notebooks passed.
Exact dependency versions and log checksums are recorded in the JSON record.

The copied runtime had Tornado 6.5.5, below the notebook kernel requirement.
Installing compatible Tornado 6.6 in the clean environment resolved that
dependency conflict; `pip check` then passed. This does not alter the versions
or provenance of earlier benchmark runs. Hosted Linux CI remains unverified.
