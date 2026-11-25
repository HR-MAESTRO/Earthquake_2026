# Hash check of the pre-registration records (2026-11-25)

Each `protocol_vN_hashes.txt` gives the freeze timestamp and SHA-256 hashes of the
protocol text, plus (v2, v3) the source code and (v3) the input data at the time of
freezing. Checked on 2026-11-25:

| Record | Result |
| --- | --- |
| v1 protocol text | `protocol_v1_as_frozen.md` matches the recorded hash. `protocol_v1_with_outcome.md` has the outcome appended after the test, so it does not match. |
| v2 protocol text and 5 source files | All match (`protocol_v2_as_frozen.md`). |
| v3 protocol text, source code and 110 data files | All match except `src/induced/models.py` (below). |

`src/induced/models.py` changed after the v3 freeze. On 2026-11-25 two models
were added for the post hoc analysis of the 2023 task (Section 6.1): an ordinal
logistic model and an optional TabPFN model. No existing model or setting was
changed, and the protocol v3 test does not use this file's new code. The v3 run
itself is recorded in `protocol_v3_run_marker.txt` and `protocol_v3_outcome.md`.

The v1 record lists the protocol under its absolute path on the analysis computer.
It is kept verbatim so that the record stays unaltered.
