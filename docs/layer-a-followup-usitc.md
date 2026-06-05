# Layer A follow-up: switch to USITC HTSUS export

## What

The current Layer A stat-suffix validator is built from a **training-derived index**: we walked `data/formatted/train.jsonl` (119,602 CROSS ruling examples), extracted every gold 10-digit HTS code, and built a map from 8-digit subheading to the list of valid 10-digit completions plus their training-set frequencies.

This is enough for the v1 paper's evaluation (our test sets are drawn from the same CROSS distribution as the training data, so coverage is near-total for the codes that actually appear in the evals).

It is **not** enough for production deployment. A model deployed against real importer queries will see 10-digit codes that never appeared in CROSS ruling training data. Those will be flagged as "invalid" by the training-derived index and routed through the picker, which will silently substitute a more-common code. This is a false-positive correction that degrades production accuracy.

## Why we deferred

Path A (USITC HTSUS export) requires:
1. Figuring out the exact USITC export URL and format. As of 2026 the export is at `https://hts.usitc.gov/reststop/exportList` (CSV or JSON formats available; format query string TBD).
2. A one-off ingestion script to parse the export and produce the same index shape as the training-derived one.
3. A decision about how to merge the official index with the training-derived frequencies (the picker needs frequency data; the USITC export has codes but not frequencies).

That's ~half a day of work. Path B was faster to get a measured lift number for the paper.

## What to do when picking this up

1. **Source the export.** Try `https://hts.usitc.gov/reststop/exportList?format=JSON` first. If the JSON endpoint is unstable, fall back to the CSV format and parse with pandas. Cache the response under `data/external/htsus_2026_export.{csv,json}` and gitignore the raw file (it's large).

2. **Parse it.** Build `data/external/htsus_valid_codes_official.json` with the same shape as the training-derived index:
   ```json
   {
     "8544.30.00": [
       {"code": "8544.30.0010", "freq": null},
       {"code": "8544.30.0020", "freq": null},
       ...
     ]
   }
   ```
   `freq: null` because the official export has no usage frequency info.

3. **Merge with training frequencies.** For codes present in both indexes, copy the training frequency. For codes only in the official index, leave `freq: null` and let the picker fall back to "first valid" or "lowest code" as a deterministic tiebreaker. Codes only in the training index (potentially because the schedule has changed) should be dropped or flagged for review.

4. **Re-run the post-hoc scorer.** `scripts/apply_stat_suffix_validator.py` already accepts an `--index` flag. Just point it at the new index file and produce a fresh comparison: training-derived lift vs official-export lift on the same predictions.jsonl.

5. **Update the paper's results table.** Add a row showing the production-faithful lift number. If it's meaningfully different from the training-derived number, that delta is itself worth a sentence in the limitations section.

## Risk this is worse than expected

If we discover after running A that the USITC official index has wildly different lift behavior from the training-derived index, that's evidence the v1 paper's Layer A measurement was over-optimistic. Worth flagging proactively rather than burying.

Realistic expectation: official-index lift will be within 1-2 pp of training-derived lift on our test sets, because the test sets are CROSS-derived. On a hypothetical out-of-distribution test set (which we don't have), the gap could be larger.

## Trigger conditions

Pick up this work if any of:
- We decide to publish a v1.1 paper with stat-suffix validation as the headline (need the production-faithful number)
- We deploy v1 to a customer or production target (need the complete valid-codes set)
- USITC publishes the 2027 HTSUS revision (annual update, would invalidate any cached export)

If none of these become true within ~6 months, the training-derived index is probably good enough as a permanent state.
