# FlowER energy-only pilot

## Authorized local focus run — 2026-10-31

The user's request to start the focused energy test authorized the **18 frozen unmatched cases**, with all 62 retained candidate entries and 111 distinct fragment–solvent jobs. Execution is local, with two workers, two prescribed conformers and a two-hour total limit. The other 25 proposed cases, training, new inference, cloud execution and reserved-test access remain held.

**Completed in 242.5 seconds; cloud spend $0.** Only **1/18** pools passed every reliability check; that case's reference moved from rank 2 to rank 1. The other 17 rows remain unresolved and retain the original prediction. Recovery is **1/18 overall selected** and **1/1 fully scored**, too little coverage to judge the energy method. No retries or post-result changes to the protocol were made.

- [Results and per-case ranks](results/reranking/energy-only-focus-20261031/REPORT.md)
- [Structures, energies and provenance for every candidate](results/reranking/energy-only-focus-20261031/cases.html)
- [Coverage diagnosis and next protocol proposal](results/reranking/energy-only-focus-20261031/POST_RUN_DIAGNOSIS.md)
- [Post-run integrity verification](results/reranking/energy-only-focus-20261031/verification.json)

- [Execution receipt and frozen scope](results/reranking/energy-only-focus-20261031/execution.json)
- [Append-only progress log](results/reranking/energy-only-focus-20261031/run.log)
- [Machine-readable progress](results/reranking/energy-only-focus-20261031/progress.json)
- [Authorization receipt](results/reranking/energy-only-focus-20261031-authorization.json)

xTB 6.7.1 is installed in `~/.local/share/flower-xtb`. A hash-bound [compatibility wrapper](scripts/xtb_671_reference_state.py) translates the frozen protocol's `bar1mol` notation to the backend's `bar1M` argument. This preserves the intended 1-bar gas/1-molar solution reference convention; the original scorer, frozen solvent assignments and proposal are unchanged. The translation follows the [versioned xTB argument parser](https://github.com/grimme-lab/xtb/blob/v6.7.1/src/prog/main.F90#L1734-L1746) and was fixed before any physical calculation.

Exact launch command (already launched; the runner refuses an existing output directory):

```bash
.venv-analysis/bin/python scripts/run_energy_focus.py \
  --out results/reranking/energy-only-focus-20261031 \
  --xtb scripts/xtb_671_reference_state.py \
  --authorization results/reranking/energy-only-focus-20261031-authorization.json
```

## Historical preparation freeze

**Preparation complete; all physical and other execution holds remain in place.**

From the 527 recoverable validation errors with an unchanged frozen assumed solvent and a declared ALPB mapping, 194 rows pass the fixed graph and complete-pool compatibility screens. The deterministic proposal contains **43 cases from 43 connected groups: 25 recorded-alternative cases and 18 unmatched cases**. The unmatched pool has 19 eligible rows in 18 groups, so its quota shortfall is seven; no criteria were relaxed.

- [Screening review: all 527 rows, proposed cases and exclusions](results/reranking/energy-only-frozen-solvents-20261031/review.html)
- [Complete candidate structures and all recorded alternatives](results/reranking/energy-only-frozen-solvents-20261031/cases.html)
- [Report, case list and cost preflight](results/reranking/energy-only-frozen-solvents-20261031/REPORT.md)
- [Frozen manifest](results/reranking/energy-only-frozen-solvents-20261031/manifest.json) and [verification receipt](results/reranking/energy-only-frozen-solvents-20261031/verification.json)

The proposed pools retain 131 candidate entries, with 215 distinct fragment–solvent jobs and 430 prescribed conformer optimizations. These are planning counts, not executed calculations. The cap remains two local workers and two elapsed hours. Physical compatibility is a static check; actual optimization/convergence has not been tested. The xTB binary was not found on PATH, and no installation or calculation was performed.

The frozen solvent assignments, original FlowER checkpoint, candidate strings/counts/rankings, and reserved test outcomes remain unchanged. No training, inference or cloud work occurred. This failure-only diagnostic cannot estimate overall accuracy improvement.
