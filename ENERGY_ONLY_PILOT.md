# FlowER energy-only pilot

## Frozen V2 execution completed

The user authorized the exact frozen V2 protocol for the same 18 cases. **71 validated jobs were reused byte-for-byte; all 40 remaining jobs were attempted locally with two workers.** Twelve new jobs passed and 28 failed, including successful single-point calculations for all eight monatomic ion jobs. Scoring and aggregation took **163.8 seconds**; cloud spend was **$0**.

Calculation coverage is **4/18 complete pools and 22/62 candidates**. Energy ranking recovered **3/18 references (3/4 complete pools)**, including the one recovered in V1 and two additional recoveries at validation lines 2711 and 20906. One fully scored pool kept the frequency winner; none switched to another strictly incorrect candidate. Fourteen unresolved pools retained the original prediction by fallback. There were no near-ties within 1 kcal/mol and no complete-pool winner changes under the first-conformer sensitivity check.

- [Detailed outcomes, energy gaps and all unresolved-case reasons](results/reranking/energy-only-focus-v2-20261031-run/DETAILED_REPORT.md)
- [Every candidate structure, energy, original rank/count and provenance](results/reranking/energy-only-focus-v2-20261031-run/cases.html)
- [Assessment summary](results/reranking/energy-only-focus-v2-20261031-run/assessment-summary.json) and [integrity verification](results/reranking/energy-only-focus-v2-20261031-run/assessment-verification.json)
- [Authorization receipt](results/reranking/energy-only-focus-v2-20261031-authorization.json) and [execution log](results/reranking/energy-only-focus-v2-20261031-run/run.log)

The original results, candidate pools, counts, rankings and solvent assignments were verified unchanged. No failed calculation energy was accepted and no new failure was retried. DeepMech, training, new FlowER inference, cloud execution and reserved-test access remain held. This failure-only sample cannot establish overall accuracy gain or measure damage to already-correct predictions.

## Historical V2 preparation freeze — execution was held at preparation

The same **18 cases and 62 candidates**, with byte-identical solvent assignments and pools, are prepared under a versioned correction. **71/111** previously successful fragment–solvent jobs passed revalidation and can be reused. All **40** failed jobs remain quarantined: the rerun plan requires **8 atomic single points and 32 two-conformer molecular jobs**, at most 72 new xTB calls. Allow about **8 minutes** on two local workers, with the existing two-hour absolute cap and no cloud spend.

Current validated calculation availability is **10/62 candidates and 1/18 complete pools**. V2 reference recovery has **not** been assessed. Original run files remain unchanged, and no physical calculation was executed during this correction.

- [Frozen correction, geometry diagnosis and exact execution commands](results/reranking/energy-only-focus-v2-20261031/REPORT.md)
- [Manifest](results/reranking/energy-only-focus-v2-20261031/manifest.json) and [verification receipt](results/reranking/energy-only-focus-v2-20261031-verification.json)
- [17 focused tests](tests/test_xtb_correction_v2.py)
- [Separate proposed DeepMech comparison](DEEPMECH_COMPARISON_NOTE.md), with no model installation, training or inference performed

The revised rules retain full structure checks and whole-pool fallback. Monatomic ions use one properly converged single point; H₂ bond recognition is corrected; resonance differences require an explicit indexed equivalence proof. Unresolved reconstruction and changed-connectivity flags remain failures. A new authorization receipt tied to the exact v2 manifest is required before execution.

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
