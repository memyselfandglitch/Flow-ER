# FlowER test on Vast.ai

## Historical operation — targeted pilot (2026-10-29)

**Latest status: pilot complete; A100 rental cancelled.** The earlier approved
time limit stopped audit inference at 996/1,024. The user then authorized another
30 minutes. After repairing the expired startup deadline through Vast's
stopped-container file transfer, audit inference resumed at row 996 and completed
the remaining 28 inputs. No checkpoint, source, model, cohort or ranking settings
changed. All earlier prediction prefixes were preserved exactly.

All 1,024 fit, 512 tune and 1,024 audit inputs now have 64 samples each. Thirty
remote pilot files were checksum-verified locally before shutdown. Instance
`58220553` was cancelled; Vast's final inventory contained no instances. Frozen
model assessment and reporting then ran locally. The final running window was
approximately 3.6 minutes ($0.07); all three A100 pilot running windows total
approximately $4.15, excluding retained-storage time, transfer charges and earlier
benchmark rentals. These are lifecycle estimates, not an invoice reconciliation.

On the fixed 1,024-input validation audit, frequency scored 862 (84.18%) and the
targeted reranker 871 (85.06%): 19 corrections, 10 introduced errors, net +0.879
percentage points, 95% grouped interval [-0.100, +1.898]. The result is
**inconclusive** under the frozen criteria. The global ablation had five
corrections and five harms; the chemistry rule had reverted to frequency during
tuning. The full benchmark remains paused at 20,152 rows.

See [completed pilot report](results/reranking/targeted-pilot-20261029/REPORT.md),
[all 38 targeted switches](results/reranking/targeted-pilot-20261029/changed-cases.jsonl),
and [operating receipt](results/reranking/targeted-pilot-20261029/operating-receipt.json).
Final supervisor: `.vast/targeted-a100-finalrun-lifecycle.json`; cancellation:
`.vast/targeted-a100-finalrun-cancelled.json`. Recovery archive:
`results/reranking/targeted-final-copy2-recovery-20261029/`.

The operational history below describes earlier launches and extensions.

The user authorized an A100 pilot and cancellation of other rentals. The retained
H100 `58056052` could not restart because host resources were unavailable. No
pilot inference ran there. All 57 local snapshot files were reverified; the H100
was then destroyed, and Vast's instance list was verified empty. Its SSH address
below is historical and must not be used for a current connection.

The replacement launcher `.vast/launch_targeted_a100.py` uses one 80 GB A100,
with a three-hour wall-clock ceiling, stopping sooner on completion. Its rate
ceiling is $1.25/hour (at most $3.75), within the original $4.29 pilot estimate.
The full benchmark remains paused at 20,152 rows.

The fixed pilot cohort is 1,024 train-fit, 512 validation-tune and 1,024 separate
validation-audit steps, each with 64 samples and K=10. The reserved 512 test rows
remain unprocessed. Artifacts and protocol:
`results/reranking/targeted-pilot-20261029/`. Fit/tune inference precedes CPU model
freezing; audit inference follows it. Local and remote guards enforce the deadline.
Progress: `.vast/targeted-a100-lifecycle.json`; stop receipt:
`.vast/targeted-a100-stop-verified.json`. Nothing resumes the full benchmark.

Completed pilot: instance `58220553`, `NVIDIA A100-SXM4-80GB`, $1.194222/hour.
Remote preflight passed: 28 uploaded files, checkpoint and preserved prediction
hashes verified, PyTorch 2.4.0, RDKit 2024.3.3, and 78.8 GiB free at preflight.
The following SSH address is historical; this rental has been cancelled:

```bash
ssh -p 22127 -i /Users/deveshi/Desktop/Flow-ER/.vast/id_ed25519 root@104.37.174.34
```

On the instance, monitor:

```bash
cat /workspace/Flow-ER/results/reranking/targeted-pilot-20261029/inference-progress.json
tail -f /workspace/Flow-ER/results/reranking/targeted-pilot-20261029/logs/inference-audit-attempt-0001.log
```

The local backup connection later timed out. Server inference completed all
1,024 fit rows, then hit a broken output pipe before starting tune. The launcher
stopped the A100. Recovery verified all 1,024 saved rows (65,536 samples) and the
unchanged original prediction prefix, then resumed tuning with detached remote
execution and retries for transient backup failures. Recovery history is under
`results/reranking/targeted-recovery-20261029/`; active recovery state is
`.vast/targeted-a100-resume1-lifecycle.json`.

The user approved a further 30 minutes, moving the maximum deadline from
16:35 to 17:05 IST on 2026-10-29. Maximum 3.5-hour cost at the observed rate is
about $4.18; stop sooner if complete. The proposal and user approval are recorded
in `.vast/targeted-deadline-extension-proposal.json` and
`.vast/targeted-a100-extension1-approval.json`. This changes the operational
deadline only; cohort membership and evaluation rules remain fixed.

## Preserved full inference — paused on 2026-10-28

At the user's request, full inference is paused at **20,152 elementary-step
cases**, with **198,845 remaining**. These are individual next-state predictions,
not complete reactions or paths from initial reactants.

- At preservation, H100 instance `58056052` was stopped: Vast reported
  `cur_state=stopped`, `intended_status=stopped`, `actual_status=exited`.
  GPU compute is stopped; retained storage remains chargeable.
- Both completion monitors were disabled before the pause. No automatic full
  benchmark continuation or instance deletion is active.
- The local snapshot is `results/paused-20261028/`. All 57 manifest files were
  SHA-256 verified, including checkpoint, predictions, logs, source code,
  dependency versions, and run settings. The test input checksum was verified
  against the already-preserved local dataset.
- The prediction prefix contains 22,582,509 bytes and SHA-256
  `0b19d2b517d926470b9dc95b19ba3a846035b3d7957d7d0e376b8409ec7d755f`.
  Every completed row has 64 samples; there is no partial final row.
- Receipts: `results/paused-20261028/run-config.json`, `snapshot-manifest.json`,
  and `pause-verified.json`. Rental metadata is in `.vast/h100-rental.json`.
- A separate pilot has been prepared, but was **not launched**. Its train/val/test
  subset manifest is `results/reranking/pilot-v2/selection.json`. Its intended
  compute cap is two hours and it writes separate prediction files.
- The latest request explicitly targets next-step ranking failures. The local
  cached-data study is complete: all 2,724 top-1-wrong/top-10-correct cases are
  isolated, and two rerankers were fitted and assessed on separate connected
  groups within the diagnosed prefix. Neither establishes a reliable gain.
  See [failure-study report](results/reranking/failure-study/REPORT.md).
- The exploratory assessment does not qualify as an untouched evaluation.
  The 512 reserved future test rows remain unprocessed. No new GPU inference
  was started for this study, and full benchmark continuation remains paused.
- The subsequent [label/evaluation audit](results/label-audit-20152/REPORT.md)
  covers every completed step. It distinguishes competing recorded branches
  from 435 predictions without another observed target for the same input.
  Numeric IDs identify potentially branching networks, not single trajectories.
  The larger reranker proposal is on hold while label interpretation is reviewed.
  No unprocessed test target labels or new GPU inference were used in this audit.
- The [50-case stratified review](results/stratified-review-50/REPORT.md) is
  complete (2026-10-29), using local cached cases only. It proposes narrow
  validity/continuation diagnostics; it does not demonstrate a reranker gain.
  The preserved full run and reserved evaluation subset were not restarted.

Resume the full benchmark only after credible pilot benefit or a decision to
obtain the final reproduction result. Restore/verify the saved prefix before
restarting, use a fresh process-bound monitor, and never use the old migration
launcher (it is bound to the earlier 11,859-row snapshot).

The retained H100 connection, after explicitly starting it, is currently:

```bash
ssh -p 20532 -i /Users/deveshi/Desktop/Flow-ER/.vast/id_ed25519 root@152.53.151.205
```

Recheck the SSH address after a restart. Current-run preservation and resumption
settings are recorded in the pause snapshot; the previous A100 migration is
historical.

### OOM recovery on 2026-10-28

The original run completed 11,281 elementary-step rows before a 394-atom input exhausted
GPU memory. The output was verified locally and remotely with SHA-256
`ab5a913b36fddd25e011bbad08a7d3976ffec47f9db5d13ac7e1bf21c1d33833`.
The instance was stopped with its storage preserved.

The recovery patch sizes sample chunks by `batch_size * atoms²`, with a default
budget of 4,000,000 matrix elements, and halves a failed chunk on CUDA OOM.
Successful chunks are retained; every step input still receives all 64 samples.
An OOM at one sample remains an explicit failure. A separate fix prevents empty
batches when resuming at an oversized step input. Sample scheduling can change
random-number assignment and adaptive ODE numerics, so this is not bitwise
identical to an unsplit run; the checkpoint, sample count, precision, and ODE
tolerances are unchanged.

Recovery passed 10 local regression tests, two PyTorch integration tests, and
the exact failing 394-atom input with all 64 samples in chunks of 16. The full
test resumed from step row 11,282. At verification, 11,284 rows were complete;
the original 11,281-row prefix retained its exact SHA-256 and all three new rows
contained 64 samples. GPU memory during the resumed large step input was about
11.7 GiB. The suggested expandable-segments allocator option is unsupported by
this image, so it is no longer a launcher default. Logs:
`logs/recovery-tests.log`, `logs/recovery-smoke.log`, `logs/recovery-runner.log`,
and the continuing `logs/full-test.log`. The original failure log is preserved
under `logs/oom-20261028/`. Completion cleanup is attached to the new launcher;
the old stopped timeout process is no longer used.

The matching data and model are staged locally:

- `data/flower_new_dataset/{train,val,test,beam}.txt`
- `checkpoints/flower_new_dataset/best_large_hyperparam/model.2940000_97.pt`
- Test set: 218,997 reaction steps; default evaluation: 64 samples per step.

The original downloaded ZIP archives remain in Downloads. The extracted folders
were moved here. Data, checkpoints, results, `.vast/`, and `.venv-vast/` are ignored
by Git. Input SHA-256 checksums are recorded in `.vast/manifest.json`.

Use `.venv-vast/bin/vastai` for account operations. The system CLI is installed
under Python 3.9 and fails to import; this isolated CLI uses Python 3.11 and the
existing account credential file. No credentials are included in the upload.

## Host and deployment

The current deployment uses one H100 SXM 80 GB GPU with image
`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` and direct SSH. Re-query offers
before creating another rental; hardware availability and prices change.

The API credential uses team context. Vast.ai rejects account-level SSH key
registration with `Team SSH keys are not supported`. Attach the dedicated public
key directly to an instance using `vastai attach ssh INSTANCE_ID .vast/id_ed25519.pub`.
The key is attached to the H100; the private key remains on this Mac.

For a fresh instance, upload the current code, test data, checkpoint, smoke inputs,
and checksum manifest under `/workspace/Flow-ER`. The original
`.vast/flower-test.tar.gz` predates the OOM fix; its code needs the current
`eval_multiGPU.py`, `utils/data_utils.py`, `utils/inference_utils.py`, and launcher.
Then run:

```bash
bash scripts/bootstrap_vast.sh
NUM_GPUS_PER_NODE=1 TEST_FILE="$PWD/.vast/smoke.txt" \
  RESULT_PATH="$PWD/results/smoke" bash scripts/test_vast.sh
```

Confirm the smoke result has 32 rows, each tally sums to 64, the GPU executes
successfully, and record runtime and memory use. The first 32 rows establish
functionality; use a representative larger subset before projecting total cost.

Launch the complete test after establishing the spending limit:

```bash
NUM_GPUS_PER_NODE=1 TEST_BATCH_SIZE=2048 MAX_SAMPLE_MATRIX_ELEMENTS=10000000 \
  NUM_WORKERS=8 bash scripts/test_vast.sh
```

Default token batch size is a conservative 512 per GPU, configurable using
`TEST_BATCH_SIZE`. `NUM_WORKERS` defaults to 4 per GPU. `SAMPLE_SIZE` defaults to
64. Adaptive chunks preserve the requested total, including uneven remainders.
Predictions append on restart using the completed row count; preserve the same
input and sampling configuration. The evaluation writer is restricted to rank
zero, and preprocessing respects the worker allocation.

Outputs appear in
`results/flower_new_dataset/best_large_hyperparam/test-64-2940000_97.txt`.
Check that the completed file contains 218,997 rows before treating it as a full
evaluation. Conservation can be scored with:

```bash
python examples/conservation/flower_conservation.py \
  results/flower_new_dataset/best_large_hyperparam/test-64-2940000_97.txt
```

Copy results and logs back locally before destroying the instance. Stopping an
instance still incurs storage charges. A spending limit needs an external
instance lifecycle watchdog; terminating the Python job alone does not stop
GPU rental charges. The earlier run used both local and container watchdogs;
those monitors were disabled for the preserved pause and must not be reused.
