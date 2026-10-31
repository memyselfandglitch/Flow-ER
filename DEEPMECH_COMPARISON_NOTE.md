# Proposed DeepMech comparison — not executed

DeepMech is a plausible additional mechanistic score for the fixed FlowER candidate pools. Its graph model predicts reactive bonds and templates of mechanistic operations, including hydrogen-count and charge changes. These capabilities are relevant to the protonation-site distinctions in this pilot. That is a motivation to test complementarity, not evidence that the combination improves our data. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13347294/)

Keep this separate from the frozen energy-only experiment. Adding DeepMech now would change its question and prevent attributing results to the energy score. DeepMech does not repair xTB convergence or bond reconstruction. A learned model also cannot recover absent reaction conditions or uniquely adjudicate competing recorded outcomes from identical model inputs.

Proposed next steps, requiring separate authorization before new inference or training:

1. Audit the released code/model and ReactMech training provenance, including potential overlap with FlowER development/evaluation reaction networks. Do not open reserved test outcomes. The authors provide an [official repository](https://github.com/alhqlearn/DeepMech); its suitability as a scorer of externally supplied candidates still needs verification.
2. Check input representation, explicit-H/charge conventions, atom correspondence, stopping-state handling and mechanistic-operation coverage for every saved FlowER candidate. A missing template or unscorable candidate is a coverage failure, not a zero probability or evidence of chemical invalidity. Preserve every candidate and define full-row fallback before evaluation.
3. On grouped validation development data, compare frequency ranking, a DeepMech-derived score on the same pools, and the frozen energy approach as separate arms. A frequency-plus-DeepMech policy would be a further declared arm with weights and switching margins fitted only inside fitting folds; include zero intervention.
4. Assess separate groups including baseline-correct controls and a representative validation sample. Report scoring coverage separately from corrections, new errors, wrong-to-wrong switches and overall paired change. The 18 inspected failure cases are development diagnostics, not an untouched assessment set.
5. Consider combination with energy only if separate ablations show complementary benefits. Freeze all methods and fallback rules before final reserved-test evaluation.

The paper reports strong results on its own ReactMech benchmark, but those numbers do not establish improvement on our FlowER split. Its stopping classifier and template-based scoring also require explicit adaptation; they are not a drop-in replacement for the existing energy-only score. No DeepMech model has been installed, trained or run for this proposal.
