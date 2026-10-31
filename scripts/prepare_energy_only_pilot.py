#!/usr/bin/env python3
"""Prepare the held energy-only pilot using immutable, versioned solvent assignments.

No xTB invocation, conformer generation, model fitting, inference or cloud API.
"""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import random
import shutil
import time

from rdkit import Chem, rdBase
import energy_only_pilot as pilot

ROOT = pilot.ROOT
DEFAULT_SOLVENT_REVIEW = ROOT/'results/reranking/solvent-review-20261031'
STRATA = pilot.STRATA
CODE_FILES = [Path(__file__), ROOT/'scripts/energy_only_pilot.py', ROOT/'scripts/score_xtb_candidates.py',
    ROOT/'scripts/rerank_analysis.py', ROOT/'scripts/phase0_reranking_audit.py',
    ROOT/'scripts/audit_recorded_targets.py', ROOT/'utils/data_utils.py']


def frozen_solvents(directory):
    """Consume frozen assignments; never recompute, amend or guess a solvent."""
    manifest = json.loads((directory/'manifest.json').read_text())
    for name in ('assignments.jsonl', 'dictionary.json'):
        if pilot.sha(directory/name) != manifest['output_sha256'][name]:
            raise ValueError('Frozen solvent artifact hash mismatch: '+name)
    allowed = {str(p.relative_to(ROOT)) for p in [pilot.VAL]+pilot.SOURCES}
    if set(manifest['source_sha256']) != allowed:
        raise ValueError('Solvent provenance must contain precisely the authorized validation sources')
    for name, expected in manifest['source_sha256'].items():
        if pilot.sha(ROOT/name) != expected:
            raise ValueError('Validation source differs from frozen solvent provenance: '+name)
    rows = [json.loads(line) for line in (directory/'assignments.jsonl').open()]
    if len(rows) != 21723 or {r['source_row_id'] for r in rows} != set(range(21723)):
        raise ValueError('Frozen assignments must cover all 21723 unique validation rows')
    return {r['source_row_id']:r for r in rows}, manifest


def join_assignment(row, assignments):
    a = assignments[row['source_row_id']]
    if a['source_line_sha256'] != row['source_line_sha256']:
        raise ValueError('Frozen solvent input provenance mismatch')
    if a['candidate_pool_sha256'] != pilot.fingerprint(row['candidates']):
        raise ValueError('A saved candidate, count, rank or other pool field changed')
    return a


def in_starting_set(row, assignment):
    return (row['target_rank'] is not None and 2 <= row['target_rank'] <= 10 and
            assignment['status'] == 'assumed' and bool(assignment['proposed_alpb_name']))


def full_pool_screen(row):
    """Screen every retained candidate, even if another candidate already fails."""
    candidate_checks, failures = [], []
    for candidate in row['candidates']:
        try:
            local = pilot.static_pool_check(dict(row, candidates=[candidate]))
        except Exception as error:
            local = [{'rank':candidate['rank'], 'reason':'unparseable_or_uncheckable_candidate',
                      'exception_type':type(error).__name__}]
        candidate_checks.append({'rank':candidate['rank'], 'smiles_sha256':pilot.fingerprint(candidate['smiles']),
            'static_compatible':not local, 'failures':local})
        failures.extend(local)
    return candidate_checks, failures


def select_distinct_groups(eligible, quota=25, seed=20261031):
    """Seeded maximum-cardinality group-to-stratum matching, then row selection.

    Augmenting paths avoid losing a scarce stratum's group to the other one
    when an alternative group can satisfy that other slot. No energy fields
    or sampling counts participate in priorities or tie-breaking.
    """
    rng = random.Random(seed); members = {}; preferences = {}
    for stratum in STRATA:
        by_group = defaultdict(list)
        for row in eligible:
            if row['stratum'] == stratum: by_group[row['group']].append(row)
        groups = sorted(by_group); rng.shuffle(groups)
        preferences[stratum] = groups
        for group in groups:
            candidates = sorted(by_group[group], key=lambda r:r['source_row_id'])
            rng.shuffle(candidates); members[(stratum, group)] = candidates[0]
    order = sorted(STRATA, key=lambda s:(len(preferences[s]), STRATA.index(s)))
    slots = [(stratum, i) for stratum in order for i in range(quota)]
    owners = {}
    def augment(slot, visited):
        for group in preferences[slot[0]]:
            if group in visited: continue
            visited.add(group)
            if group not in owners or augment(owners[group], visited):
                owners[group] = slot
                return True
        return False
    for slot in slots: augment(slot, set())
    chosen = [members[(slot[0], group)] for group, slot in owners.items()]
    return sorted(chosen, key=lambda r:(STRATA.index(r['stratum']), r['source_row_id']))


def chemical_failure_reasons(proof):
    accepted = {'no_change', 'verified_graph_proton_relocation'}
    result = []
    for key, relation in proof.items():
        if relation['status'] not in accepted: result.append(key+':'+relation['status'])
    return result or ['comparison_is_not_a_distinct_single_proton_choice']


def render_review(out, selected, screening):
    # Reuse the complete-structure renderer without changing any candidate.
    pilot.case_table(out, selected)
    text = (out/'cases.html').read_text()
    for case in selected:
        old = f'<h2>Validation line {case["source_row_id"]+1} · '
        new = f'<h2 id="line-{case["source_row_id"]+1}">Validation line {case["source_row_id"]+1} · '
        text = text.replace(old, new)
    (out/'cases.html').write_text(text)
    columns = ['Validation line','Category','Frozen nominal solvent','Reference rank',
               'Graph screen','Complete-pool screen','Selection / exclusion']
    page = ['<!doctype html><meta charset="utf-8"><title>FlowER pilot screening review</title>',
        '<style>body{font:14px/1.5 system-ui;margin:24px;color:#203342}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd5dc;text-align:left;vertical-align:top}th{background:#edf2f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}input,select{font:inherit;margin:8px;padding:7px}</style>',
        '<h1>Energy-only pilot: proposed cases and all 527 screening decisions</h1>',
        '<p>Preparation only. All candidate pools and solvent assignments are unchanged. A static compatibility pass does not guarantee successful embedding, convergence or topology preservation. Energy and recovery results are unavailable.</p>',
        '<p><a href="REPORT.md">Report</a> · <a href="cases.html">Complete proposed structures and recorded alternatives</a> · <a href="screening.jsonl">All screening records</a> · <a href="manifest.json">Frozen provenance</a></p>',
        '<label>Show <select id="status"><option value="">All</option><option value="proposed">Proposed</option><option value="excluded">Excluded</option><option value="eligible_not_selected">Eligible, not selected</option></select></label><label>Search <input id="query" placeholder="line, solvent, category, reason"></label><p id="count"></p>',
        '<table><thead><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr></thead><tbody>']
    for r in sorted(screening, key=lambda r:r['source_row_id']):
        reasons = r['exclusions'] or [r['selection_reason']]
        detail = {'chemical_evidence':r['chemical_proof'], 'static_failures':r['static_failures'],
                  'checked_candidate_ranks':[c['rank'] for c in r['candidate_checks']],
                  'frozen_solvent_reasoning':r['solvent']['reasoning']}
        line = r['source_row_id']+1
        first = f'<a href="cases.html#line-{line}">{line}</a>' if r['selection_status']=='proposed' else str(line)
        values = [first, html.escape(r['stratum']), html.escape(r['solvent']['nominal_name'])+' (assumed)',
            str(r['target_rank']), 'pass' if r['chemical_eligible'] else 'excluded',
            'pass ('+str(len(r['candidate_checks']))+' candidates)' if r['static_compatible'] else 'excluded',
            html.escape('; '.join(reasons))+'<details><summary>Evidence and provenance</summary><pre>'+html.escape(json.dumps(detail,indent=2))+'</pre></details>']
        page.append('<tr data-status="'+r['selection_status']+'">'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    page.append('</tbody></table><script>const rows=[...document.querySelectorAll("tbody tr")];function filter(){let n=0;const status=document.getElementById("status").value,q=document.getElementById("query").value.toLowerCase();for(const r of rows){const show=(!status||r.dataset.status===status)&&(!q||r.textContent.toLowerCase().includes(q));r.hidden=!show;n+=show}document.getElementById("count").textContent=n+" rows shown"}document.getElementById("status").onchange=filter;document.getElementById("query").oninput=filter;filter();</script>')
    (out/'review.html').write_text('\n'.join(page))


def write_report(out, summary, selected):
    report = ['# Proposed energy-only failure-recovery pilot', '',
        '**Preparation only. Physical calculations, model training, inference, cloud rentals and reserved test outcomes remain held.**', '',
        'The frozen solvent assignments were consumed without recalculation or modification. The verified starting set contains **527** validation rows with reference ranks 2–10, an assumed nominal solvent and a declared ALPB mapping. All 21,723 official-validation identities and original candidate pools were checked against the source files.', '',
        '| Screening stage | Another recorded outcome | Unmatched in validation catalog |',
        '| --- | ---: | ---: |']
    for name in ('starting_rows','chemical_pass_rows','static_compatible_rows','both_pass_rows','eligible_groups','proposed_rows','quota_shortfall'):
        report.append('| '+name.replace('_',' ')+' | '+' | '.join(str(summary['strata'][s][name]) for s in STRATA)+' |')
    report += ['',f"The proposal contains **{len(selected)} cases from {len({c['group'] for c in selected})} distinct connected groups**. No criterion was relaxed to fill a quota. Every starting row was screened for both chemistry and all-candidate static compatibility; the stage pass counts overlap rather than forming a sequential funnel.", '',
        '[Review all 527 screening decisions and exclusions](review.html). [Inspect every proposed candidate structure and all recorded alternatives](cases.html). Machine-readable evidence is in [screening.jsonl](screening.jsonl), [selected-cases.jsonl](selected-cases.jsonl), and [exclusions.jsonl](exclusions.jsonl).', '',
        '## Fixed eligibility and grouping', '',
        'The conservative graph test requires an input-to-reference and input-to-frequency-winner comparison that is either no change or a single heteroatom proton relocation, with at least one actual relocation. It checks conserved inventory/net charge, unchanged heavy-atom bond graph, one H attachment lost and one gained, and matching formal-charge changes under every enumerated heavy-graph correspondence. It cancels identical spectator fragments first. Carbon-site transfers, unsupported changed free-H fragments, heavy-bond changes, radical changes and incomplete/ambiguous correspondence enumeration remain excluded. This is a structural eligibility test, not independent chemical or kinetic adjudication. Reference atom maps do not supply candidate correspondence.', '',
        'Every retained candidate is checked for parseability, element/isotope inventory and net-charge conservation, supported elements, closed-shell/even-electron representation, absence of unsupported isotopes, and at most 40 heavy atoms per fragment. A failure in any candidate excludes the row before selection. No candidate is removed to make a row pass. Embedding, optimization convergence and topology preservation cannot be checked without a later authorized physical run.', '',
        'Connected groups join identical exact model inputs, evaluator-normalized complete mixtures, provided grouping metadata and numeric validation reaction networks, transitively across all validation rows. The fixed seed is **20261031**. Seeded group priorities and maximum-cardinality group-to-category matching enforce both 25-case quotas where feasible; scarce groups cannot be unnecessarily consumed by the other category. A row is chosen deterministically within each assigned group. Counts, energies, rarity and prospective recovery do not influence selection.', '',
        '“Another recorded outcome” means a frequency winner found among **official-validation targets for the same evaluator-normalized full input**. The inspected test prefix and training labels are not searched. Exact model-input agreement is also included per case. “Unmatched” means no witness in this validation catalog, not proven chemical invalidity. All recorded alternatives and their validation source lines are shown for competing-label cases.', '',
        '## Proposed cases', '',
        '| Validation line (source ID) | Category | Frozen assumed solvent | Reference rank | Winner / reference count |',
        '| --- | --- | --- | ---: | ---: |']
    for c in selected:
        ref = next(x for x in c['candidates'] if x['smiles']==c['target'])
        report.append(f"| [{c['source_row_id']+1} ({c['source_row_id']})](cases.html#line-{c['source_row_id']+1}) | {'recorded alternative' if c['stratum']==STRATA[0] else 'unmatched'} | {c['solvent']['nominal_name']} | {c['target_rank']} | {c['candidates'][0]['count']} / {ref['count']} |")
    report += ['', '## Exclusions', '', 'Reasons below count rows and can overlap:', '',
        '| Reason | Recorded-alternative rows | Unmatched rows |', '| --- | ---: | ---: |']
    reasons = sorted(set().union(*(summary['strata'][s]['exclusion_reasons'] for s in STRATA)))
    for reason in reasons:
        report.append('| '+reason+' | '+' | '.join(str(summary['strata'][s]['exclusion_reasons'].get(reason,0)) for s in STRATA)+' |')
    b = summary['preflight']
    report += ['', '## Preserved protocol and cost preflight', '',
        f"The selected pools contain **{b['candidate_entries']} candidate entries**. The plan contains **{b['unique_fragment_solvent_jobs']} distinct fragment–solvent jobs**, at two prescribed conformers each (**{b['conformer_optimizations_upper_bound']} optimizations**). Counts retain every molecule/ion with its multiplicity in candidate sums; identical fragment jobs may be cached without removing multiplicity.", '',
        f"The unchanged cap is two local workers, one thread each, 20 wall-seconds per fragment including both conformers, at most 720 unique jobs and **two elapsed hours**. The job-time bound is {b['aggregate_job_wall_seconds_upper_bound']} aggregate worker-seconds; ideal two-worker scheduling would be at most {b['ideal_two_worker_seconds_upper_bound']} seconds before orchestration overhead. These are timeout bounds, not measured runtimes or promised successful coverage. No compute was rented or used for physical calculations.", '',
        'The proposed physical score remains GFN2-xTB + ALPB total energy, summed over fragments using the minimum of two prescribed converged, topology-preserving conformers. Use the frozen row-level solvent, explicit `bar1mol` reference state, 300 K electronic-temperature setting (not reaction temperature), ETKDGv3 seed 42, and normal optimization. ALPB total energy already includes solvation. Ranking uses energy only; no frequency term, fitted lambda or learned correction. A common baseline-energy subtraction is reporting only. A 1 kcal/mol near-tie annotation is fixed for reporting and is not a tuned switching rule.', '',
        'Any later failed/unreliable candidate makes its whole selected row unresolved, retains frequency operationally, and stays in the selected-sample denominator. Do not replace failed cases or rank a successful subset. Selection, complete pools, solvent-assignment hashes and protocol settings are frozen now for review. Future amendments require a new version; no existing solvent assignment was changed.', '',
        f"Local xTB readiness: **{b['xtb_readiness']}**. The package was not installed or invoked by this preparation. A real binary/version check and the frozen-protocol execution integration are still needed before an explicitly authorized physical run. The existing `run` command refuses execution while held.", '',
        '## Checks and commands', '',
        'The preparation verifies source/assignment hashes before and after the run, exact pool joins, 21,723-row coverage, the 527-row starting set, every candidate check, quota bounds and group uniqueness. Energy/recovery fields are deliberately unavailable; no overall accuracy conclusion follows from this failure-only design.', '',
        '```bash', '.venv-analysis/bin/python scripts/prepare_energy_only_pilot.py --out results/reranking/energy-only-frozen-solvents-REPRODUCTION',
        '.venv-analysis/bin/python -m unittest discover -s tests -p test_prepare_energy_only_pilot.py', '```', '',
        'Use a fresh output directory. This script only prepares files; it has no physical execution mode. The [manifest](manifest.json) records input, source-code, complete-pool, solvent and artifact hashes.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--solvent-review', type=Path, default=DEFAULT_SOLVENT_REVIEW)
    args = parser.parse_args(); out = args.out; solvent_dir = args.solvent_review.resolve()
    if out.exists(): raise SystemExit('Use a fresh output directory; existing artifacts are preserved.')
    out.mkdir(parents=True); started = time.monotonic()
    code_hashes = {str(p.relative_to(ROOT)):pilot.sha(p) for p in CODE_FILES}
    solvent_hashes = {name:pilot.sha(solvent_dir/name) for name in ('manifest.json','assignments.jsonl','dictionary.json')}
    assignments, solvent_manifest = frozen_solvents(solvent_dir)
    proto = pilot.protocol('immutable assignments from '+str(solvent_dir.relative_to(ROOT)))
    proto['version'] = 'flower-energy-only-frozen-solvents-20261031-v2'
    proto['selection'].update(expected_starting_rows=527,
        randomization='seeded group priorities, maximum-cardinality group-to-stratum matching, seeded row within assigned group',
        frozen_solvent_assignments_sha256=solvent_hashes['assignments.jsonl'])
    pilot.save(out/'protocol.json', proto)
    print('Verifying complete validation records and frozen assignment joins', flush=True)
    rows, source_counts = pilot.load_validation()
    for r in rows: join_assignment(r, assignments)
    starting = [r for r in rows if in_starting_set(r, assignments[r['source_row_id']])]
    if len(starting) != 527: raise ValueError(f'Expected 527 starting rows; found {len(starting)}. No silent scope change.')
    print('Confirmed 527 starting rows; building transitive groups and validation-only label catalogs', flush=True)
    catalog, exact_catalog, witnesses = pilot.identities_and_catalogs(rows)
    screening, eligible = [], []
    for r in starting:
        a = assignments[r['source_row_id']]; labels = catalog[r['reactant']]
        first = r['candidates'][0]['smiles']; stratum = STRATA[0] if first in labels else STRATA[1]
        chemical, proof = pilot.chemical_eligibility(r)
        candidate_checks, failures = full_pool_screen(r)
        exclusions = []
        if not chemical: exclusions += chemical_failure_reasons(proof)
        exclusions += sorted({'candidate_pool:'+f['reason'] for f in failures})
        info = {'source_row_id':r['source_row_id'], 'group':r['_group'], 'stratum':stratum,
            'target_rank':r['target_rank'], 'chemical_eligible':chemical, 'chemical_proof':proof,
            'static_compatible':not failures, 'candidate_checks':candidate_checks, 'static_failures':failures,
            'exclusions':exclusions, 'solvent':{'status':a['status'], 'solvent':a['proposed_alpb_name'],
                'nominal_name':a['proposed_nominal_solvent'], 'reasoning':a['reasoning'],
                'frozen_assignment_sha256':pilot.fingerprint(a)},
            'candidate_pool_sha256':a['candidate_pool_sha256'], 'network_id':r['raw_sequence_id']}
        screening.append(info)
        if exclusions: continue
        alternatives = [{'smiles':s, 'count':n, 'source_row_ids':witnesses[r['reactant']][s]} for s,n in sorted(labels.items())]
        eligible.append(dict(info, row_id=r['source_row_id'], reactant=r['reactant'], target=r['target'],
            candidates=r['candidates'], sample_count=r['sample_count'],
            recorded_alternatives=alternatives, exact_input_sha256=r['_exact_input'],
            top1_has_exact_input_recorded_match=first in exact_catalog[r['_exact_input']],
            frozen_solvent_assignment=a, source_file=r['_source_file'], source_line_sha256=r['source_line_sha256']))
    selected = select_distinct_groups(eligible); selected_ids = {r['source_row_id'] for r in selected}
    selected_groups = {r['group'] for r in selected}
    assert len(selected) == len(selected_groups) and all(sum(r['stratum']==s for r in selected)<=25 for s in STRATA)
    for item in screening:
        if item['source_row_id'] in selected_ids:
            item.update(selection_status='proposed', selection_reason='selected_by_fixed_seed_with_unique_group')
        elif item['exclusions']:
            item.update(selection_status='excluded', selection_reason='failed_fixed_screen')
        else:
            reason = 'group_already_represented' if item['group'] in selected_groups else 'category_quota_reached'
            item.update(selection_status='eligible_not_selected', selection_reason=reason)
    jobs = sorted({(c['solvent']['solvent'], fragment) for c in selected for candidate in c['candidates'] for fragment in pilot.fragments(candidate['smiles'])})
    if len(jobs)>proto['budget']['max_unique_jobs']: raise ValueError('Job cap exceeded; criteria were not relaxed or cases silently removed.')
    summary = {'status':'proposed_for_review_physical_execution_held','verified_validation_rows':len(rows),
        'source_counts':source_counts,'starting_rows':len(starting),'strata':{},'selected_rows':len(selected),
        'selected_groups':len(selected_groups),'no_criteria_relaxed':True,'frozen_solvents_changed':False,
        'energy_calculations':0,'new_inference':0,'training_runs':0,'cloud_spend':0}
    for s in STRATA:
        part = [r for r in screening if r['stratum']==s]; e = [r for r in eligible if r['stratum']==s]; chosen = [r for r in selected if r['stratum']==s]
        summary['strata'][s] = {'starting_rows':len(part),'starting_groups':len({r['group'] for r in part}),
            'chemical_pass_rows':sum(r['chemical_eligible'] for r in part),
            'static_compatible_rows':sum(r['static_compatible'] for r in part),
            'both_pass_rows':len(e),'eligible_groups':len({r['group'] for r in e}),
            'proposed_rows':len(chosen),'quota_shortfall':25-len(chosen),
            'exclusion_reasons':dict(Counter(reason for r in part for reason in r['exclusions']))}
    summary['preflight'] = {'candidate_entries':sum(len(c['candidates']) for c in selected),
        'unique_fragment_solvent_jobs':len(jobs),'conformer_optimizations_upper_bound':2*len(jobs),
        'aggregate_job_wall_seconds_upper_bound':20*len(jobs),
        'ideal_two_worker_seconds_upper_bound':20*((len(jobs)+1)//2),
        'elapsed_cap_seconds':7200,'workers':2,'xtb_readiness':'found_not_executed' if shutil.which('xtb') else 'not_found_on_PATH'}
    pilot.jsonl(out/'screening.jsonl', screening); pilot.jsonl(out/'eligible-cases.jsonl', eligible)
    pilot.jsonl(out/'selected-cases.jsonl', selected); pilot.jsonl(out/'exclusions.jsonl', [r for r in screening if r['exclusions']])
    pilot.save(out/'jobs.json', [{'solvent':s,'fragment':f} for s,f in jobs])
    pilot.save(out/'summary.json', summary)
    print(json.dumps(summary,indent=2), flush=True)
    render_review(out, selected, screening); write_report(out, summary, selected)
    for name,h in code_hashes.items():
        if pilot.sha(ROOT/name)!=h: raise RuntimeError('Code changed during preparation; refuse freeze')
    for name,h in solvent_hashes.items():
        if pilot.sha(solvent_dir/name)!=h: raise RuntimeError('Frozen solvent input changed during preparation')
    for name,h in solvent_manifest['source_sha256'].items():
        if pilot.sha(ROOT/name)!=h: raise RuntimeError('Original validation source changed during preparation')
    names = ['protocol.json','screening.jsonl','eligible-cases.jsonl','selected-cases.jsonl','exclusions.jsonl','jobs.json','summary.json','cases.html','review.html','REPORT.md']
    pilot.save(out/'manifest.json', {'status':'frozen_proposal_for_review_not_execution',
        'validation_source_sha256':solvent_manifest['source_sha256'],
        'frozen_solvent_directory':str(solvent_dir.relative_to(ROOT)), 'frozen_solvent_files_sha256':solvent_hashes,
        'code_sha256':code_hashes, 'artifact_sha256':{name:pilot.sha(out/name) for name in names},
        'selected_source_row_ids':[r['source_row_id'] for r in selected],
        'rdkit_version':rdBase.rdkitVersion,'preparation_elapsed_seconds':time.monotonic()-started,
        'execution_holds':proto['execution_holds'],'test_observations_used':False,'reserved_test_outcomes_opened':False})
    print('Frozen proposal complete; all physical execution remains held.', flush=True)


if __name__=='__main__':main()
