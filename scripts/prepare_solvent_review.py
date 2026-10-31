#!/usr/bin/env python3
"""Label-blind whole-molecule solvent recognition; no physical calculations."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

from rdkit import Chem, rdBase, RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/reranking/full-validation-test20k-20261029'
VAL = ROOT/'data/flower_new_dataset/val.txt'
SOURCES = [BASE/'cache/val.jsonl']+[BASE/f'inference/records/val{i:03}.jsonl' for i in range(5)]
DICTIONARY = Path(__file__).with_name('solvent_dictionary.json')
PARSER = Chem.SmilesParserParams(); PARSER.removeHs = False


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): result.update(block)
    return result.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_jsonl(path, values):
    with path.open('w') as stream:
        for value in values: stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


@lru_cache(maxsize=100000)
def whole_molecule_key(smiles):
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Dictionary matching requires a parseable whole molecule/ion')
    for a in mol.GetAtoms(): a.SetAtomMapNum(0)
    # Standardize ordinary explicit H only; retain isotope, charge and stereochemistry.
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def mixture_counts(smiles):
    # A dot separates disconnected whole molecules in these saved SMILES.
    return Counter(whole_molecule_key(part) for part in smiles.split('.'))


def load_dictionary(path=DICTIONARY):
    contents = json.loads(path.read_text()); catalog = {}
    for entry in contents['entries']:
        key = whole_molecule_key(entry['smiles'])
        if key in catalog: raise ValueError('Duplicate whole-molecule dictionary entry')
        catalog[key] = dict(entry, canonical_smiles=key)
    return contents, catalog


def recognize(input_smiles, candidate_smiles, catalog):
    """Assignment accepts no target, correctness, rank frequency, energy or rarity."""
    inputs = mixture_counts(input_smiles)
    candidate_parts, unreadable = [], []
    for rank, smiles in enumerate(candidate_smiles, 1):
        try: candidate_parts.append(mixture_counts(smiles))
        except Exception:
            candidate_parts.append(None); unreadable.append(rank)
    recognized = []
    for key in sorted(inputs):
        if key not in catalog: continue
        entry = catalog[key]
        counts = [parts.get(key, 0) if parts is not None else None for parts in candidate_parts]
        changed = [rank for rank, n in enumerate(counts, 1) if n != inputs[key]]
        recognized.append({'name': entry['name'], 'canonical_smiles': key,
            'organic': entry['organic'], 'likely_roles': entry['roles'],
            'input_explicit_molecule_count': inputs[key],
            'candidate_explicit_molecule_counts_in_saved_order': counts,
            'unchanged_across_complete_pool': bool(candidate_smiles) and not changed,
            'changed_or_unreadable_candidate_ranks': changed,
            'possible_alpb_name': entry['alpb_name']})
    flags, proposed, alpb = [], None, None
    status = 'unresolved'
    if not recognized:
        flags.append('no_dictionary_solvent_recognized')
        reasoning = 'No exact whole-molecule solvent match; do not infer a solvent from other fragments.'
    elif len(recognized) > 1:
        flags.append('possible_mixed_solvent_or_reagent_roles')
        reasoning = 'Multiple recognized solvent-capable molecules, including water/alcohols where present. Their actual roles and solvent proportions require review; no pure-solvent assignment.'
    elif not recognized[0]['organic']:
        flags.append('water_only_role_unclear')
        reasoning = 'Water may be solvent, reagent or by-product. The authorized sole-organic-solvent rule does not assign aqueous conditions automatically.'
    elif not candidate_smiles:
        flags.append('empty_saved_pool')
        reasoning = 'No saved candidate pool is available to verify unchanged solvent identity and multiplicity.'
    elif not recognized[0]['unchanged_across_complete_pool']:
        flags.append('recognized_solvent_changes_or_cannot_be_checked')
        reasoning = 'The recognized molecule is missing, changes multiplicity, or cannot be checked in at least one retained candidate. Review its possible reagent/proton-acceptor role.'
    else:
        status = 'assumed'; proposed = recognized[0]['name']; alpb = recognized[0]['possible_alpb_name']
        reasoning = 'Exactly one recognized organic solvent; the same whole molecule and multiplicity occur in the input and every saved candidate. Proposed as a nominal solvent only; bulk identity and concentration are not experimentally documented.'
        if alpb is None:
            flags.append('alpb_solvent_mapping_not_declared')
            reasoning += ' No ALPB solvent mapping is declared; review before any future physical protocol.'
    if unreadable: flags.append('unreadable_saved_candidate')
    return {'recognized_molecules': recognized,
        'unrecognized_input_fragments': [{'canonical_smiles': key, 'explicit_count': n} for key, n in sorted(inputs.items()) if key not in catalog],
        'proposed_nominal_solvent': proposed, 'proposed_alpb_name': alpb,
        'status': status, 'reasoning': reasoning, 'review_required': status == 'unresolved' or bool(flags),
        'review_flags': flags, 'saved_candidate_count': len(candidate_smiles),
        'input_canonical_smiles': '.'.join(key for key, n in sorted(inputs.items()) for _ in range(n)),
        'experimental_solvent_documentation': None,
        'multiplicity_interpretation': 'Molecular inventory only; never solvent proportions, concentration or evidence of bulk dominance.'}


def recognize_saved_record(record, raw_line, catalog):
    """Only input and saved candidate structures determine the assignment."""
    source = raw_line.strip().split('>>', 1)[0]
    decision = recognize(source, [c['smiles'] for c in record['candidates']], catalog)
    return dict(decision, source_row_id=record['source_row_id'], validation_line=record['source_row_id']+1,
        source_line_sha256=record['source_line_sha256'],
        candidate_pool_sha256=stable_hash(record['candidates']))


def render_html(path, rows, summary):
    # Render text with textContent; JSON is escaped for safe embedding in a local page.
    payload = json.dumps(rows, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('&', '\\u0026')
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlowER nominal-solvent review</title><style>
body{font:14px/1.5 system-ui;color:#203342;margin:24px;background:#f7f9fa}h1{margin-bottom:8px}p{max-width:1100px}select,input,button{font:inherit;padding:8px;margin:4px;border:1px solid #b4c2ce;border-radius:5px;background:white}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;vertical-align:top;border:1px solid #d6dfe5;padding:10px}th{background:#eaf0f3}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}small{color:#526572}.status{font-weight:700}details{margin-top:8px}.controls{margin:18px 0}.rowline{min-width:80px}#pager{margin:12px 0}#summary{font-weight:600}
</style><h1>FlowER · nominal-solvent recognition and review</h1>
<p>All 21,723 official-validation rows. Matches are canonical whole molecules, including water and alcohols. Assignments use the input and all saved candidate structures; no reference answers, frequencies, rarity or energies enter the assignment rule. “Assumed” is a nominal proposal, not documented reaction conditions. Explicit molecule counts are not solvent proportions.</p>
<p id="summary"></p><p><a href="REPORT.md">Report</a> · <a href="assignments.jsonl">Frozen assignments</a> · <a href="review-required.jsonl">Rows requiring review</a> · <a href="dictionary.json">Curated dictionary</a> · <a href="manifest.json">Provenance</a></p>
<div class="controls"><label>Status <select id="status"><option value="">All</option><option>assumed</option><option>unresolved</option><option>documented</option></select></label><label>Review <select id="review"><option value="">All</option><option value="yes">Required</option><option value="no">Not flagged</option></select></label><label>Molecule <select id="molecule"><option value="">All</option></select></label><label>Validation line <input id="line" type="number" min="1" max="21723" placeholder="One-based line"></label><label>Reason <input id="reason" placeholder="water, mixed, changed…"></label></div>
<div id="pager"><button id="previous">Previous</button><span id="position"></span><button id="next">Next</button></div>
<table><thead><tr><th>Validation row</th><th>Recognized molecules and possible roles</th><th>Proposed nominal solvent</th><th>Reasoning / review</th><th>Status</th></tr></thead><tbody id="body"></tbody></table>
<script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent);let page=0,filtered=rows;const size=50;
const ids=['status','review','molecule','line','reason'], els=Object.fromEntries(ids.map(x=>[x,document.getElementById(x)]));
const names=[...new Set(rows.flatMap(r=>r.recognized_molecules.map(m=>m.name)))].sort();
function el(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}
for(const name of names){const option=el('option',name);option.value=name;els.molecule.append(option)}
document.getElementById('summary').textContent=__SUMMARY__;
function filter(){page=0;const text=els.reason.value.trim().toLowerCase();filtered=rows.filter(r=>(!els.status.value||r.status===els.status.value)&&(!els.review.value||r.review_required===(els.review.value==='yes'))&&(!els.molecule.value||r.recognized_molecules.some(m=>m.name===els.molecule.value))&&(!els.line.value||r.validation_line===Number(els.line.value))&&(!text||(r.reasoning+' '+r.review_flags.join(' ')).toLowerCase().includes(text)));draw()}
function draw(){const body=document.getElementById('body');body.replaceChildren();for(const r of filtered.slice(page*size,(page+1)*size)){const tr=el('tr');const id=el('td','Line '+r.validation_line);id.className='rowline';id.append(el('small',' · source ID '+r.source_row_id));const input=el('details');input.append(el('summary','Complete input'),el('pre',r.input_canonical_smiles));id.append(input);tr.append(id);const molecules=el('td');if(!r.recognized_molecules.length)molecules.append(el('p','None in dictionary'));for(const m of r.recognized_molecules){molecules.append(el('strong',m.name),el('div',m.likely_roles.join('; ')),el('small','Explicit count '+m.input_explicit_molecule_count+'; '+(m.unchanged_across_complete_pool?'unchanged in every saved candidate':'changed/unreadable candidate ranks: '+m.changed_or_unreadable_candidate_ranks.join(', '))));const detail=el('details');detail.append(el('summary','Exact structure and counts'),el('pre',m.canonical_smiles+'\nCandidate counts in saved order: '+JSON.stringify(m.candidate_explicit_molecule_counts_in_saved_order)));molecules.append(detail)}tr.append(molecules);const proposed=el('td',r.proposed_nominal_solvent||'Unresolved');proposed.append(el('small',r.proposed_alpb_name?' · proposed ALPB name: '+r.proposed_alpb_name:' · no ALPB assignment'));tr.append(proposed);const why=el('td',r.reasoning);why.append(el('pre',r.review_flags.join('\n')));tr.append(why);const state=el('td',r.status);state.className='status';tr.append(state);body.append(tr)}document.getElementById('position').textContent=' '+filtered.length+' matching rows · page '+(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/size))+' ';document.getElementById('previous').disabled=page===0;document.getElementById('next').disabled=(page+1)*size>=filtered.length}
for(const e of Object.values(els))e.addEventListener('input',filter);document.getElementById('previous').onclick=()=>{page--;draw()};document.getElementById('next').onclick=()=>{page++;draw()};draw();
</script></html>'''
    text = f"Documented: {summary['statuses'].get('documented',0):,} · Assumed: {summary['statuses'].get('assumed',0):,} · Unresolved: {summary['statuses'].get('unresolved',0):,} · Review required: {summary['review_required']:,}"
    path.write_text(template.replace('__DATA__', payload).replace('__SUMMARY__', json.dumps(text)))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out
    if out.exists(): raise SystemExit('Use a fresh output directory; frozen assignments are not overwritten.')
    out.mkdir(parents=True)
    start = time.monotonic(); code_before = sha(Path(__file__)); dictionary_before = sha(DICTIONARY)
    sources_before = {str(p.relative_to(ROOT)): sha(p) for p in [VAL]+SOURCES}
    raw = VAL.read_text().splitlines(keepends=True)
    if len(raw) != 21723: raise ValueError('Unexpected official-validation size')
    dictionary, catalog = load_dictionary(); write_json(out/'dictionary.json', dictionary)
    rows, seen, file_counts = [], set(), {}
    for path in SOURCES:
        n = 0
        for line in path.open():
            r = json.loads(line); i = r['source_row_id']; n += 1
            if i in seen or i not in range(21723) or r['source_split'] != 'val':
                raise ValueError('Duplicate, invalid or non-validation source identity')
            seen.add(i)
            if hashlib.sha256(raw[i].encode()).hexdigest() != r['source_line_sha256']:
                raise ValueError('Validation line provenance mismatch')
            if len(r['candidates']) > 10 or [c['rank'] for c in r['candidates']] != list(range(1,len(r['candidates'])+1)):
                raise ValueError('Incomplete or misordered saved pool')
            rows.append(recognize_saved_record(r, raw[i], catalog))
        file_counts[str(path.relative_to(ROOT))] = n
        print(f'{path.name}: {n} rows; cumulative {len(rows)}', flush=True)
    if seen != set(range(21723)): raise ValueError('Incomplete validation source coverage')
    if code_before != sha(Path(__file__)) or dictionary_before != sha(DICTIONARY):
        raise RuntimeError('Preparation code/dictionary changed during execution; do not freeze this output')
    if any(sha(ROOT/p) != h for p,h in sources_before.items()):
        raise RuntimeError('An authorized data source changed during preparation')
    rows.sort(key=lambda r:r['source_row_id'])
    flags = Counter(flag for r in rows for flag in r['review_flags'])
    summary = {'validation_rows': len(rows), 'unique_rows': len(seen), 'source_counts': file_counts,
        'statuses': dict(Counter(r['status'] for r in rows)), 'review_required': sum(r['review_required'] for r in rows),
        'review_flags_overlapping': dict(flags),
        'assumed_nominal_solvents': dict(Counter(r['proposed_nominal_solvent'] for r in rows if r['status']=='assumed')),
        'dictionary_entries': len(catalog), 'physical_calculations': 0, 'training_runs': 0,
        'new_inference': 0, 'cloud_execution': 0, 'reserved_test_outcomes_accessed': False,
        'assignments_use_reference_answers': False, 'pilot_selection_performed': False,
        'preparation_elapsed_seconds': time.monotonic()-start}
    write_jsonl(out/'assignments.jsonl', rows); write_jsonl(out/'review-required.jsonl', [r for r in rows if r['review_required']])
    write_json(out/'summary.json', summary)
    render_html(out/'review.html', rows, summary)
    report = ['# FlowER solvent recognition and review', '',
        '**Recognition only; no energy calculations, training, new inference, cloud work or pilot-case selection.**', '',
        f"Verified {len(rows):,} unique official-validation rows from the six completed prediction files against all 21,723 lines of `val.txt`. No test observations or reserved test outcomes were opened.", '',
        '| Status | Rows |', '| --- | ---: |']
    report += [f"| {s} | {summary['statuses'].get(s,0):,} |" for s in ('documented','assumed','unresolved')]
    report += ['',f"Rows flagged for review: **{summary['review_required']:,}**.", '',
        '[Open the searchable review table](review.html). It includes the one-based validation line, source row ID, recognized molecules and possible roles, proposed nominal solvent, reasoning, status, exact whole-molecule structures, and checks across every saved candidate.', '',
        'The rule recognizes full disconnected molecules by canonical SMILES after removing atom maps and ordinary explicit hydrogen notation. Charge, isotopes and stereochemistry remain part of matching. It uses no substructure match or fragment position. Input fragments outside the 40-entry dictionary remain unclassified; the dictionary is not exhaustive.', '',
        'One recognized organic solvent that is unchanged, including multiplicity, in every saved candidate receives an **assumed** nominal assignment. This is the user-authorized modeling assumption, not experimental solvent documentation. Water and alcohols are included when detecting possible mixtures. Multiple recognized solvent-capable molecules, water alone, a changed recognized solvent, an empty/unreadable pool or no recognized solvent leave the context unresolved. No solvent proportions or concentrations are inferred from explicit molecule counts.', '',
        'Role descriptions are plausible alternatives from the curated dictionary, not reaction-specific chemical adjudication. A molecule can be solvent, reagent or by-product. Mixed cases require human review; no dominant solvent is chosen automatically. An ALPB name is only a proposed future model setting. Entries without a declared ALPB mapping remain flagged even when the nominal-context status is assumed; no substitute solvent is selected.', '',
        'Assignments depend only on the input and complete saved candidate structures. The reference target, correctness, sampling frequencies, rarity and energies do not enter the assignment function. Counts/ranks/strings are preserved through source and pool hashes, not used to select a solvent. Synthetic tests vary target/correctness/count fields to check this boundary.', '',
        'All proposed assignments and unresolved statuses are frozen in `assignments.jsonl`, with source, dictionary, code and output hashes in `manifest.json`. Review resolutions must be recorded in a new version before any physical run, retaining this original. Every candidate of a later selected row must use its one frozen row-level solvent; unresolved rows are unsupported.', '',
        'The earlier failure-pilot preparation and balanced feasibility proposals are not executed by this deliverable. Case selection and physical execution remain deferred. This table covers all validation rows so solvent assignment does not depend on reference-recovery status.', '',
        'Exact reproduction command from the repository root (use a fresh directory):', '',
        '```bash', '.venv-analysis/bin/python scripts/prepare_solvent_review.py --out results/reranking/solvent-review-REPRODUCTION',
        '.venv-analysis/bin/python -m unittest discover -s tests -p test_solvent_review.py', '```', '',
        'ALPB solvent names are provisional mappings drawn from the [official solvent list](https://xtb-docs.readthedocs.io/en/latest/gbsa.html#parameterized-solvents); installed-version support and the physical protocol still require verification before any authorized calculation.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    outputs = ['dictionary.json','assignments.jsonl','review-required.jsonl','summary.json','review.html','REPORT.md']
    write_json(out/'manifest.json', {'status': 'frozen_recognition_and_review_only', 'source_sha256': sources_before,
        'code_sha256': {str(Path(__file__).relative_to(ROOT)): code_before}, 'dictionary_sha256': dictionary_before,
        'rdkit_version': rdBase.rdkitVersion, 'output_sha256': {p:sha(out/p) for p in outputs},
        'decision_inputs': ['raw input molecular mixture', 'all saved candidate molecular mixtures', 'curated dictionary'],
        'execution_holds': {k:True for k in ['physical_calculations','training','new_inference','cloud_execution','reserved_test_outcomes']}})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
