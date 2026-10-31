#!/usr/bin/env python3
"""Validate and assemble the frozen 50-case qualitative review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

DIAGNOSES = {
    'premature_stopping': 'No change before the recorded next step',
    'competing_site_or_path': 'Competing reaction site or pathway',
    'protonation_or_charge_choice': 'Protonation or charge choice',
    'step_granularity_or_representation': 'Step granularity or representation',
    'sampling_or_conversion_failure': 'Sampling or conversion failure',
    'unresolved': 'Unresolved',
}
ASSESSMENTS = {
    'reference_preference_supported': 'Structural argument supports reference preference',
    'context_dependent_alternatives': 'Context-dependent alternatives',
    'reference_needs_provenance_review': 'Reference needs provenance review',
    'not_assessable': 'Not assessable',
}
ACTIONS = {
    'test_input_candidate_signal': 'Test a specific input/candidate signal',
    'no_reliable_selector': 'No reliable selector established',
    'reference_outside_top10': 'Reference outside top 10',
    'no_saved_candidate': 'No saved candidate',
}
TEXT_FIELDS = [
    'observed_difference', 'reference_step_interpretation', 'assessment_reason',
    'reference_support', 'uncertainty', 'proposed_inference_time_signal',
    'suggested_remedy', 'figure_inspected',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=Path('results/stratified-review-50'))
    base = parser.parse_args().review
    selection = json.loads((base/'selection.json').read_text())
    dossiers = [json.loads(p.read_text()) for p in sorted((base/'dossiers').glob('*.json'))]
    reviews = []
    for name in ('a', 'b', 'c'):
        reviews.extend(json.loads((base/'reviews'/f'reviewer_{name}.json').read_text()))
    assert len(dossiers) == len(reviews) == 50
    assert len({r['review_id'] for r in reviews}) == 50
    by_id = {r['review_id']: r for r in reviews}
    combined = []
    for d in dossiers:
        r = by_id[d['review_id']]
        assert r['test_line'] == d['test_line']
        assert r['primary_diagnosis'] in DIAGNOSES, r
        assert r['chemical_assessment'] in ASSESSMENTS, r
        assert r['reranking_action'] in ACTIONS, r
        assert all(isinstance(r[k], str) and r[k].strip() for k in TEXT_FIELDS), r
        assert isinstance(r['evidence'], list) and r['evidence']
        assert r['figure_inspected'].endswith(d['figure']), r
        assert (base/d['figure']).is_file()
        assert len(d['candidates']) <= 10
        assert sum(d['sample_metrics']) == 64
        rank = d['target_rank']
        if r['reranking_action'] in ('test_input_candidate_signal', 'no_reliable_selector'):
            assert rank is not None and 1 < rank <= 10, r
        if r['reranking_action'] == 'reference_outside_top10':
            assert d['candidates'] and (rank is None or rank > 10), r
        if r['reranking_action'] == 'no_saved_candidate':
            assert not d['candidates'], r
        combined.append(dict(d, review=r))
    assert [d['test_line'] for d in combined] == selection['test_lines']
    assert Counter(d['comparison']['category'] for d in combined) == selection['category_quotas']
    reserved = json.loads(Path('results/reranking/pilot-v2/holdout-rows.json').read_text())
    assert not ({d['row_id'] for d in combined} & {r['source_row_id'] for r in reserved})
    assert not ({d['sequence_id'] for d in combined} & {r['sequence_id'] for r in reserved})
    paths = {
        'unmatched': Path('results/label-audit-20152/unmatched-top1.jsonl'),
        'records': Path('results/reranking/failure-study/records.jsonl'),
        'script': Path('scripts/prepare_stratified_review.py'),
    }
    for key, path in paths.items():
        assert sha(path) == selection['source_sha256'][key]
    summary = {
        'status': '50 qualitative reviews completed',
        'scope': 'Stratified diagnostic selection, not prevalence or performance estimation',
        'sample': 50, 'source_population': 435,
        'network_count': len({d['sequence_id'] for d in combined}),
        'category_quotas': selection['category_quotas'],
        'reference_in_top10': sum(d['strict_top10'] for d in combined),
        'reference_below_top10_but_sampled': sum(d['target_rank'] is not None and d['target_rank'] > 10 for d in combined),
        'reference_absent_from_saved_samples': sum(d['target_rank'] is None for d in combined),
        'no_saved_candidate': sum(not d['candidates'] for d in combined),
        'top1_no_change': sum(d['top1_no_change'] for d in combined),
        'reference_no_change': sum(d['reference_no_change'] for d in combined),
        'primary_diagnosis': dict(Counter(d['review']['primary_diagnosis'] for d in combined)),
        'chemical_assessment': dict(Counter(d['review']['chemical_assessment'] for d in combined)),
        'reranking_action': dict(Counter(d['review']['reranking_action'] for d in combined)),
        'review_protocol': 'review-protocol.md',
        'review_mode': 'Three disjoint AI-assisted review partitions, shared project context, root synthesis; not blinded expert validation',
        'reserved_evaluation_rows_accessed': 0,
        'reserved_evaluation_overlap': 0,
        'new_gpu_inference': False, 'reranker_fitted': False, 'energy_calculations': False,
    }
    (base/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (base/'reviewed-cases.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in combined))
    payload = json.dumps(combined, separators=(',', ':')).replace('<', '\\u003c').replace('&', '\\u0026')
    html = TEMPLATE.replace('__PAYLOAD__', payload).replace('__DIAGNOSES__', json.dumps(DIAGNOSES)).replace('__ASSESSMENTS__', json.dumps(ASSESSMENTS)).replace('__ACTIONS__', json.dumps(ACTIONS))
    (base/'review.html').write_text(html)
    (base/'validation.json').write_text(json.dumps({
        'reviews':50, 'dossiers':50, 'figures':100, 'source_hashes_unchanged':True,
        'frozen_selection_matches':True, 'category_quotas_match':True,
        'case_ids_unique':True, 'availability_actions_consistent':True,
        'sample_tallies_sum_to_64':True, 'reserved_row_and_network_overlap':0,
    }, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlowER · 50-case review</title><style>
:root{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#193244;background:#f4f6f8}
*{box-sizing:border-box}body{margin:0}main{max-width:1580px;margin:auto;padding:28px}
h1{font-size:30px;line-height:1.2;margin:0 0 12px}h2{font-size:22px;margin:0 0 10px}h3{font-size:16px;margin:20px 0 5px}
p{margin:8px 0 14px}a{color:#145fa1}header p{max-width:1100px}.eyebrow{font-size:12px;letter-spacing:.09em;color:#526878;text-transform:uppercase}
.metrics{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}.metric{background:white;border:1px solid #d5dee5;border-radius:8px;padding:12px 20px;flex:1;min-width:150px}.metric b{font-size:26px;display:block;color:#125b60}
.filters{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}input,select,button{font:inherit;padding:8px 10px;border:1px solid #b9c7d1;border-radius:5px;background:white;color:inherit}input{min-width:260px;flex:1}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}
.layout{display:grid;grid-template-columns:290px minmax(0,1fr);gap:20px}.case-list,article{background:white;border:1px solid #d5dee5;border-radius:8px}.case-list{height:740px;overflow:auto;position:sticky;top:16px;padding:9px}.case-button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #e7edf1;border-radius:0;padding:12px}.case-button:hover{background:#f2f6f9}.case-button[aria-current=true]{background:#e4f2f1;box-shadow:inset 3px 0 #207b79}.case-button small{display:block;color:#536b7b;font-size:12px;line-height:1.4}
article{padding:22px;min-height:740px}.tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}.tag{font-size:12px;padding:3px 8px;background:#eef3f7;border-radius:4px}.tag.action{background:#e5f1ec;color:#245f46}
figure{margin:18px 0;border:1px solid #e0e6eb;border-radius:6px;overflow:hidden}figure img{display:block;width:100%;height:auto}figcaption{padding:8px 12px;font-size:12px;color:#526878;background:#fafbfd}
.assessment{border-left:3px solid #c99a39;padding:2px 15px;background:#fffaf0;margin:18px 0}.smiles{overflow-wrap:anywhere;white-space:pre-wrap;background:#f5f7f9;border-radius:4px;padding:10px;font:12px/1.5 monospace}
details{border-top:1px solid #e1e7ec;margin-top:16px;padding-top:10px}summary{cursor:pointer;font-weight:600}.pager{display:flex;gap:12px;align-items:center;justify-content:space-between;margin-bottom:14px}.muted{color:#526878}.candidate{border-bottom:1px solid #e1e7ec;padding-bottom:8px}.reference{border-left:3px solid #4a9c7b;padding-left:12px}footer{font-size:13px;color:#526878;margin-top:24px}
@media(max-width:1000px){.layout{grid-template-columns:1fr}.case-list{height:240px;position:static}.metrics{gap:8px}main{padding:16px}article{padding:16px}}
</style></head><body><main>
<header><p class="eyebrow">FlowER · qualitative diagnosis · frozen selection</p><h1>What explains these 50 ranking mismatches?</h1>
<p>Each case compares the supplied input, frequency-ranked top 1, and recorded next state. The sample covers 50 reaction networks from the 435 unmatched errors. It deliberately includes difficult and rare cases; its counts do not estimate rates across the full benchmark.</p>
<p><a href="REPORT.md">Read findings and next steps</a> · <a href="reviewed-cases.jsonl">All review data</a> · <a href="selection.json">Frozen selection</a> · <a href="review-protocol.md">Review criteria</a></p></header>
<div class="metrics"><div class="metric"><b>50</b>cases reviewed</div><div class="metric"><b>31</b>reference in top 10</div><div class="metric"><b>2</b>reference at rank 11</div><div class="metric"><b>17</b>reference absent from saved samples</div><div class="metric"><b>17</b>top 1 predicts no change</div></div>
<div class="filters"><input id="search" aria-label="Search reviews" placeholder="Search case, line, chemical term, or diagnosis">
<select id="diagnosis" aria-label="Filter diagnosis"><option value="">All diagnoses</option></select>
<select id="action" aria-label="Filter next action"><option value="">All reranking assessments</option></select>
<select id="coverage" aria-label="Filter candidate coverage"><option value="">All candidate coverage</option><option value="present">Reference in top 10</option><option value="rank11">Reference at rank 11</option><option value="absent">Reference absent</option></select></div>
<div class="layout"><nav class="case-list" id="case-list" aria-label="Reviewed cases"></nav><article id="detail"></article></div>
<footer>AI-assisted structural review with source evidence and explicit uncertainty; no experimental adjudication, energy calculation, new inference, or learned reranker. Figures omit only fragments common to all available states for readability. Complete molecular systems and saved candidates remain accessible in each case. Source test lines are one-based. Sample frequency is not a calibrated probability. The reserved evaluation subset was not inspected.</footer>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent), diagnoses=__DIAGNOSES__, assessments=__ASSESSMENTS__, actions=__ACTIONS__;
const search=document.getElementById('search'),diagnosis=document.getElementById('diagnosis'),action=document.getElementById('action'),coverage=document.getElementById('coverage'),list=document.getElementById('case-list'),detail=document.getElementById('detail');
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function link(text,path){const a=el('a',text);a.href=path;return a}
for(const [select,labels] of [[diagnosis,diagnoses],[action,actions]])for(const [k,v] of Object.entries(labels)){const o=el('option',v);o.value=k;select.append(o)}
let filtered=data,selected=location.hash.slice(1)||'R01';
function section(title,text){detail.append(el('h3',title),el('p',text))}
function draw(){
  list.replaceChildren();
  for(const d of filtered){const b=el('button',d.review_id+' · line '+d.test_line,'case-button');b.setAttribute('aria-current',String(d.review_id===selected));b.append(el('small',diagnoses[d.review.primary_diagnosis]+' · reference '+(d.target_rank?'#'+d.target_rank:'absent')));b.onclick=()=>{selected=d.review_id;history.replaceState(null,'','#'+selected);draw()};list.append(b)}
  const d=filtered.find(x=>x.review_id===selected);detail.replaceChildren();
  if(!d){detail.append(el('h2','No matching cases'),el('p','Change the filters to inspect another case.'));return}
  const r=d.review,pos=filtered.indexOf(d),pager=el('div',undefined,'pager'),prev=el('button','← Previous'),next=el('button','Next →');
  prev.disabled=pos===0;next.disabled=pos===filtered.length-1;
  for(const [b,offset] of [[prev,-1],[next,1]])b.onclick=()=>{selected=filtered[pos+offset].review_id;history.replaceState(null,'','#'+selected);draw()};
  pager.append(prev,el('span',(pos+1)+' / '+filtered.length+' matching cases','muted'),next);detail.append(pager);
  detail.append(el('h2',d.review_id+' · test line '+d.test_line+' · '+d.sequence_id));
  const tags=el('div',undefined,'tags');tags.append(el('span',diagnoses[r.primary_diagnosis],'tag'),el('span',actions[r.reranking_action],'tag action'),el('span','Top 1: '+d.top1_count+'/64','tag'),el('span','Reference: '+d.reference_count+'/64 · '+(d.target_rank?'rank '+d.target_rank:'not saved'),'tag'));detail.append(tags);
  detail.append(el('p',r.observed_difference));
  const f=el('figure'),a=link('',d.figure),img=el('img');img.src=d.figure;img.alt=d.review_id+': supplied input, predicted top1, and recorded reference';a.append(img);f.append(a,el('figcaption','Click the structures to view full resolution. Common fragments are omitted here; this is a display choice only.'));detail.append(f);
  section('Recorded step',r.reference_step_interpretation);
  const assessment=el('div',undefined,'assessment');assessment.append(el('h3',assessments[r.chemical_assessment]),el('p',r.assessment_reason));detail.append(assessment);
  section('What the recorded network supports',r.reference_support);
  section('What remains uncertain',r.uncertainty);
  section('Possible signal available during inference',r.proposed_inference_time_signal);
  section('Suggested next action',r.suggested_remedy);
  const full=el('details');full.append(el('summary','Complete supplied molecular system'));const fi=el('img');fi.style.width='100%';fi.src='figures/input-'+String(d.test_line).padStart(5,'0')+'.png';fi.alt='All input fragments for '+d.review_id;fi.loading='lazy';full.append(fi);detail.append(full);
  const cand=el('details');cand.append(el('summary','Saved top-10 candidates, counts, and exact strings'));
  if(!d.candidates.length)cand.append(el('p','No candidate was saved. See the review evidence for the evaluator failure category.'));
  for(const c of d.candidates){const block=el('div',undefined,'candidate'+(c.correct?' reference':''));block.append(el('h3','#'+c.rank+' · '+c.count+'/64'+(c.correct?' · recorded reference':'')),el('div',c.smiles,'smiles'));cand.append(block)}
  if(d.target_rank>10)cand.append(el('p','The original writer recorded the reference at rank '+d.target_rank+'. This dossier stores the first 10 candidate strings; the reference is separately retained below.'));
  cand.append(el('h3','Reference string'),el('div',d.reference,'smiles'));detail.append(cand);
  const evidence=el('details');evidence.append(el('summary','Evidence and source dossier'),link('Open complete JSON dossier','dossiers/case-'+String(d.test_line).padStart(5,'0')+'.json'));
  const ul=el('ul');for(const e of r.evidence){const text=typeof e==='string'?e:JSON.stringify(e),li=el('li');if(text.startsWith('https://'))li.append(link(text,text.split(' ')[0]));else li.textContent=text;ul.append(li)}evidence.append(ul);detail.append(evidence);
}
function filter(){const q=search.value.toLowerCase().trim();filtered=data.filter(d=>(!diagnosis.value||d.review.primary_diagnosis===diagnosis.value)&&(!action.value||d.review.reranking_action===action.value)&&(!coverage.value||(coverage.value==='present'?d.target_rank&&d.target_rank<=10:coverage.value==='rank11'?d.target_rank===11:d.target_rank===null))&&(!q||JSON.stringify(d.review).toLowerCase().includes(q)||d.sequence_id.toLowerCase().includes(q)));if(!filtered.some(d=>d.review_id===selected))selected=filtered[0]?.review_id;draw()}
for(const e of [search,diagnosis,action,coverage])e.addEventListener('input',filter);draw();
</script></body></html>'''


if __name__ == '__main__':
    main()
