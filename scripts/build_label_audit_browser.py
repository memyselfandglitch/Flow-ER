#!/usr/bin/env python3
"""Browse every strict top-1 mismatch and illustrate named audit examples."""
import argparse
import json
from pathlib import Path
from rdkit.Chem import Draw
from build_failure_browser import changed_fragments

LABELS = {
    'same_heavy_bonds_H_charge_difference': 'Same heavy bonds; H/charge differences',
    'rdkit_resonance_related': 'RDKit resonance-related (unmapped representation)',
    'different_heavy_connectivity': 'Different heavy connectivity',
    'same_connectivity_bond_order_or_H_charge_difference': 'Same connectivity; bond-order/H/charge differences',
    'no_saved_candidate': 'No saved candidate',
    'unsupported_saved_string': 'Unsupported saved string',
    'serialization_only': 'Equal under evaluator normalization'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.audit/'all-top1-errors.jsonl').open()]
    compact = [{k:r[k] for k in ['test_line','sequence_id','target_rank','input','reference','top1','top1_count',
                'observed_top1','exact_input_observed_top1','largest_organic_available','largest_organic_equal',
                'witness_lines','comparison','top1_no_change','reference_no_change']}
               | {'recorded_graph':r.get('recorded_graph'), 'mapped_difference':r.get('mapped_difference'),
                  'candidates':[{k:c[k] for k in ['rank','smiles','count','correct']} for c in r['candidates']]}
               for r in rows]
    payload = json.dumps(compact,separators=(',',':')).replace('<','\\u003c').replace('&','\\u0026')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlowER label audit</title><style>
body{font:15px/1.5 system-ui,sans-serif;background:#f4f6f9;color:#183045;margin:0}main{max-width:1400px;margin:auto;padding:24px}h1{margin-bottom:6px}p{max-width:1050px}a{color:#1254a0}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}input,select,button{font:inherit;border:1px solid #bac7d3;border-radius:5px;padding:8px}input{min-width:260px;flex:1}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}article,.table{background:white;border:1px solid #d5dfe8;padding:14px;border-radius:6px}.table{max-height:650px;overflow:auto;padding:0}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #e0e7ed}th{position:sticky;top:0;background:#e9f0f6}tr:hover{background:#f0f6ff}tbody tr{cursor:pointer}h2{font-size:20px}h3{font-size:15px}
.smiles{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 monospace;background:#f1f4f7;padding:10px}.tag{display:inline-block;background:#e7eef6;padding:3px 7px;margin:3px;border-radius:4px;font-size:12px}
.pager{display:flex;gap:10px;align-items:center;margin-top:12px}.correct{border-left:4px solid #16815b;padding-left:10px}img{max-width:100%}.note{color:#526779}@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:14px}}
</style><main><h1>FlowER: recorded labels and ranking mismatches</h1>
<p>All <strong>2,834</strong> strict top-1 mismatches in the 20,152-step prefix: <strong>2,399</strong> match another recorded target for the same normalized input; <strong>435</strong> do not. This is a retrospective label audit, not an improved model or a new benchmark score.</p>
<p><a href="REPORT.md">Read the audit report</a> · <a href="competing-recorded-targets.jsonl">Competing recorded outcomes</a> · <a href="unmatched-top1.jsonl">435 unmatched cases</a> · <a href="summary.json">Counts</a></p>
<div class="filters"><input id="search" placeholder="Search line, network ID, or SMILES"><select id="cohort"><option value="">All mismatches</option><option value="observed">Other recorded target</option><option value="unmatched">No other recorded target</option><option value="rescuable">Unmatched, reference in top 10</option></select><select id="category"><option value="">All structural categories</option></select><select id="organic"><option value="">All fragment comparisons</option><option value="same">Largest organic fragment agrees</option><option value="different">Largest fragment differs</option><option value="unavailable">Proxy unavailable</option></select></div>
<div class="grid"><section><div class="table"><table><thead><tr><th>Test line</th><th>Network ID</th><th>Reference rank</th><th>Other target?</th></tr></thead><tbody id="rows"></tbody></table></div><div class="pager"><button id="prev">Previous</button><span id="count"></span><button id="next">Next</button></div></section><article id="detail">Select a case to inspect its input, reference, predictions, and supporting reference rows.</article></div>
<h2>Recorded alternatives</h2><p class="note">Illustrations selected for readable contrasts, not to estimate prevalence. Common fragments are omitted only from these figures. Network IDs can contain multiple branches. Recorded graph endpoints are not model rollouts.</p><img src="recorded-alternatives.png" alt="Four pairs of recorded alternative states">
<h2>Cases without an alternative recorded target</h2><p class="note">These disagree with observed labels; absence from the recorded alternatives does not prove chemical impossibility.</p><img src="unmatched-examples.png" alt="Four pairs of unmatched predictions and references">
<p class="note">Source lines are one-based. Largest organic fragment is a proxy, not a verified experimental product. RDKit resonance matching ignores atom maps; three flagged pairs also relocate mapped hydrogen. Every original candidate remains in the saved data.</p>
</main><script id="payload" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),labels=__LABELS__;
const search=document.getElementById('search'),cohort=document.getElementById('cohort'),category=document.getElementById('category'),organic=document.getElementById('organic');
for(const [key,label] of Object.entries(labels)){const o=document.createElement('option');o.value=key;o.textContent=label;category.append(o)}
let filtered=data,page=0;const size=50;
function el(tag,text,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n}
function inspect(r){const box=document.getElementById('detail');box.replaceChildren(el('h2','Test line '+r.test_line+' · '+r.sequence_id));
box.append(el('span',labels[r.comparison.category],'tag'),el('p','Other recorded target: '+(r.observed_top1?'yes':'no')+' · exact reconstructed input match: '+(r.exact_input_observed_top1?'yes':'no')));
if(r.witness_lines.length)box.append(el('p','Reference rows supporting the predicted outcome: '+r.witness_lines.join(', ')));
box.append(el('p','Largest organic fragment: '+(!r.largest_organic_available?'unavailable':r.largest_organic_equal?'agrees':'differs')));
if(r.recorded_graph)box.append(el('p','Recorded branches share a full terminal state: '+(r.recorded_graph.common_full_terminal?'yes':'no')+'; share a largest terminal organic fragment: '+(r.recorded_graph.common_largest_terminal_fragment?'yes':'no')));
if(r.comparison.category==='rdkit_resonance_related'&&r.mapped_difference?.changed_heavy_H_counts?.length)box.append(el('p','Caution: mapped H counts change at atoms '+r.mapped_difference.changed_heavy_H_counts.join(', ')+'. Resonance relation is under unmapped molecular equivalence.'));
for(const [title,value] of [['Supplied input',r.input],['Recorded reference',r.reference]])box.append(el('h3',title),el('div',value,'smiles'));
box.append(el('h3','Saved candidates (counts out of 64)'));
for(const c of r.candidates){const d=el('div','',c.correct?'correct':'');d.append(el('h3','#'+c.rank+' · '+c.count+'/64'+(c.correct?' · recorded target':'')),el('div',c.smiles,'smiles'));box.append(d)}
if(!r.candidates.length)box.append(el('p','No saved candidate. This row remains in every denominator.'))}
function render(){const body=document.getElementById('rows');body.replaceChildren();for(const r of filtered.slice(page*size,(page+1)*size)){const tr=document.createElement('tr');for(const v of [r.test_line,r.sequence_id,r.target_rank??'absent',r.observed_top1?'yes':'no'])tr.append(el('td',String(v)));tr.onclick=()=>inspect(r);body.append(tr)}
document.getElementById('count').textContent=filtered.length+' cases · page '+(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/size));document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=(page+1)*size>=filtered.length}
function filter(){const q=search.value.toLowerCase().trim();filtered=data.filter(r=>(!cohort.value||(cohort.value==='observed'?r.observed_top1:cohort.value==='unmatched'?!r.observed_top1:!r.observed_top1&&r.target_rank>1&&r.target_rank<=10))&&(!category.value||r.comparison.category===category.value)&&(!organic.value||(organic.value==='unavailable'?!r.largest_organic_available:organic.value==='same'?r.largest_organic_equal:r.largest_organic_available&&!r.largest_organic_equal))&&(!q||[r.test_line,r.sequence_id,r.input,r.reference,r.top1].join(' ').toLowerCase().includes(q)));page=0;render()}
for(const e of [search,cohort,category,organic])e.addEventListener('input',filter);document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();
</script></html>'''.replace('__DATA__',payload).replace('__LABELS__',json.dumps(LABELS))
    (args.audit/'cases.html').write_text(html)
    byline = {r['test_line']:r for r in rows}
    groups = {
        'recorded-alternatives': [(8,'Same organic structure; different acid/base partners'),
            (36,'Recorded resonance-related intermediates'),(86,'Bond formation versus proton removal first'),
            (4077,'Different hydrogen/charge placement')],
        'unmatched-examples': [(224,'Unmatched hydrogen/charge state'),(27,'Different reacting nitrogen'),
            (3959,'Different ring attack site'),(5221,'Premature no-change prediction')]}
    metadata = []
    for name, examples in groups.items():
        mols, legends = [], []
        for line, description in examples:
            r = byline[line]; (left,right),omitted = changed_fragments(r['top1'],r['reference'])
            mols.extend([left,right])
            legends += [f'Line {line} | Top 1 ({r["top1_count"]}/64)\n{description}',
                        f'Recorded target | rank {r["target_rank"]}\n{omitted} common fragments omitted']
            metadata.append({'line':line,'description':description,'figure':name+'.png',
                             'common_fragments_omitted':omitted,'witness_lines':r['witness_lines']})
        options = Draw.MolDrawOptions(); options.legendFontSize=18; options.bondLineWidth=2
        Draw.MolsToGridImage(mols,molsPerRow=2,subImgSize=(700,400),legends=legends,drawOptions=options).save(str(args.audit/(name+'.png')))
    (args.audit/'illustration-index.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'browser_cases':len(rows),'illustrations':len(metadata)}))


if __name__ == '__main__':
    main()
