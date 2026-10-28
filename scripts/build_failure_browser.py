#!/usr/bin/env python3
"""Build a self-contained searchable browser for every isolated failure case."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rdkit import Chem,RDLogger
from rdkit.Chem import Draw
from rerank_analysis import PARSER

RDLogger.DisableLog('rdApp.*')
LABELS={'top1_is_other_recorded_target':'Other recorded target',
        'same_heavy_bonds':'Same heavy bonds: H/charge/radical differences',
        'bond_order_change_same_connectivity':'Bond-order/aromaticity difference',
        'different_heavy_connectivity':'Different heavy connectivity',
        'top1_no_op':'Top 1 predicts no change','target_is_termination':'Reference is termination',
        'tied_for_highest_frequency':'Frequency tie','frequency_gap_le4':'Top 1–target gap ≤4'}


def changed_fragments(left,right):
    def fragments(s):
        mol=Chem.MolFromSmiles(s,PARSER)
        return Counter(Chem.MolToSmiles(f,isomericSmiles=False,allHsExplicit=True)
                       for f in Chem.GetMolFrags(mol,asMols=True))
    a,b=fragments(left),fragments(right);shared=a&b
    a-=shared;b-=shared
    if not a or not b: return None
    mols=[]
    for group in [a,b]:
        mol=Chem.MolFromSmiles('.'.join(k for k,n in group.items() for _ in range(n)),PARSER)
        mols.append(Chem.RemoveHs(mol))
    return mols,sum(shared.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);a=parser.parse_args()
    rows=[json.loads(line) for line in (a.study/'top1_wrong_top10_correct.jsonl').open()]
    summary=json.loads((a.study/'summary.json').read_text())
    compact=[]
    for row in rows:
        compact.append({'line':row['source_row_id']+1,'pathway':row['raw_sequence_id'],'rank':row['target_rank'],
                        'top_count':row['candidates'][0]['count'],
                        'target_count':next(c['count'] for c in row['candidates'] if c['correct']),
                        'flags':[key for key,value in row['diagnostic_flags'].items() if value and key in LABELS],
                        'reactant':row['reactant'],'target':row['target'],
                        'candidates':[{'smiles':c['smiles'],'count':c['count'],'correct':c['correct']} for c in row['candidates']]})
    data=json.dumps(compact,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    labels=json.dumps(LABELS).replace('<','\\u003c')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlowER ranking failures</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#192838;background:#f5f7fa;margin:0}main{max-width:1400px;margin:auto;padding:28px}
h1{margin:0}p{max-width:980px}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat,article{background:white;border:1px solid #dbe1e8;border-radius:8px;padding:16px}.stat strong{font-size:24px;display:block}
.filters{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}input,select,button{font:inherit;padding:8px;border:1px solid #b9c5d3;border-radius:5px}input{flex:1;min-width:260px}
.grid{display:grid;grid-template-columns:minmax(420px,1fr) minmax(420px,1fr);gap:20px}.list{max-height:650px;overflow:auto;background:white;border:1px solid #dbe1e8}table{border-collapse:collapse;width:100%}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #e6ebf0}th{background:#e8eff7;position:sticky;top:0}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.selected{background:#eef5ff}.smiles{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;background:#f1f4f8;padding:10px;border-radius:5px}h2{font-size:19px}h3{font-size:15px;margin-bottom:5px}.tag{display:inline-block;font-size:12px;background:#e8eff7;margin:3px;padding:3px 6px;border-radius:4px}.good{border-left:4px solid #17805a;padding-left:10px}.muted{color:#596a7c}.pagination{display:flex;gap:12px;align-items:center;margin-top:12px}img{max-width:100%}@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:16px}}
</style><main><h1>FlowER: top 1 wrong, target within top 10</h1>
<p>All 2,724 isolated elementary-step cases from the preserved 20,152-row prefix. A mismatch means disagreement with the recorded next state, not proof that the predicted branch is chemically impossible. Pattern labels overlap.</p>
<div class="stats"><div class="stat"><strong>2,724</strong>isolated failures</div><div class="stat"><strong>2,381</strong>top 1 is another observed target</div><div class="stat"><strong>1,814</strong>same heavy-atom bonds</div><div class="stat"><strong>86</strong>tied for highest frequency</div></div>
<div class="filters"><input id="search" placeholder="Search dataset line, pathway ID, or SMILES"><select id="pattern"><option value="">All patterns</option></select><select id="rank"><option value="">All target ranks</option></select></div>
<div class="grid"><section><div class="list"><table><thead><tr><th>Test line</th><th>Pathway</th><th>Target rank</th><th>Top 1 / target counts</th></tr></thead><tbody id="rows"></tbody></table></div><div class="pagination"><button id="prev">Previous</button><span id="count"></span><button id="next">Next</button></div></section><article id="detail"><p>Select a case to inspect its full states and candidates.</p></article></div>
<h2>Illustrative structures</h2><p class="muted">These examples were selected for readable changed fragments. Identical fragments shared by the two outcomes are omitted only in the illustration; full states and every candidate are retained above and in the JSONL export.</p><img src="illustrative-cases.png" alt="Side-by-side predicted top-1 and recorded next-state fragments">
<p><a href="top1_wrong_top10_correct.jsonl">Download every isolated case (JSONL)</a> · <a href="summary.json">Diagnostic summary</a></p>
<p class="muted">Source lines are 1-based positions in the original test file. Candidate counts are out of 64 draws. Ranking preserves original order for ties. This is an ordered diagnostic prefix, not an untouched evaluation or a complete-pathway benchmark.</p></main>
<script id="case-data" type="application/json">__DATA__</script><script>
const cases=JSON.parse(document.getElementById('case-data').textContent),labels=__LABELS__;
const pattern=document.getElementById('pattern'),rank=document.getElementById('rank'),search=document.getElementById('search');
for(const [value,label] of Object.entries(labels)){const o=document.createElement('option');o.value=value;o.textContent=label;pattern.append(o)}
for(let i=2;i<=10;i++){const o=document.createElement('option');o.value=i;o.textContent='Target rank '+i;rank.append(o)}
let filtered=cases,page=0,selected=null;const size=60;
function element(tag,text,cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e}
function detail(row){selected=row.line;const d=document.getElementById('detail');d.replaceChildren(element('h2','Test line '+row.line+' · pathway '+row.pathway));
for(const flag of row.flags)d.append(element('span',labels[flag],'tag'));
for(const [title,smi] of [['Supplied input state',row.reactant],['Recorded next state (rank '+row.rank+')',row.target]]){d.append(element('h3',title),element('div',smi,'smiles'))}
d.append(element('h3','Saved frequency-ranked candidates'));
row.candidates.forEach((c,i)=>{const box=document.createElement('div');if(c.correct)box.className='good';box.append(element('h3','#'+(i+1)+' · '+c.count+'/64'+(c.correct?' · recorded target':'')),element('div',c.smiles,'smiles'));d.append(box)});render()}
function render(){const body=document.getElementById('rows');body.replaceChildren();for(const row of filtered.slice(page*size,(page+1)*size)){const tr=document.createElement('tr');if(row.line===selected)tr.className='selected';for(const v of [row.line,row.pathway,row.rank,row.top_count+' / '+row.target_count])tr.append(element('td',String(v)));tr.onclick=()=>detail(row);body.append(tr)}
document.getElementById('count').textContent=filtered.length+' cases · page '+(page+1)+' / '+Math.max(1,Math.ceil(filtered.length/size));document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=(page+1)*size>=filtered.length}
function filter(){const q=search.value.trim().toLowerCase();filtered=cases.filter(r=>(!pattern.value||r.flags.includes(pattern.value))&&(!rank.value||r.rank===Number(rank.value))&&(!q||[r.line,r.pathway,r.reactant,r.target,...r.candidates.map(c=>c.smiles)].join(' ').toLowerCase().includes(q)));page=0;render()}
for(const e of [pattern,rank,search])e.addEventListener('input',filter);document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();
</script></html>'''.replace('__DATA__',data).replace('__LABELS__',labels)
    (a.study/'cases.html').write_text(html)
    illustration=[];used=set()
    criteria=[('same_heavy_bonds','H / charge placement'),('bond_order_change_same_connectivity','Bond order / aromaticity'),
              ('different_heavy_connectivity','Different connectivity'),('top1_no_op','No-change prediction')]
    for key,label in criteria:
        candidates=[]
        for row in rows:
            if not row['diagnostic_flags'].get(key) or row['row_id'] in used: continue
            diff=changed_fragments(row['candidates'][0]['smiles'],row['target'])
            if diff:
                mols,omitted=diff;heavy=sum(m.GetNumHeavyAtoms() for m in mols)
                if 3<=heavy<=55:candidates.append((hashlib.sha256(str(row['row_id']).encode()).hexdigest(),row,mols,omitted))
        if not candidates: continue
        _,row,mols,omitted=min(candidates,key=lambda item:item[0]);used.add(row['row_id'])
        correct=next(c for c in row['candidates'] if c['correct'])
        illustration.append({'line':row['row_id']+1,'label':label,'omitted_common_fragments':omitted,
                             'top1_count':row['candidates'][0]['count'],'target_count':correct['count'],'target_rank':row['target_rank']})
        row['_illustration_mols']=mols
    mols=[];legends=[]
    byline={r['row_id']+1:r for r in rows}
    for info in illustration:
        mols.extend(byline[info['line']]['_illustration_mols'])
        legends.extend([f"Line {info['line']} | {info['label']}\nTop 1: {info['top1_count']}/64",f"Recorded target | rank {info['target_rank']}\n{info['target_count']}/64; {info['omitted_common_fragments']} common fragments omitted"])
    if mols:
        options=Draw.MolDrawOptions();options.legendFontSize=17;options.bondLineWidth=2
        img=Draw.MolsToGridImage(mols,molsPerRow=2,subImgSize=(600,330),legends=legends,drawOptions=options)
        img.save(str(a.study/'illustrative-cases.png'))
    (a.study/'illustrative-cases.json').write_text(json.dumps(illustration,indent=2))
    print(json.dumps({'isolated_cases':len(rows),'browser':str(a.study/'cases.html'),'illustrations':illustration},indent=2))


if __name__=='__main__':main()
