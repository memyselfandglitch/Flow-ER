#!/usr/bin/env python3
"""Record descriptive atom-level checks for the frozen review (no filtering)."""
import json
from pathlib import Path
from rdkit import Chem
from rerank_analysis import PARSER

base = Path('results/stratified-review-50')
rows = []
for path in sorted((base/'dossiers').glob('*.json')):
    d = json.loads(path.read_text())
    item = {'review_id':d['review_id'], 'test_line':d['test_line'], 'states':{}}
    for role in ('input', 'top1', 'reference'):
        if not d[role]:
            continue
        mol = Chem.MolFromSmiles(d[role], PARSER)
        # Raw aromatic 1.5 bond sums are NOT valence diagnostics. Assign a
        # Kekule form before inspecting integer bond orders (indices retained).
        Chem.Kekulize(mol, clearAromaticFlags=True)
        checks = []
        for atom in mol.GetAtoms():
            bond_sum = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
            if abs(atom.GetFormalCharge()) >= 2 or (atom.GetAtomicNum() == 6 and bond_sum > 4):
                checks.append({
                    'atom_index_zero_based':atom.GetIdx(), 'element':atom.GetSymbol(),
                    'formal_charge':atom.GetFormalCharge(),
                    'radical_electrons':atom.GetNumRadicalElectrons(),
                    'explicit_bond_count':atom.GetDegree(), 'kekule_bond_order_sum':bond_sum,
                    'neighbors':[n.GetSymbol() for n in atom.GetNeighbors()],
                })
        item['states'][role] = {
            'parseable':True,
            'total_radical_electrons':sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
            'total_formal_charge':sum(a.GetFormalCharge() for a in mol.GetAtoms()),
            'atoms_for_inspection':checks,
        }
    rows.append(item)
output = {
    'purpose':'Descriptive graph audit, not a physical stability test or automatic rejection rule',
    'criteria':'Absolute atom formal charge >=2 or carbon integer Kekule bond-order sum >4',
    'atom_indices':'Local to each parsed state; not cross-state mapping',
    'aromaticity':'Kekulize(clearAromaticFlags=True) before bond-order sums; aromatic 1.5 sums are not valence',
    'cases':rows,
}
(base/'electronic-checks.json').write_text(json.dumps(output, indent=2)+'\n')
for r in rows:
    checks = r['states'].get('top1', {}).get('atoms_for_inspection', [])
    if checks:
        print(json.dumps({'review_id':r['review_id'], 'top1_atoms':checks}))
