#!/usr/bin/env python3
"""Replay saved held-out decisions and audit every combined-development fit."""
import argparse
import json
from pathlib import Path
import re

import train_combined_residual as experiment
import learned_residual_model as model
import train_learned_residual as prior
import numpy as np
from scipy import sparse
from phase0_reranking_audit import digest, save


def verify(out):
    output=out/'integrity-verification.json'
    if output.exists():raise FileExistsError('Preserve the existing verification receipt')
    protocol=json.loads((out/'protocol.json').read_text())
    prep=json.loads((out/'preparation.json').read_text())
    frozen=json.loads((out/'model/frozen.json').read_text())
    completion=json.loads((out/'completion.json').read_text())
    experiment.check_sources(protocol)
    for name,h in prep['artifact_sha256'].items():
        if digest(out/name)!=h:raise ValueError('Preparation artifact changed: '+name)
    if digest(out/'model/frozen.json')!=completion['model_receipt_sha256']:
        raise ValueError('Frozen receipt changed')
    if digest(out/'development-summary.json')!=completion['summary_sha256']:
        raise ValueError('Development result changed')
    if digest(out/'model/weights.npz')!=frozen['model_sha256']:
        raise ValueError('Frozen weights changed')
    rows=[json.loads(l) for l in (out/'development-rows.jsonl').read_text().splitlines()]
    X=sparse.load_npz(out/'features.npz');offsets=np.load(out/'offsets.npy',allow_pickle=False)
    support=[r['supported'] for r in json.loads((out/'feature-support.json').read_text())]
    all_ids=list(range(len(rows)));fitted_models={};convergence=[]
    for receipt_path in sorted((out/'fits').glob('*.json')):
        name=receipt_path.stem;receipt=json.loads(receipt_path.read_text())
        match=re.fullmatch(r'outer([0-4])-inner-fold([0-3])-(uniform|bounded_failure)-l2_[0-9.]+',name)
        if match:
            outer,inner=map(int,match.group(1,2))
            partition=[i for i,r in enumerate(rows) if r['fold']!=outer]
            assignments=model.inner_folds(rows,partition,experiment.SEED+outer)
            ids=[i for i in partition if assignments[i]!=inner]
            model.assert_disjoint(rows,ids,[i for i in partition if assignments[i]==inner])
        elif match:=re.fullmatch(r'outer([0-4])-refit',name):
            fold=int(match[1]);ids=[i for i,r in enumerate(rows) if r['fold']!=fold]
            model.assert_disjoint(rows,ids,[i for i,r in enumerate(rows) if r['fold']==fold])
        elif match:=re.fullmatch(r'final-development-cv-fold([0-4])-(uniform|bounded_failure)-l2_[0-9.]+',name):
            fold=int(match[1]);ids=[i for i,r in enumerate(rows) if r['fold']!=fold]
            model.assert_disjoint(rows,ids,[i for i,r in enumerate(rows) if r['fold']==fold])
        elif name=='final-all-development':ids=all_ids
        else:raise ValueError('Unexpected fit receipt: '+name)
        if receipt['fitting_source_row_ids']!=[rows[i]['row_uid'] for i in ids]:
            raise ValueError('Fitting partition differs: '+name)
        info=receipt['info']
        if info['catalog_sha256']!=model.digest(model.catalog(rows,ids)):
            raise ValueError('Label catalog differs from fitting partition: '+name)
        if not info['initialized_from_zero']:raise ValueError('Reused earlier weights: '+name)
        binding=model.digest({'rows':[(rows[i]['row_uid'],rows[i]['target'],rows[i]['normalized_input']) for i in ids],
            'l2':info['l2'],'weighting':info['weighting'],
            'features':prep['artifact_sha256']['features.npz'],'code':protocol['code_sha256']})
        fitted=prior.load_checkpoint(out/'fits'/name,binding)
        convergence.append(bool(fitted['converged']))
        if name.endswith('-refit') or name=='final-all-development':fitted_models[name]=fitted
    if len(convergence)!=206:raise ValueError('Expected all 206 fit receipts')
    replay=[]
    for fold in range(5):
        fitting=[i for i,r in enumerate(rows) if r['fold']!=fold]
        held=[i for i,r in enumerate(rows) if r['fold']==fold]
        selection=json.loads((out/f'outer{fold}-inner-selection.json').read_text())
        if selection['selection_row_uids']!=[rows[i]['row_uid'] for i in fitting]:
            raise ValueError('Outer labels entered policy selection')
        result=json.loads((out/f'outer{fold}-result.json').read_text())
        if result['config']!=selection['selected']['config']:
            raise ValueError('Policy differs from inner selection')
        scores=model.corrections(fitted_models[f'outer{fold}-refit'],X,offsets,held)
        values=experiment.enriched_outcomes(rows,held,model.choices(rows,held,scores,support,result['config']),support)
        saved=[json.loads(l) for l in (out/f'outer{fold}-held-out.jsonl').read_text().splitlines()]
        if saved!=values:raise ValueError('Held-out replay differs')
        replay.extend(values)
    by_uid={r['row_uid']:r for r in replay}
    if len(replay)!=len(rows) or len(by_uid)!=len(rows):raise ValueError('Repeated or omitted assessment rows')
    replay=[by_uid[r['row_uid']] for r in rows]
    saved=[json.loads(l) for l in (out/'nested-oof-decisions.jsonl').read_text().splitlines()]
    if replay!=saved:raise ValueError('Combined decisions differ')
    summary=json.loads((out/'development-summary.json').read_text())
    if experiment.summarize(replay)!=summary['combined']:raise ValueError('Primary statistics differ')
    for cohort,value in summary['by_cohort'].items():
        if experiment.summarize([r for r in replay if r['cohort']==cohort])!=value:
            raise ValueError('Source-specific statistics differ')
    selected=json.loads((out/'final-development-cv-selection.json').read_text())
    if selected['selected']['config']!=frozen['configuration']:
        raise ValueError('Frozen policy differs from final development selection')
    with np.load(out/'model/weights.npz',allow_pickle=False) as weights:
        for key in ('weights','scale'):
            np.testing.assert_array_equal(weights[key],fitted_models['final-all-development'][key])
    catalog=model.catalog(rows,all_ids)
    if catalog!=json.loads((out/'model/label-catalog.json').read_text()):
        raise ValueError('Final catalog differs')
    if digest(out/'model/label-catalog.json')!=frozen['label_catalog_sha256']:
        raise ValueError('Frozen catalog hash differs')
    data=model.fit_inputs(rows,X,offsets,support,all_ids,frozen['configuration']['weighting'])
    np.testing.assert_array_equal(data['scale'],fitted_models['final-all-development']['scale'])
    experiment.check_sources(protocol)
    receipt={'status':'passed','fit_partitions_verified':len(convergence),
        'all_fits_converged':all(convergence),'all_fits_started_from_zero':True,
        'all_fitting_catalogs_reconstructed':True,'held_out_predictions_replayed':len(replay),
        'primary_and_source_statistics_reproduced':True,'final_scaler_reconstructed':True,
        'frozen_weights_match_final_fit':True,'previous_frozen_model_and_sources_unchanged':True,
        'reserved_test_outcomes_or_candidates_opened':False,
        'audit_script_sha256':digest(Path(__file__))}
    save(output,receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    verify(parser.parse_args().out.resolve())
