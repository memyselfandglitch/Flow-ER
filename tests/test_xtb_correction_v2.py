"""Saved-geometry regression and mocked execution tests. No xTB is launched."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import energy_only_pilot as pilot
import prepare_energy_focus_v2 as prep
import score_xtb_candidates_v2 as adapter
import xtb_structure_v2 as structure
from run_energy_focus_v2 import authorize,aggregate
from run_energy_focus import assess

PARENT=prep.PARENT
EXECUTION=json.loads((PARENT/'execution.json').read_text())


def record(prefix):
    paths=list((PARENT/'cache').glob(prefix+'*.json'))
    if len(paths)!=1:raise ValueError('Regression fixture is missing or ambiguous')
    return json.loads(paths[0].read_text())


def fake_output(energy=-1.25,solvent='dmf'):
    return (f'Solvation model: ALPB\nSolvent {solvent}\n'
        'Reference state gsolv [1 bar gas/1 M solution]\n'
        '*** convergence criteria satisfied after 3 iterations ***\n'
        f'| TOTAL ENERGY {energy:.12f} Eh |\nnormal termination of xtb\n')


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.args=prep.settings('dmf',Path('/not/used/cache'))

    def test_monatomic_runs_one_single_point_without_optimizer(self):
        with patch.object(adapter.subprocess,'run',return_value=SimpleNamespace(
                returncode=0,stdout=fake_output())) as run:
            result=adapter.calculate_fragment('[Na+]',self.args)
        self.assertEqual(run.call_count,1)
        cmd=run.call_args.args[0]
        self.assertIn('--sp',cmd);self.assertNotIn('--opt',cmd)
        self.assertEqual(cmd[cmd.index('--chrg')+1],'1')
        self.assertEqual(cmd[cmd.index('--uhf')+1],'0')
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['requested_conformers'],1)

    def test_low_energy_in_failed_monatomic_calculation_is_rejected(self):
        with patch.object(adapter.subprocess,'run',return_value=SimpleNamespace(
                returncode=1,stdout=fake_output(-999999.))) as run:
            result=adapter.calculate_fragment('[Cl-]',self.args)
        self.assertEqual(run.call_count,1);self.assertEqual(result['status'],'failed')
        self.assertIsNone(result['energy_hartree']);self.assertEqual(result['conformer_energies_hartree'],[])

    def test_scc_convergence_and_normal_termination_are_required_for_single_point(self):
        item={'command':adapter.command('atom.xyz',1,True,self.args),'returncode':0,'stdout':fake_output()}
        self.assertTrue(adapter.physical_check(item,1,'dmf',True)['valid'])
        for bad in [fake_output().replace('convergence criteria satisfied','not converged'),
                    fake_output().replace('normal termination','abnormal termination'),
                    fake_output()+'\n[ERROR] failing calculation',fake_output(solvent='thf')]:
            self.assertFalse(adapter.physical_check(dict(item,stdout=bad),1,'dmf',True)['valid'])

    def test_molecular_command_keeps_optimization_and_does_not_use_single_point(self):
        cmd=adapter.command('input.xyz',0,False,self.args)
        self.assertNotIn('--sp',cmd);self.assertEqual(cmd[cmd.index('--opt')+1],'normal')

    def test_hydrogen_molecule_is_checked_and_dissociation_rejected(self):
        xyz='2\nH2\nH 0 0 0\nH 0 0 0.74\n'
        self.assertTrue(structure.check_geometry('[H][H]',xyz,xyz)['valid'])
        result=structure.check_geometry('[H][H]',xyz,xyz.replace('0.74','1.8'))
        self.assertFalse(result['valid']);self.assertEqual(result['removed_bonds'],[[0,1]])

    def test_missing_changed_atoms_and_overlapping_coordinates_are_rejected(self):
        xyz='2\nH2\nH 0 0 0\nH 0 0 0.74\n'
        for bad in ['1\nH\nH 0 0 0\n',xyz.replace('H 0 0 0.74','O 0 0 0.74'),xyz.replace('0.74','0.0')]:
            self.assertFalse(structure.check_geometry('[H][H]',xyz,bad)['valid'])

    def test_saved_resonance_equivalence_does_not_admit_failed_energy(self):
        r=record('538e531660');c=r['conformer_records'][0]
        result=structure.check_geometry(r['smiles'],c['input_xyz'],c['optimized_xyz'])
        self.assertTrue(result['valid']);self.assertEqual(result['reason'],'verified_indexed_resonance_equivalence')
        self.assertFalse(adapter.reuse_decision(r,EXECUTION)['reusable'])

    def test_unproved_sulfur_representation_remains_unresolved(self):
        r=record('2444260aae');c=r['conformer_records'][0]
        result=structure.check_geometry(r['smiles'],c['input_xyz'],c['optimized_xyz'])
        self.assertFalse(result['valid']);self.assertEqual(result['reason'],'bond_order_or_charge_identity_unresolved')

    def test_observed_connectivity_change_still_fails(self):
        found=False
        for f in sorted((PARENT/'cache').glob('*.json')):
            r=json.loads(f.read_text())
            for c in r.get('conformer_records',[]):
                if c.get('reason')!='connectivity_changed_or_perception_failed' or 'optimized_xyz' not in c:continue
                result=structure.check_geometry(r['smiles'],c['input_xyz'],c['optimized_xyz'])
                if result['reason']=='indexed_connectivity_changed_or_ambiguous':
                    self.assertFalse(result['valid']);self.assertTrue(result['added_bonds'] or result['removed_bonds'])
                    found=True;break
            if found:break
        self.assertTrue(found)

    def test_resonance_limit_abstains(self):
        r=record('538e531660');c=r['conformer_records'][0]
        with patch.object(structure,'RESONANCE_LIMIT',1):
            result=structure.check_geometry(r['smiles'],c['input_xyz'],c['optimized_xyz'])
        self.assertFalse(result['valid']);self.assertEqual(result['reason'],'resonance_enumeration_limit')

    def test_successful_cache_revalidated_with_provenance(self):
        r=record('e4c7507695');d=adapter.reuse_decision(r,EXECUTION)
        self.assertTrue(d['reusable'])
        result=adapter.imported_result(r,d,self.args,'original.json','source-hash')
        self.assertEqual(result['energy_hartree'],r['energy_hartree'])
        self.assertEqual(result['reuse']['source_sha256'],'source-hash')
        self.assertNotIn('reuse',r)

    def test_partial_or_tampered_or_incompatible_cache_is_not_imported(self):
        original=record('e4c7507695')
        variants=[]
        r=copy.deepcopy(original);r['status']='failed';variants.append(r)
        r=copy.deepcopy(original);r['conformer_records'].pop();variants.append(r)
        r=copy.deepcopy(original);r['energy_hartree']-=10;variants.append(r)
        r=copy.deepcopy(original);r['computation_protocol']['reference_state']='gsolv';variants.append(r)
        r=copy.deepcopy(original);r['conformer_records'][0]['optimized_xyz']=r['conformer_records'][0]['optimized_xyz'].replace('energy:', 'unverified:');variants.append(r)
        r=copy.deepcopy(original);r['conformer_records'][0]['returncode']=1;variants.append(r)
        for r in variants:
            decision=adapter.reuse_decision(r,EXECUTION)
            self.assertFalse(decision['reusable'])
            with self.assertRaises(ValueError):adapter.imported_result(r,decision,self.args,'source','sha')

    def test_authorization_is_new_and_bound_to_exact_freeze(self):
        for receipt in [{},EXECUTION['authorization'],{'local_physical_execution_authorized':True,
                'manifest_sha256':'another-version','workers':2,'maximum_elapsed_seconds':7200}]:
            with self.assertRaises(ValueError):authorize(receipt,'this-version')
        authorize({'local_physical_execution_authorized':True,'manifest_sha256':'this-version',
                   'workers':2,'maximum_elapsed_seconds':7200},'this-version')

    def test_v2_timeout_terminates_the_full_process_group(self):
        process=Mock(pid=987654)
        process.communicate.side_effect=[subprocess.TimeoutExpired('worker',20),('','')]
        self.args.deadline=float('inf')
        with patch.object(adapter.subprocess,'Popen',return_value=process) as launch, \
                patch.object(adapter.os,'killpg') as kill:
            result=adapter.compute_bounded('[Na+]',self.args)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(987654,adapter.signal.SIGKILL)
        self.assertEqual(result['status'],'failed');self.assertIsNone(result['energy_hartree'])

    def test_preparation_launches_no_physics_and_freeze_detects_changes(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(adapter.subprocess,'run',side_effect=AssertionError('No subprocess allowed')), \
                patch.object(adapter.subprocess,'Popen',side_effect=AssertionError('No subprocess allowed')), \
                contextlib.redirect_stdout(io.StringIO()):
            out=Path(folder)/'prepared'
            prep.prepare(out)
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['physical_execution_held'])
            self.assertEqual(summary['validated_reusable_jobs'],71)
            self.assertEqual(summary['pending_jobs'],40)
            self.assertIsNone(summary['reference_recovery_v2'])
            manifest,cases=prep.verify_freeze(out)
            self.assertEqual(len(cases),18)
            with patch.object(adapter,'protocol',return_value={'version':'incompatible-runtime'}):
                with self.assertRaises(ValueError):prep.verify_freeze(out)
            with (out/'selected-cases.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaises(ValueError):prep.verify_freeze(out)

    def test_fragment_multiplicity_is_preserved(self):
        smi=pilot.fragments('[H][H]')[0];args=self.args
        data={**adapter.configuration(smi,args),'status':'ok','energy_hartree':-1.,
              'conformer_energies_hartree':[-1.,-.9]}
        case={'source_row_id':1,'reactant':'[H][H].[H][H]','solvent':{'solvent':'dmf'},
              'candidates':[{'rank':1,'smiles':'[H][H].[H][H]'}]}
        result=aggregate([case],{prep.job_id('dmf',smi):data},args.cache)
        self.assertEqual(result[0]['energy_hartree'],-2.)
        self.assertEqual(len(result[0]['fragment_provenance']),2)

    def test_failed_candidate_makes_whole_pool_unresolved(self):
        case={'source_row_id':1,'reactant':'CCO','target':'COC','target_rank':2,
            'solvent':{'solvent':'dmf','nominal_name':'DMF'},'recorded_alternatives':[{'smiles':'COC'}],
            'candidates':[{'rank':1,'smiles':'CCO'},{'rank':2,'smiles':'COC'}]}
        results={}
        for rank,smi in enumerate(['CCO','COC']):
            f=pilot.fragments(smi)[0]
            results[prep.job_id('dmf',f)]={**adapter.configuration(f,self.args),
                'status':'ok' if rank else 'failed','energy_hartree':-1. if rank else None,
                'conformer_energies_hartree':[-1.,-1.] if rank else [],'reason':None if rank else 'nonconverged'}
        scores=aggregate([case],results,self.args.cache);rows,summary=assess([case],scores)
        self.assertEqual(summary['complete_pool_rows'],0)
        self.assertEqual(summary['recovery_all_selected'],0)
        self.assertEqual(rows[1]['status'],'unresolved');self.assertIsNone(rows[1]['reference_rank_after'])


if __name__=='__main__':unittest.main()
