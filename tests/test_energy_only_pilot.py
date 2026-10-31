"""Diagnostic selection/ranking/physical-boundary tests; no xTB calculation."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import energy_only_pilot as pilot
import score_xtb_candidates as xtb
from test_xtb_pilot import ImmediateExecutor


def sample_case():
    return {'target_rank': 2, 'target': 'A', 'candidates': [
        {'smiles': 'B', 'rank': 1, 'count': 63}, {'smiles': 'A', 'rank': 2, 'count': 1}],
        'recorded_alternatives': [{'smiles': 'A'}, {'smiles': 'B'}]}


def scores(a=-1., b=-2.):
    return [{'smiles': s, 'rank': i+1, 'status': 'ok', 'energy_hartree': e,
             'computation_protocol_sha256': 'one-frozen-protocol',
             'first_conformer_energy_hartree': e, 'conformer_spread_kcal_mol': 0.}
            for i, (s, e) in enumerate([('B', a), ('A', b)])]


class SelectionTests(unittest.TestCase):
    def test_attached_proton_and_consistent_charge_verified_without_atom_maps(self):
        relation = pilot.proton_relation('CN.Cl', 'C[NH3+].[Cl-]')
        self.assertEqual(relation['status'], 'verified_graph_proton_relocation')
        self.assertEqual(sorted(e['delta_H'] for e in relation['example_mapping_edits']), [-1, 1])

    def test_heavy_graph_equality_alone_is_not_proton_transfer(self):
        self.assertNotEqual(pilot.proton_relation('CCO', 'C[CH-][OH2+]')['status'],
                            'verified_graph_proton_relocation')
        self.assertNotEqual(pilot.proton_relation('C=O', '[CH2+][O-]')['status'],
                            'verified_graph_proton_relocation')

    def test_no_change_and_site_choice_are_distinguished(self):
        self.assertEqual(pilot.proton_relation('CO', 'OC')['status'], 'no_change')
        row = {'reactant': 'CN.CO.Cl', 'target': 'C[NH3+].CO.[Cl-]',
               'candidates': [{'smiles': 'CN.C[OH2+].[Cl-]'}]}
        eligible, evidence = pilot.chemical_eligibility(row)
        self.assertTrue(eligible)
        self.assertEqual(evidence['input_to_frequency_winner']['status'], 'verified_graph_proton_relocation')

    def test_solvent_requires_declared_rule_and_no_implicit_water_or_mixture(self):
        row = {'reactant': 'CN.Cl.C1CCOC1', 'candidates': [{'smiles': 'C[NH3+].[Cl-].C1CCOC1'}]}
        self.assertIsNone(pilot.solvent_context(row, 'documented')['solvent'])
        self.assertEqual(pilot.solvent_context(row, 'sole-organic-spectator')['solvent'], 'thf')
        row['reactant'] += '.O'; row['candidates'][0]['smiles'] += '.O'
        self.assertIsNone(pilot.solvent_context(row, 'sole-organic-spectator')['solvent'])
        row = {'reactant': 'CN.Cl.O', 'candidates': [{'smiles': 'C[NH3+].[Cl-].O'}]}
        self.assertIsNone(pilot.solvent_context(row, 'sole-organic-spectator')['solvent'])

    def test_global_group_uniqueness_and_seeded_reproducibility(self):
        eligible = [{'stratum': s, 'group': str(i), 'source_row_id': 100*j+i}
            for j, s in enumerate(pilot.STRATA) for i in range(80)]
        a = pilot.deterministic_sample(eligible); b = pilot.deterministic_sample(list(reversed(eligible)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 50)
        self.assertEqual(len({r['group'] for r in a}), 50)
        self.assertEqual([sum(r['stratum']==s for r in a) for s in pilot.STRATA], [25, 25])

    def test_shortage_does_not_repeat_groups_or_fill_with_ineligible_rows(self):
        rows = [{'stratum': s, 'group': 'shared', 'source_row_id': i} for i, s in enumerate(pilot.STRATA)]
        self.assertEqual(len(pilot.deterministic_sample(rows)), 1)

    def test_run_is_held_before_reading_or_writing_any_data(self):
        with tempfile.TemporaryDirectory() as name:
            run = subprocess.run([sys.executable, str(pilot.ROOT/'scripts/energy_only_pilot.py'),
                'run', '--out', name], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('Physical execution is held', run.stderr)
            self.assertEqual(list(Path(name).iterdir()), [])


class RankingTests(unittest.TestCase):
    def test_energy_only_ignores_counts_and_common_reference(self):
        case = sample_case(); a = pilot.energy_ranking(case, scores())
        case['candidates'][0]['count'] = 33; case['candidates'][1]['count'] = 31
        b = pilot.energy_ranking(case, scores(-101., -102.))
        self.assertEqual(a['chosen_smiles'], 'A'); self.assertEqual(b['chosen_smiles'], 'A')
        self.assertEqual(a['reference_rank_after'], 1)
        self.assertAlmostEqual(a['candidates'][1]['delta_from_frequency_kcal_mol'], -pilot.HARTREE_KCAL)

    def test_exact_tie_uses_candidate_string_not_frequency(self):
        r = pilot.energy_ranking(sample_case(), scores(-1., -1.))
        self.assertEqual(r['chosen_smiles'], 'A')
        self.assertTrue(r['reference_shared_minimum'])
        self.assertEqual([c['shared_energy_rank'] for c in r['candidates']], [1, 1])

    def test_near_tie_is_annotation_not_a_switch_threshold(self):
        r = pilot.energy_ranking(sample_case(), scores(-1., -1.0001))
        self.assertTrue(r['recovered']); self.assertEqual(r['near_tie_original_ranks'], [1, 2])

    def test_any_missing_failed_or_reordered_candidate_unresolves_whole_row(self):
        for values in (scores()[:1], list(reversed(scores())), scores(float('nan'), -2.)):
            r = pilot.energy_ranking(sample_case(), values)
            self.assertEqual(r['status'], 'unresolved'); self.assertFalse(r['recovered'])
            self.assertIsNone(r['reference_rank_after']); self.assertEqual(r['operational_rank'], 1)
        values = scores(); values[0].update(status='failed', reason='topology_changed')
        self.assertEqual(pilot.energy_ranking(sample_case(), values)['status'], 'unresolved')

    def test_conformer_sensitivity_and_recorded_alternative_reported(self):
        values = scores(-2., -1.); values[0]['first_conformer_energy_hartree'] = 0.
        r = pilot.energy_ranking(sample_case(), values)
        self.assertTrue(r['chosen_is_other_recorded_outcome'])
        self.assertTrue(r['conformer_sensitivity']['winner_changes'])

    def test_input_pool_is_not_mutated(self):
        case, values = sample_case(), scores(); before = copy.deepcopy((case, values))
        pilot.energy_ranking(case, values); self.assertEqual((case, values), before)

    def test_mixed_or_missing_physical_protocol_makes_whole_row_unresolved(self):
        values = scores(); values[1]['computation_protocol_sha256'] = 'different-solvent'
        self.assertEqual(pilot.energy_ranking(sample_case(), values)['status'], 'unresolved')
        del values[1]['computation_protocol_sha256']
        self.assertEqual(pilot.energy_ranking(sample_case(), values)['status'], 'unresolved')


class AdapterControlTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(xtb='not-invoked', xtb_version='mock', xtb_sha256='mock', solvent='thf',
            reference_state='bar1mol', electronic_temperature=300., conformers=2, timeout=20.,
            max_heavy_atoms=40, require_all_conformers=True, strict_elements=True, jobs=2,
            max_unique_fragments=720, cache='/not-used', complete_pool_required=True, k=10)

    def test_explicit_reference_state_and_temperature_in_command_and_cache(self):
        args = self.args(); cmd = xtb.xtb_command('geom.xyz', -1, args)
        self.assertEqual(cmd[-3:], ['--alpb', 'thf', 'bar1mol'])
        self.assertEqual(cmd[cmd.index('--etemp')+1], '300.0')
        a = xtb.fragment_configuration('O', args); args.reference_state = 'gsolv'
        b = xtb.fragment_configuration('O', args)
        self.assertNotEqual(xtb.cache_path(a, args), xtb.cache_path(b, args))
        args.solvent = 'water'; c = xtb.fragment_configuration('O', args)
        self.assertNotEqual(xtb.cache_path(b, args), xtb.cache_path(c, args))

    def test_budget_exhausted_before_dispatch_retains_all_fragment_failures(self):
        args = self.args(); args.deadline = 1.
        scorer = Mock()
        with patch.object(xtb.time, 'monotonic', return_value=2.):
            result = xtb.run_unique_fragments(['O', 'CO'], args, scorer,
                ImmediateExecutor, progress=lambda message: None)
        scorer.assert_not_called(); self.assertEqual(set(result), {'O', 'CO'})
        self.assertTrue(all(r['reason']=='total_budget_exhausted' for r in result.values()))

    def test_total_job_cap_rejects_before_any_dispatch(self):
        args = self.args(); args.max_unique_fragments = 1
        scorer = Mock()
        with self.assertRaises(ValueError):
            xtb.run_unique_fragments(['O', 'CO'], args, scorer, ImmediateExecutor)
        scorer.assert_not_called()

    def test_truncation_and_transition_metal_spin_are_rejected(self):
        args = self.args(); args.k = 1
        row = {'row_id': 0, 'reactant': 'CO', 'candidates': [
            {'smiles': 'CO', 'rank': 1}, {'smiles': 'C[OH2+]', 'rank': 2}]}
        with self.assertRaises(ValueError): xtb.prepare_candidates([row], args)
        self.assertIsNotNone(xtb.fragment_unsupported('[Fe+2]', args))

    def test_one_structure_changing_conformer_invalidates_fragment_despite_good_minimum(self):
        args = self.args()
        def fake_run(command, **kwargs):
            (Path(kwargs['cwd'])/'xtbopt.xyz').write_text('mock geometry')
            return SimpleNamespace(returncode=0,
                stdout='GEOMETRY OPTIMIZATION CONVERGED\nTOTAL ENERGY -2.000000\nGsolv -0.100000\n')
        source = xtb.Chem.AddHs(xtb.Chem.MolFromSmiles('CO'))
        changed = xtb.Chem.AddHs(xtb.Chem.MolFromSmiles('C=O'))
        with patch.object(xtb.AllChem, 'EmbedMultipleConfs', return_value=[0, 1]), \
             patch.object(xtb.Chem, 'MolToXYZFile', side_effect=lambda mol, path, **k: Path(path).write_text('mock input')), \
             patch.object(xtb.subprocess, 'run', side_effect=fake_run), \
             patch.object(xtb.Chem, 'MolFromXYZFile', side_effect=[source, changed]), \
             patch.object(xtb.rdDetermineBonds, 'DetermineBonds'):
            result = xtb._calculate_fragment(xtb.canonical(xtb.Chem.MolToSmiles(source)), args)
        self.assertEqual(result['status'], 'failed'); self.assertIsNone(result['energy_hartree'])
        self.assertEqual(result['converged_conformers'], 1)
        # The successful calculation already contains solvation in TOTAL ENERGY.
        self.assertEqual(result['conformer_energies_hartree'], [-2.])


if __name__ == '__main__': unittest.main()
