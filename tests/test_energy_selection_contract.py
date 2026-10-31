"""Synthetic energy-policy tests; never start xTB, fit a model or load outcomes."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from energy_selection_contract import energy_choice


class EnergySelectionTests(unittest.TestCase):
    def choose(self, counts, energies, lam=1., margin=0., **kwargs):
        flags = {'solvent_resolved': True, 'pool_compatible': True,
                 'reliable': [True] * len(counts)}
        flags.update(kwargs)
        return energy_choice(counts, energies, lam, margin, **flags)

    def test_zero_weight_is_frequency_even_with_no_physical_information(self):
        self.assertEqual(energy_choice([40, 24], None, 0., 0.), (0, 'zero_intervention'))

    def test_energy_can_overcome_frequency_but_higher_energy_does_not_get_rewarded(self):
        self.assertEqual(self.choose([40, 24], [0., -1.]), (1, 'energy_override'))
        self.assertEqual(self.choose([40, 24], [0., 1.]), (0, 'frequency_retained'))

    def test_common_energy_offset_cannot_change_decision(self):
        self.assertEqual(self.choose([40, 24], [0., -1.]),
                         self.choose([40, 24], [-100000., -100001.]))

    def test_override_requires_strict_margin(self):
        self.assertEqual(self.choose([32, 32], [0., -1.], margin=1.), (0, 'frequency_retained'))
        self.assertEqual(self.choose([32, 32], [0., -1.], margin=.99), (1, 'energy_override'))

    def test_stable_ties_and_uniform_energy_leave_frequency_winner(self):
        self.assertEqual(self.choose([32, 32], [-2., -2.])[0], 0)
        self.assertEqual(self.choose([40, 24], [-2., -2.])[0], 0)

    def test_failure_of_any_candidate_forces_whole_pool_fallback(self):
        for energies in ([0., -10., None], [0., -10., float('nan')], [0., -10.], None):
            self.assertEqual(self.choose([40, 20, 4], energies)[0], 0)
        self.assertEqual(self.choose([40, 20, 4], [0., -10., 1.], reliable=[True, True, False]),
                         (0, 'unreliable_candidate'))

    def test_uncertain_solvent_or_incompatible_protocol_never_promotes(self):
        self.assertEqual(self.choose([40, 24], [0., -100.], solvent_resolved=False),
                         (0, 'uncertain_solvent'))
        self.assertEqual(self.choose([40, 24], [0., -100.], pool_compatible=False),
                         (0, 'incompatible_pool'))

    def test_explicit_trust_flags_are_required(self):
        self.assertEqual(energy_choice([40, 24], [0., -100.], 1., 0.)[0], 0)
        self.assertEqual(self.choose([40, 24], [0., -100.], reliable=None)[0], 0)

    def test_invalid_counts_or_parameters_rejected(self):
        for counts in ([24, 40], [40, 25], [0, 1], [1.5], [float('nan')], [True]):
            with self.assertRaises(ValueError):
                self.choose(counts, [0.] * len(counts))
        for lam, margin in ((-1., 0.), (0., -1.), (float('nan'), 0.), (1., float('inf'))):
            with self.assertRaises(ValueError):
                self.choose([40, 24], [0., -1.], lam, margin)

    def test_empty_pool_and_overflow_do_not_drop_or_promote_rows(self):
        self.assertEqual(self.choose([], []), (None, 'empty_pool'))
        self.assertEqual(self.choose([40, 24], [1e308, -1e308]), (0, 'invalid_energy'))

    def test_call_does_not_mutate_saved_counts_or_energies(self):
        counts, energies = [40, 24], [0., -1.]
        before = copy.deepcopy((counts, energies))
        self.choose(counts, energies)
        self.assertEqual((counts, energies), before)


if __name__ == '__main__':
    unittest.main()
