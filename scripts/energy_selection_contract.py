"""Pure frequency-plus-energy policy. No data loading, fitting or physical calls."""
import math


def energy_choice(counts, energies_kcal_mol, lam, margin, *,
                  solvent_resolved=False, pool_compatible=False, reliable=None):
    """Return (original candidate index, reason); any unreliable pool abstains.

    The integration layer must verify candidate hashes, protocol identity,
    conservation, solvent evidence and topology before setting these flags.
    Energies use one common within-row protocol and kcal/mol units. This
    function does not itself certify a chemical calculation's reliability.
    """
    if not math.isfinite(lam) or not math.isfinite(margin) or lam < 0 or margin < 0:
        raise ValueError('Lambda and margin must be finite and nonnegative')
    counts = tuple(counts)
    if any(isinstance(n, bool) or not math.isfinite(n) or n <= 0 or
           n != int(n) or n > 64 for n in counts):
        raise ValueError('Counts must be positive integers no larger than 64')
    if sum(counts) > 64 or any(a < b for a, b in zip(counts, counts[1:])):
        raise ValueError('Preserve the original descending frequency counts')
    if not counts:
        return None, 'empty_pool'
    if lam == 0:
        return 0, 'zero_intervention'
    if not solvent_resolved:
        return 0, 'uncertain_solvent'
    if not pool_compatible:
        return 0, 'incompatible_pool'
    if reliable is None or len(reliable) != len(counts) or any(v is not True for v in reliable):
        return 0, 'unreliable_candidate'
    if energies_kcal_mol is None:
        return 0, 'missing_energy'
    energies = tuple(energies_kcal_mol)
    if len(energies) != len(counts):
        return 0, 'missing_energy'
    try:
        if any(v is None or isinstance(v, bool) or not math.isfinite(v) for v in energies):
            return 0, 'invalid_energy'
        scores = [math.log(n / 64.) - lam * (e - energies[0])
                  for n, e in zip(counts, energies)]
    except (TypeError, ValueError, OverflowError):
        return 0, 'invalid_energy'
    if not all(math.isfinite(s) for s in scores):
        return 0, 'invalid_energy'
    winner = max(range(len(scores)), key=scores.__getitem__)
    if scores[winner] - scores[0] > margin:
        return winner, 'energy_override'
    return 0, 'frequency_retained'
