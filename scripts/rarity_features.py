"""Map-independent local-change signatures for descriptive training-support audits."""
from collections import Counter
from functools import lru_cache
import hashlib
import json

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')
VERSION = 'explicit-h-morgan-difference-r1-r2-v1'
PARSER = Chem.SmilesParserParams()
PARSER.removeHs = False
PARSER.sanitize = True
GENERATORS = tuple(rdFingerprintGenerator.GetMorganGenerator(radius=r, includeChirality=False)
                   for r in (1, 2))


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=4096)
def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol is None:
        raise ValueError('RDKit could not parse the supplied molecular state')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return tuple(tuple(sorted(g.GetSparseCountFingerprint(mol).GetNonzeroElements().items()))
                 for g in GENERATORS)


def difference(left, right):
    a, b = dict(left), dict(right)
    delta = tuple((key, b.get(key, 0) - a.get(key, 0))
                  for key in sorted(a.keys() | b.keys()) if b.get(key, 0) != a.get(key, 0))
    return 'no_local_change' if not delta else digest(delta)


def signatures(source, target):
    before, after = descriptors(source), descriptors(target)
    return tuple(difference(a, b) for a, b in zip(before, after))


def input_fingerprint(smiles):
    return digest(descriptors(smiles)[1])


def support_band(n):
    if n == 0:
        return '0'
    if n < 10:
        return '1-9'
    if n < 100:
        return '10-99'
    if n < 1000:
        return '100-999'
    return '1000+'
