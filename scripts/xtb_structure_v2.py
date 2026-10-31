"""Conservative, atom-indexed validation of saved xTB geometries; no physics."""
import math

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from rerank_analysis import PARSER

VERSION = 'indexed_connectivity_h2_resonance_v2'
RESONANCE_LIMIT = 256
RESONANCE_FLAGS = int(Chem.ResonanceFlags.ALLOW_CHARGE_SEPARATION |
                      Chem.ResonanceFlags.ALLOW_INCOMPLETE_OCTETS)


def expected_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles, PARSER)
    if mol is None:
        raise ValueError('unparseable_fragment')
    return Chem.AddHs(mol)


def edges(mol):
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}


def indexed_key(mol):
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    return (tuple((a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
                   a.GetNumRadicalElectrons()) for a in mol.GetAtoms()),
            tuple(sorted((*sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())),
                          b.GetBondTypeAsDouble()) for b in mol.GetBonds())))


def resonance_keys(mol):
    supplier = Chem.ResonanceMolSupplier(mol, RESONANCE_FLAGS, RESONANCE_LIMIT+1)
    supplier.SetNumThreads(1)
    if len(supplier) > RESONANCE_LIMIT:
        return None
    return {indexed_key(mol)} | {indexed_key(m) for m in supplier}


def parse_geometry(block, expected):
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        raise ValueError('invalid_xyz')
    if [a.GetAtomicNum() for a in mol.GetAtoms()] != [a.GetAtomicNum() for a in expected.GetAtoms()]:
        raise ValueError('atom_count_or_order_mismatch')
    coords = mol.GetConformer().GetPositions()
    if not all(math.isfinite(float(v)) for row in coords for v in row):
        raise ValueError('nonfinite_coordinates')
    if any(math.dist(coords[i], coords[j]) < .4 for i in range(len(coords)) for j in range(i)):
        raise ValueError('overlapping_atoms')
    return mol


def connectivity(mol, expected):
    # RDKit's default connect-the-dots path omits H-H. Its covalent-radius
    # implementation recognizes H2 without borrowing any candidate bonds.
    h2 = expected.GetNumAtoms() == 2 and all(a.GetAtomicNum() == 1 for a in expected.GetAtoms())
    rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False,
        charge=Chem.GetFormalCharge(expected), useVdw=h2, covFactor=1.3)
    return 'covalent_radii_1.3_H2_only' if h2 else 'rdkit_connect_the_dots'


def stereo_preserved(expected, geometry):
    # Unspecified stereochemistry is not invented. Specified stereochemistry
    # must agree with a coordinate-based assignment on the expected graph.
    has_stereo = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in expected.GetAtoms()) or any(
        b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY) for b in expected.GetBonds())
    if not has_stereo:
        return True
    observed = Chem.Mol(expected)
    observed.RemoveAllConformers()
    observed.AddConformer(Chem.Conformer(geometry.GetConformer()), assignId=True)
    Chem.AssignStereochemistryFrom3D(observed, replaceExistingTags=True)
    for a, b in zip(expected.GetAtoms(), observed.GetAtoms()):
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and a.GetChiralTag() != b.GetChiralTag():
            return False
    for a, b in zip(expected.GetBonds(), observed.GetBonds()):
        if a.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY) and a.GetStereo() != b.GetStereo():
            return False
    return True


def check_geometry(smiles, input_xyz, optimized_xyz):
    """No energy, outcome label, frequency or reference mapping is consulted."""
    result = {'version': VERSION, 'valid': False}
    try:
        expected = expected_molecule(smiles)
        original = parse_geometry(input_xyz, expected)
        observed = parse_geometry(optimized_xyz, expected)
        if expected.GetNumAtoms() == 1:
            # Element/order, finite coordinate and the separate command-charge/
            # spin checks are sufficient: a single atom has no bonds/conformers.
            return dict(result, valid=True, reason='monatomic_inventory_preserved')
        mode = connectivity(observed, expected)
        connectivity(original, expected)
        wanted, found = edges(expected), edges(observed)
        result.update(connectivity_method=mode,
            input_connectivity_matches_expected=edges(original) == wanted,
            removed_bonds=[list(e) for e in sorted(wanted-found)],
            added_bonds=[list(e) for e in sorted(found-wanted)])
        if found != wanted:
            result['reason'] = 'indexed_connectivity_changed_or_ambiguous'
            coords = observed.GetConformer().GetPositions()
            result['changed_bond_distances_angstrom'] = [
                {'atoms':list(e), 'elements':[expected.GetAtomWithIdx(i).GetSymbol() for i in e],
                 'distance':math.dist(coords[e[0]], coords[e[1]])} for e in sorted(wanted ^ found)]
            return result
        if not stereo_preserved(expected, observed):
            return dict(result, reason='specified_stereochemistry_changed_or_unresolved')
        try:
            rdDetermineBonds.DetermineBondOrders(observed, charge=Chem.GetFormalCharge(expected),
                allowChargedFragments=True, embedChiral=False)
            Chem.SanitizeMol(observed)
        except Exception as exc:
            return dict(result, reason='bond_order_reconstruction_unresolved', exception=str(exc))
        result['reconstructed_smiles'] = Chem.MolToSmiles(observed, isomericSmiles=False)
        if indexed_key(expected) == indexed_key(observed):
            return dict(result, valid=True, reason='indexed_graph_preserved')
        first, second = resonance_keys(expected), resonance_keys(observed)
        if first is None or second is None:
            return dict(result, reason='resonance_enumeration_limit')
        if first & second:
            return dict(result, valid=True, reason='verified_indexed_resonance_equivalence',
                resonance_forms_examined=[len(first), len(second)])
        return dict(result, reason='bond_order_or_charge_identity_unresolved')
    except Exception as exc:
        return dict(result, reason='structure_validation_exception', exception=str(exc))
