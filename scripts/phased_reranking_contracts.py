"""Testable scoring/label boundaries for the planned residual model; no fitter."""
import numpy as np


def residual_choice(counts,correction,lam,margin):
    """Stable original frequency winner unless the strict override margin is met."""
    counts=np.asarray(counts,dtype=float)
    if not len(counts): return None
    if np.any(counts<=0) or np.any(counts>64) or counts.sum()>64 or np.any(counts[:-1]<counts[1:]):
        raise ValueError('Expected original positive counts sorted by frequency')
    if lam<0 or margin<0: raise ValueError('Negative policy parameters')
    if lam==0 or correction is None: return 0
    correction=np.asarray(correction,dtype=float)
    if correction.shape!=counts.shape or not np.isfinite(correction).all(): return 0
    scores=np.log(counts/64.)+lam*correction
    winner=int(np.argmax(scores))
    return winner if scores[winner]-scores[0]>margin else 0


def soft_targets(candidates,fitting_catalog_counts):
    """All observed fitting-fold outcomes retain positive mass when in the pool."""
    if len(candidates)!=len(set(candidates)): raise ValueError('Duplicate candidate state')
    if any(n<=0 for n in fitting_catalog_counts.values()): raise ValueError('Invalid fitting counts')
    raw=np.array([fitting_catalog_counts.get(c,0) for c in candidates],dtype=float)
    total=sum(fitting_catalog_counts.values())
    covered=float(raw.sum()/total) if total else 0.
    return (raw/raw.sum() if raw.sum() else None),covered


def bounded_failure_weight(positive_weights,emphasize=False):
    """The only optional emphasis is capped at two; no rarity term exists."""
    if positive_weights is None: return 0.
    return 2. if emphasize and positive_weights[0]==0 else 1.


def correspondence_status(source_maps,candidate_maps,source_elements,candidate_elements,provenance_verified):
    if not provenance_verified: return 'unverified_provenance'
    if len(source_maps)!=len(source_elements) or len(candidate_maps)!=len(candidate_elements):
        return 'inventory_length_mismatch'
    if not source_maps or not candidate_maps or 0 in source_maps or 0 in candidate_maps:
        return 'maps_missing'
    if len(set(source_maps))!=len(source_maps) or len(set(candidate_maps))!=len(candidate_maps):
        return 'duplicate_maps'
    if set(source_maps)!=set(candidate_maps): return 'map_inventory_mismatch'
    if dict(zip(source_maps,source_elements))!=dict(zip(candidate_maps,candidate_elements)):
        return 'mapped_element_mismatch'
    return 'verified'
