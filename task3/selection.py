"""Selection rules for one candidate prefix (PA3 Task 3, step 2).

All rules see only the revealed candidates, in the current random order. Ties: highest-score selection
keeps the earliest candidate with the top score; answer votes (self-consistency and RWC) are resolved by
the answer whose earliest supporting rollout comes first. RWC uses average ranks for tied scores.
"""
import numpy as np
from scipy.stats import rankdata

from task3 import config as C


def vote(groups, weights=None):
    """Winning answer group and its vote table. groups: group id per candidate, in order."""
    weights = np.ones(len(groups)) if weights is None else np.asarray(weights, dtype=np.float64)
    uniq, first = np.unique(groups, return_index=True)
    W = np.array([weights[groups == g].sum() for g in uniq])
    Wr = np.round(W, 12)
    top = Wr.max()
    tied = np.flatnonzero(Wr == top)
    win = tied[np.argmin(first[tied])]
    return int(uniq[win]), int(first[win]), uniq, W


def rwc_weights(scores):
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = rankdata(-np.asarray(scores, dtype=np.float64), method="average")   # rank 1 = best
    return (n - r) / (n - 1)


def rwc(groups, scores):
    w = rwc_weights(scores)
    g, first, uniq, W = vote(groups, w)
    if len(W) > 1:
        s = np.sort(W)[::-1]
        margin = (s[0] - s[1]) / W.sum() if W.sum() > 0 else 0.0
    else:
        margin = 1.0 if W.sum() > 0 else 0.0   # runner-up vote is zero when only one answer exists
    return g, first, margin


def highest(scores):
    return int(np.argmax(np.asarray(scores)))   # first maximum = earliest in the current order


def visconf(kl_tokens, vis_tokens, seg):
    """Candidate-calibrated confidence for a candidate set.

    kl_tokens / vis_tokens: concatenated token values of the candidates; seg: start offsets (len N + 1).
    Returns C_i (mean calibrated confidence), the VisConf score Z_i = V_i * C_i, the tokenwise fusion
    Z_i^token = mean_t c_t v_t, and the per-token calibrated confidence c."""
    q05, q95 = np.percentile(kl_tokens, [C.CAL_LO, C.CAL_HI])          # linear interpolation (numpy default)
    c = np.clip((kl_tokens - q05) / max(q95 - q05, C.CAL_EPS), 0.0, 1.0)
    lens = np.diff(seg)
    Ci = np.add.reduceat(np.append(c, 0.0), seg[:-1]) / np.maximum(lens, 1)
    Vi = np.add.reduceat(np.append(vis_tokens, 0.0), seg[:-1]) / np.maximum(lens, 1)
    Zt = np.add.reduceat(np.append(c * vis_tokens, 0.0), seg[:-1]) / np.maximum(lens, 1)
    Ci[lens == 0], Vi[lens == 0], Zt[lens == 0] = 0.0, 0.0, 0.0
    return Ci, Vi * Ci, Zt, c
