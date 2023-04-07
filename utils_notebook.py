import torch
import numpy as np
from typing import Dict, Optional, List


def probs_decrease(probs: np.array) -> np.array:
    L = len(probs)
    diffs = []
    for i in range(L):
        for j in range(i + 1, L):
            diffs.append(probs[j] - probs[i])
    return np.array(diffs)


def modal_probs_decreasing(
    _preds: Dict[int, torch.Tensor],
    _probs: torch.Tensor,
    layer: Optional[int] = None,
    verbose: bool = False,
    N: int = 10000,
    diffs_type: str = "consecutive",
    thresholds: List[float] = [-0.01, -0.05, -0.1, -0.2, -0.5],
) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime

    function can also be used for grount truth probabilities, set layer=None
    """
    nr_non_decreasing = {threshold: 0 for threshold in thresholds}
    # diffs = []
    for i in range(N):
        if layer is None:
            c = _preds[i]
        else:
            c = _preds[layer - 1][i]
        probs_i = _probs[:, i, c].cpu().numpy()
        if diffs_type == "consecutive":
            diffs_i = np.diff(probs_i)
        elif diffs_type == "all":
            diffs_i = probs_decrease(probs_i)
        else:
            raise ValueError()
        # diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i >= threshold):
                nr_non_decreasing[threshold] += 1
            else:
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {
        -1.0 * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()
    }
    return nr_decreasing


def modal_probs_decreasing_count(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int, verbose: bool = False, N: int = 10000) -> Dict[float, float]:
    """
    """
    diffs = 0
    for i in range(N):
        probs_i = _probs[:, i, _preds[layer - 1][i]].cpu().numpy()
        diffs_i = np.diff(probs_i)
        if diffs_i.min() < 0:
            diffs += 1
    return diffs

def modal_probs_decreasing_relative(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int, verbose: bool = False, N: int = 10000) -> Dict[float, float]:
    """
    """
    diffs = []
    for i in range(N):
        probs_i = _probs[:, i, _preds[layer - 1][i]].cpu().numpy()
        diffs.append((probs_i[1:] / probs_i[:-1] - 1.).min())
    return diffs

def f_probs_ovr_poe_logits_weighted(logits, threshold=0.):
    C = logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = (probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2))
    return probs


def f_probs_ovr_poe_logits_weighted_generalized(logits, threshold=0., weights=None):
    L, C = logits.shape[0], logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.
    if weights is not None:
        assert logits.shape[0] == weights.shape[0]
        for l in range(L):
            probs[l, :, :] = probs[l, :, :] ** weights[l]
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = (probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2))
    return probs


def modal_probs_average(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int) -> Dict[float, float]:
    """
    average modal probability in anytime-prediction regime
    """
    preds = []
    for i in range(10000):
        preds.append(_probs[:, i, _preds[layer - 1][i]])
    return torch.stack(preds, dim=1).mean(axis=1)


def get_scale_probs(probs_name_arr, probs_arr, T_arr, targets, C, L):

    def scale_probs(_probs, _targets, T):
        probs_scaled = _probs ** T
        probs_scaled = probs_scaled / np.repeat(probs_scaled.sum(axis=2)[:, :, np.newaxis], C, axis=2)
        preds_scaled = {i: torch.argmax(probs_scaled, dim=2)[i, :] for i in range(L)}
        acc_scaled = [(_targets == preds_scaled[i]).sum() / len(_targets) for i in range(L)]
        return probs_scaled, preds_scaled, acc_scaled

    scaled_dict = {x: {} for x in probs_name_arr}
    for probs_name, probs_arr in zip(probs_name_arr, probs_arr):
        for T in T_arr:
            scaled_dict[probs_name][T] = scale_probs(probs_arr, targets, T)
    
    return scaled_dict


def anytime_caching(_probs: torch.Tensor, N: int, L: int) -> torch.Tensor:
    _preds = []
    _probs_stateful = []
    for n in range(N):
        preds_all, probs_stateful_all = [], []
        max_prob_all, pred_all, max_id = 0., None, 0.
        for l in range(L):
            _max_prob, _pred = _probs[l, n, :].max(), _probs[l, n, :].argmax()
            if _max_prob >= max_prob_all:
                max_prob_all = _max_prob
                pred_all = _pred
                prob_stateful_all = _probs[l, n, :]
                max_id = l
            else:
                prob_stateful_all = _probs[max_id, n, :]
            preds_all.append(pred_all)
            probs_stateful_all.append(prob_stateful_all)
        _preds.append(torch.stack(preds_all))
        _probs_stateful.append(torch.stack(probs_stateful_all))

    _preds = torch.stack(_preds)
    _probs_stateful = torch.stack(_probs_stateful)
    return _probs_stateful.permute(1, 0, 2)