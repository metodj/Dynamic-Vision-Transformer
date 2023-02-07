import torch
import numpy as np
from typing import Dict


def modal_probs_decreasing(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int, verbose: bool = False, N: int = 10000) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime
    """
    nr_non_decreasing = {-0.01: 0, -0.05: 0, -0.1: 0, -0.2: 0, -0.5: 0}
    diffs = []
    for i in range(N):
        probs_i = _probs[:, i, _preds[layer - 1][i]].cpu().numpy()
        diffs_i = np.diff(probs_i)
        diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i >= threshold):
                nr_non_decreasing[threshold] += 1
            else:
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {-1. * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()}
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


def modal_probs_average(_preds: Dict[int, torch.Tensor], _probs: torch.Tensor, layer: int) -> Dict[float, float]:
    """
    average modal probability in anytime-prediction regime
    """
    preds = []
    for i in range(10000):
        preds.append(_probs[:, i, _preds[layer - 1][i]])
    return torch.stack(preds, dim=1).mean(axis=1)