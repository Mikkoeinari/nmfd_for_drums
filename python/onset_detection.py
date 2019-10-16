'''
This module handles Onset Detection tasks
'''
from scipy.signal import argrelmax
import numpy as np
from scipy.ndimage.filters import maximum_filter

def spectral_difference(spec):
    diff = np.log(spec * 1 + 1)
    diff[1:, :] = np.diff(spec, n=1, axis=0)
    spec_diff = np.sum(diff, axis=1)
    spec_diff = np.clip(spec_diff / max(spec_diff), 0, 1)
    return spec_diff


def pick_onsets(F, threshold=0.15, w=3.5):
    """
    Simple onset peak picking algorithm, picks local maxima that are
    greater than median of local maxima + correction factor.
    :param F: numpy array, Detection Function
    :param threshold: float, threshold correction factor
    :return: numpy array, peak indices
    """
    # Indices of local maxima in F
    localMaximaInd = argrelmax(F, order=1)

    # Values of local maxima in F
    localMaxima = F[localMaximaInd[0]]

    # Pick local maxima greater than threshold
    onsets = np.where(localMaxima >= threshold)
    # Onset indices array
    rets = localMaximaInd[0][onsets]

    # Check that onsets are valid onsets
    i = 0
    while i in range(len(rets) - 1):
        # Check that the ODF goes under the threshold between onsets
        if F[rets[i]:rets[i + 1]].min() >= threshold:
            rets = np.delete(rets, i + 1)
        # Check that two onsets are not too close to each other
        elif rets[i] - rets[i + 1] > -w:
            rets = np.delete(rets, i + 1)
        else:
            i += 1
    # Return onset indices
    return rets
