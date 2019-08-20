'''
This module handles Onset Detection tasks
'''

from scipy.ndimage.filters import maximum_filter
from scipy.signal import argrelmax
import numpy as np

# superlux from madmom (Boeck et al)
def superflux(spec_x=[], A=None, B=None, win_size=8):
    """
    Calculate the superflux envelope according to Boeck et al.

    :param spec_x: optional, numpy array, A Spectrogram the superflux envelope is calculated from, X
    :param A: optional, numpy array, frequency response of the decomposed spectrogram, W
    :param B: optional, numpy array, activations of a decomposed spectrogram, H
    :param win_size: int, Hann window size, used to smooth the recomposition of

    :return: Superflux ODF of a spectrogram
    :notes: Must check inputs so that A and B have to be present together
    """

    # if A and B, the spec_x is recalculated
    if A is not None:
        # window function
        kernel = np.hamming(win_size)

        # apply window
        B = np.convolve(B, kernel, 'same')

        # rebuild spectrogram

        spec_x = np.outer(A, B)

    # To log magnitude
    spec_x = np.log(spec_x * 1 + 1)

    diff = np.zeros_like(spec_x.T)

    # Apply max filter
    max_spec = maximum_filter(spec_x.T, size=(1, 3))

    # Spectral difference
    diff[1:] = (spec_x.T[1:] - max_spec[: -1])

    # Keep only positive difference
    pos_diff = np.maximum(0, diff)

    # Sum bins
    sf = np.sum(pos_diff, axis=1)

    # normalize
    sf = sf / max(sf)

    # return ODF
    return sf


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
    # (-.2 to move optimal threshold range away from zero in automatic threshold
    # calculation, This should not make a difference but it does, investigate)
    onsets = np.where(localMaxima >= threshold -.2)
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
