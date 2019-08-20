'''
This module handles Onset Detection tasks
'''
from scipy.signal import argrelmax
import numpy as np

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
