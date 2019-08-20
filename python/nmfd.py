'''
This module provides several NMFD and NMD implementations
'''

import numpy as np

#global variable to stop iteration and dump result
REQUEST_RESULT = False

def NMFD(X, iters=128, Wpre=[], include_priors=False,n_heads=1,remove_tails=False, semi_adaptive=False, beta=4, partially_fixed=False, add_h=0, fully_adaptive=False, random_init=False, hand_break=True):
    """
    Perform NMFD source separation
    :param X:  numpy array, The spectrogram
    :param iters: int, maximum number of iterations
    :param Wpre: numpy array, prior basis matrix
    :param include_priors: boolean, include training data to analysis to avoid false activations
    if no hits of a certain drum is present in the data.
    :param n_heads: int, number of Wpre heads
    :param remove_tails: boolean, remove tail prior templates
    :param semi_adaptive: boolean, use semi adaptive updates for W
    :param beta: beta of SA-NMFD
    :param partially_fixed: DO NOT USE, work in progress.
    :param add_h: int, add extra random basis vectors
    :param fully_adaptive: boolean, adapt W every iteration
    :param random_init: boolean, initialize W with random values
    :param hand_break:
    :return: numpy array, numpy array, int, H(Activations of different drums in X), W(basis matrix), normalized errors at iterations

    """
    # epsilon, added to matrix elements to ensure non-negativity in matrices
    eps = 10 ** -18

    #remove tails
    if remove_tails:
        Wpre=Wpre[:, :n_heads, :]

    # Add prior templates to the audio to prevent normalization error when no hits of certain drums are present
    # The multiplier is arbitrarily chosen, lower than .5 ensures that the quieter hits are not lost
    # as the soundcheck hits are possibly very loud compared to actual playing
    if include_priors:
        for i in range(int(Wpre.shape[1] / 2)):
            pass
            X = np.hstack((X, .28 * Wpre[:, i, :]))

    #remove prior templates
    if random_init:
        split=0
        if partially_fixed:
            split=n_heads
        Wpre[:,split:,:]=np.random.rand(Wpre.shape[0],Wpre.shape[1]-split, Wpre.shape[2])

    # add noise priors for partially_fixed, DO NOT USE, minimal improvement with serious speed drop.
    if partially_fixed or add_h > 0:
        W_z = np.random.rand(Wpre.shape[0], add_h, Wpre.shape[2])
        Wpre = np.concatenate((Wpre, W_z), axis=1)

    # data size
    M, N = X.shape
    W = Wpre
    M, R, T = W.shape
    # Initial H, non negative, non negative. any non negative value works
    H = np.full((R, N), .3)

    # Make sure the input is normalized
    W=W-W.min()
    W = W / W.max()
    X=X-X.min()
    X = X / X.max()

    # all ones matrix
    repMat = np.ones((M, N))

    # Spaceholder for errors
    err = np.zeros(iters)

    #Partially fixed coefficients:
    pf_alpha = R/(n_heads)
    pf_beta = (R-n_heads)/R

    # KLDivergence for cost calculation
    def KLDiv(x, y):
        return (x * np.log(x/y) - x + y).sum()

    # Itakura-Saito Divergence
    def ISDiv(x,y):
        return (x/y - np.log(x/y) - 1).sum()

    # Itakura-Saito Divergence
    # precomputed division, gives a little different results, only for the better
    def ISDiv2(x):
        return (x - np.log(x) - 1).sum()

    # Lambda calculation
    def sumLambda(W, H):
        W_temp=W.copy()
        if(partially_fixed):
            W_temp[:,:n_heads,:]=pf_alpha*W_temp[:,:n_heads,:]
            W_temp[:, n_heads:, :]=pf_beta*W_temp[:,n_heads:,:]
        Lambda = np.zeros((W_temp.shape[0], H.shape[1]))
        shifter = np.zeros((R, N + T + 1))
        shifter[:, T:-1] = H
        for t in range(T):
            Lambda += W_temp[:, :, t] @ shifter[:, T - t: -(t + 1)]
        return Lambda/Lambda.max()

    mu =0.8 # sparsity constraint 0.8 in use mode

    for i in range(iters):

        # Stop processing and spit out result if True
        if REQUEST_RESULT:
            return H, W, err / err.max()

        # Initialize Lambda and Lambda^
        if i==0:
            Lambda = sumLambda(W, H)+eps
            LambHat =X/Lambda + eps

        shifter = np.zeros((M, N + T))
        # Lambhat with T columns of zero padding
        shifter[:, :-T] = LambHat
        Hhat1 = np.zeros((R, N))
        Hhat2 = np.zeros((R, N))

        # the convolution calculation
        for t in range(T):
            Wt = W[:, :, t].T
            # Numerator calc
            Hhat1 += Wt @ shifter[:, t: t + N]
            # Denominator calc
            Hhat2 += Wt @ repMat + eps

        H = H * Hhat1 / (Hhat2 + mu)

        # precompute for error and next round
        Lambda = sumLambda(W, H)+eps
        LambHat = X/Lambda + eps

        # 2*eps to ensure the eps don't cancel out each other in the division.
        #err[i] = KLDiv(X + eps, Lambda + eps)

        #err[i] = ISDiv(X+eps,Lambda+eps)

        err[i]=ISDiv2(LambHat+eps)#+np.linalg.norm(H,ord=np.inf)*mu

        # Stopping criteria check
        if (i >= 1):
            errDiff = (abs(err[i] - err[i - 1]) / (err[0] - err[i] +eps))

            #print(errDiff)
            if errDiff < 0.0005 or err[i] > err[i - 1]:
                if hand_break:
                    break
                pass

        # Adaptive W - no considerable improvement double running time
        if semi_adaptive or partially_fixed or fully_adaptive:
            #adapt everything
            split = 0
            #fully adaptive alpha
            alpha = 0
            if (semi_adaptive):
                alpha = (1 - i / iters) ** beta
            if (partially_fixed):
                split=n_heads
            shifter = np.zeros((N + T, R))
            shifter[:-T, :] = H.T[:, :]

            for t in range(T):
                W[:, split:, t] = W[:, split:, t] * (
                        LambHat @ shifter[t:-(T - t), split:] / (repMat @ shifter[t:-(T - t), split:] + eps + (mu / T)))

            if partially_fixed:
                W = Wpre * (1-alpha) + (alpha) * W
            else:
                W = Wpre * alpha + (1 - alpha) * W

            W = W / W.max()

    return H, W, err / err.max()


def semi_adaptive_NMFB(X, Wpre, iters=100,n_heads=1, semi_adaptive=False, beta=4,partially_fixed=False,add_h=0, div=1, hand_break=True):
    '''
    :param X:  numpy array, The spectrogram
    :param Wpre: numpy array, prior basis matrix
    :param iters: int, maximum number of iterations
    :param n_heads: int, number of Wpre heads templates
    :param semi_adaptive: boolean, use sa-nmf
    :param beta: float, sa-nmf beta
    :param partially_fixed: boolean, use pf-nmf algorithm
    :param add_h: int, add extra random basis vectors
    :param div: int, what beta divergence to use, 0 for IS, 1 for KL, 2 for EUC ...
    :param hand_break:
    :return: Numpy array, list of Int, Activations matrix H , normalized errors at iterations
    '''
    epsilon = 10 ** -18
    X = X + epsilon
    if True:
        for i in range(int(Wpre.shape[1] / 2)):
            pass
            X = np.hstack((X, .28 * Wpre[:, i, :]))
    #add noise priors
    Wpre=Wpre[:,:n_heads,:]
    if partially_fixed or add_h > 0:
        W_z = np.random.rand(Wpre.shape[0], add_h, Wpre.shape[2])
        Wpre = np.concatenate((Wpre, W_z), axis=1)
    # Use only the first frame of NMFD prior
    W =Wpre[:,:,0]
    M,R=W.shape[0], W.shape[1]
    # Initial H, non negative, non zero. any non zero value works
    H = np.full((W.shape[1], X.shape[1]), .5)
    # normalize input
    W = W / W.max()
    X = X / X.max()
    # error space
    err = np.zeros(iters)

    def KLDiv(x, y):
        return sum(sum(x * np.log(x / y+epsilon) + (y - x)))
    def ISDiv(x, y):
        return ((x / y + epsilon) - np.log(x / y + epsilon) - 1).sum()

    mu = 0.8
    for i in range(iters):
        if partially_fixed:
            pf_alpha = R / (n_heads)
            pf_beta = (R - n_heads) / R
            Wt_alpha = W[:, :n_heads].T
            Wt_beta = W[:, n_heads:].T
            WH_alpha = pf_alpha * W[:, :n_heads] @ H[:n_heads, :]
            WH_beta = pf_beta * W[:, n_heads:] @ H[n_heads:, :]
        else:
            Wt = W.T
            WH = epsilon + np.dot(W, H)
        if partially_fixed:
            H_alpha= (Wt_alpha @ (X * (WH_alpha+WH_beta) ** (div- 2)))/(epsilon + (Wt_alpha @ (WH_alpha+WH_beta) ** (div- 1)))
            H_beta= (Wt_beta @ (X * (WH_alpha+WH_beta) ** (div- 2)))/(epsilon + (Wt_beta @ (WH_alpha+WH_beta) ** (div- 1)))
            H[:n_heads,:]   =  H[:n_heads,:]   * H_alpha+ mu
            H[n_heads:,:]  =  H[n_heads:,:]  * H_beta+ mu

        else:
            H= H*((Wt @ (X * WH ** (div- 2)))/(epsilon + (Wt @ WH ** (div- 1)))+ mu)

        if partially_fixed:
            WH_alpha=pf_alpha*W[:,:n_heads]@H[:n_heads,:]
            WH_beta=pf_beta*W[:,n_heads:]@H[n_heads:,:]
        else:
            WH = epsilon + np.dot(W, H)
        if partially_fixed:
            if div==1:
                err[i] = KLDiv(X + epsilon, (WH_alpha+WH_beta) + epsilon)
            else:
                err[i] = ISDiv(X + epsilon, (WH_alpha+WH_beta) + epsilon)
        else :
            if div==1:
                err[i] = KLDiv(X + epsilon, WH + epsilon)  # + (sparsity * normH)
            else:
                err[i] = ISDiv(X + epsilon, WH + epsilon)
        if (i >= 1):
            errDiff = (abs(err[i] - err[i - 1]) / (err[1] - err[i] + epsilon))
            if errDiff < 0.003 or err[i] > err[i - 1]:
                pass
                if hand_break:
                    break

        if semi_adaptive or partially_fixed:
            alpha=0
            if (semi_adaptive):
                alpha = (1 - i / iters) ** beta
            if partially_fixed:
                Ht=H[n_heads:,:].T
                Wu = (X * WH_beta ** (div - 2)@Ht)
                Wd= (WH_beta ** (div - 1))@Ht+epsilon
                W[:, n_heads:]=(Wt_beta.T*(Wu/(Wd))+mu)
            else:
                Ht=H.T
                Wu = (X * WH ** (div - 2)@Ht)
                Wd=(WH ** (div - 1))@Ht+epsilon
                W=W*(Wu/(Wd))+mu
            if semi_adaptive:
                W = Wpre[:,:,0] * alpha + (1 - alpha) * W

    return H, err / err.max()
