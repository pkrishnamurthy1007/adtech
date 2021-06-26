import numpy as np
import typing

def spread_outliers(S, percentile=97.5) -> typing.Iterable:
    OUTTHRESH = np.percentile(S, percentile)
    OUTI = S > OUTTHRESH
    T = OUTI * OUTTHRESH + (1-OUTI) * S
    T = (S.sum() / T.sum()) * T
    assert abs(T.sum() - S.sum()) < 1e-10
    return T
def cma(S, window) -> typing.Iterable:
    L = S.__len__()
    CMAker = [1/window] * window
    return np.convolve([*S, *S, *S], CMAker, mode="same")[L:-L]
def ema(S, window) -> typing.Iterable:
    """
    \sum_{r^i} = s = 1 + r + r^2 + ....
    s*r = r + r^2 + r^3 + ... = s-1
    s * r = s - 1 ===> s = 1 / (1-r)
    s - 1 = 1 / (1-r) - 1 = r / (1-r)
    r \approx (window-1)/window

    ema(X,t) = (1-r)*X[t] + r*ema(X,t-1)
    """
    L = S.__len__()
    r = (window-1)/window
    EMAker = (1-r) * 1/(1-r**window) * np.array([r**i for i in range(window)])
    assert abs(EMAker.sum() - 1) < 1e-10
    return np.convolve([*S, *S, *S], EMAker, mode="same")[L:-L]

def wavg(V, W):
    if W.sum() == 0:return 0
    if V.shape.__len__() > 1:
        return (V*W.values.reshape(-1,1)).sum() / W.sum()
    else:
        return (V*W).sum() / W.sum()
def wvar(V,W):
    mu = wavg(V,W)
    var = wavg((V - mu)**2,W)
    return var
def wstd(V,W):
    return wvar(V,W)**0.5
def get_wavg_by(df, col):
    def wavg_by(V):
        return wavg(V, W=df.loc[V.index, col])
    return wavg_by
def get_wstd_by(df, col):
    def wstd_by(V):
        return wstd(V, W=df.loc[V.index, col])
    return wstd_by

def lapprox(X, W, l, r):
    return X[l]
def midapprox(X, W, l, r):
    return X[(l+r)//2]
def wavgapprox(X, W, l, r):
    return wavg(X[l:r], W[l:r])
def interval_fit(X, W, nintervals, xapprox) -> typing.Tuple[typing.List[int],float]:
    """
    PREMISE:
        define subset of X,W w/ leftmost bound of l
        we then say there must be a unique minimum interval split for k remaining intervals

        then we test the end pt for this interval for every remaining index from l to N
    """
    assert len(X) == len(W)
    N = len(X)
    # dp matrix of size (N+1),(nintervals+1) representing fit err and interval splits
    #   for subsets starting at time index `r` and w/ `c` intervels left to allocate
    dp = np.empty((N+1, nintervals+2, 2)).astype(object)
    # l >= len(X|W): all indices assigned to interval - terminate w / 0 MSE
    dp[N, :] = 0, []
    # k > nintervals: k represetns # of intervals allocated - so if k > nintervals
    #                 we have used too many intervals - terminate w / `inf` MSE
    dp[:, -1] = float('inf'), []
    for l in reversed(range(N)):
        for k in reversed(range(0, nintervals+1)):
            # probe remaining time slots for first interval break
            def yield_suffix_fits():
                for r in range(l+1, N+1):
                    # interval err over l:r
                    interval_eps = W[l:r] * (X[l:r] - xapprox(X, W, l, r))**2
                    eps_suffix, int_suffix = dp[r, k+1]
                    yield interval_eps.sum() + eps_suffix, [r] + int_suffix
            dp[l, k] = min(yield_suffix_fits())
    eps,interval_bounds = dp[0, 0]
    return interval_bounds,eps

def interval_transform(X,W,nintervals,xapprox,interval_bounds,*_):
    assert len({*interval_bounds}) == nintervals
    interval_bounds = [0, *interval_bounds]
    Xapprox = np.zeros(len(X))
    for lb, ub in zip(interval_bounds[:-1], interval_bounds[1:]):
        Xapprox[lb:ub] = xapprox(X, W, lb, ub)
    return Xapprox

def interval_fit_transform(X, W, nintervals, xapprox):
    return interval_transform(X,W,nintervals,xapprox,*interval_fit(X, W, nintervals, xapprox))