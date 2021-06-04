# -*- coding: utf-8 -*-
"""
This module provides the Carathéodory-Fejér algorithm to calculate the Partial
Fraction Decomposition representation of the best rational approximation of the
exponential function on the negative real axis.

The algorithm is a Python implementation in arbitrary precision of the MATLAB
script given in

    T. Schmelzer and L. N. Trefethen, “Evaluating matrix functions for exponential integrators via Carathéodory–Fejér approximation and contour integrals,” Electronic Transactions on Numerical Analysis, vol. 29, pp. 1–18, 2007.
"""

import numpy as np
import mpmath as mp
from scipy.linalg import hankel

def fft(yg, inverse=False):
    """
    Modified version from the apfft package:

    Performs an fft in arbitrary precision arithmetic
    Arithmetic is implemented via the package mpmath
    Precision should be set by setting the variable mp.mp.dps to the desired precision

    Currently, the input vector must have a length given by a power of 2
    The algorithm used here was based on code posted at https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/

    Parameters
    ----------
    yg: 1D numpy vector, with yg.shape[0] a power of 2
        elements of yg should be of type mp.mpc

    Returns
    -------
    nd.array
    """

    ifac = 1 if inverse else -1
    exp = np.vectorize(mp.exp)
    N = yg.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")
    N_min = min(N, 8)
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.array(mp.arange(N_min))
    k = n[:, None]
    A = ifac*mp.mpc(2j) * mp.pi * n * k / mp.mpf(N_min)
    M = exp(A)
    X = M.dot(yg.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        A = ifac*mp.mpc(1j) * mp.pi * np.array(mp.arange(X.shape[0])) / mp.mpf(X.shape[0])
        factor = exp(A)[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])
    # build-up each level of the recursive calculation all at once
    if inverse:
        return X.ravel()/mp.mpf(N)
    else:
        return X.ravel()

def CaratheodoryFejer(n, verbose=False, dps=30, K=75, nf=1024):
    """ Wrapper around mpCaratheodoryFejer to return ndarrays of complex type.

    This is wrapper around the multi-precision Caratheodory-Fejer routine to
    return np.ndarray's of complex type

    Parameters
    ----------
    n : int
        the order of the approximation
    verbose : bool, optional
        if True, the routine will print what it's doing (default: False)
    dps : int, optional
        the number of digits in the mpmath calculation (default: 15)
    K : int, optional
        the order of the Chebyshev approximation (default: 75)
    nf : int, optional
        the size of the FFT used (default: 1024)

    Returns
    -------
    zk :ndarray
        1D array of complex type containing the poles of the PFD
    ck: ndarray
        1D array of complex type  containing the coefficients of the PFD
    rinf: float
        Asymptotic error of the approximation
    """
    def _tocomplex(x):
        return np.array(x, dtype=complex)

    with mp.workdps(dps):
        zk, ck, rinf = mpCaratheodoryFejer(n, verbose, dps, K, nf)
        zk = _tocomplex(zk)
        ck = _tocomplex(ck)
        rinf = float(rinf)

    return zk, ck, rinf

def mpCaratheodoryFejer(n, verbose=False, dps = 30, K =75, nf =1024):
    """ Calculates the best rational approxmation to exp(x) on negative real
    axis.

    This function calculations the best rational approximation to exp(x) on the
    negative real axis in arbitrary precision and returns the result in partial
    fraction decompostion in double precision.

    The algorithm is a Python implementation in arbitrary precision of the MATLAB
    script given in

    T. Schmelzer and L. N. Trefethen, “Evaluating matrix functions for exponential integrators via Carathéodory–Fejér approximation and contour integrals,” Electronic Transactions on Numerical Analysis, vol. 29, pp. 1–18, 2007.

    Parameters
    ----------
    n : int
        the order of the approximation
    verbose : bool, optional
        if True, the routine will print what it's doing (default: False)
    dps : int, optional
        the number of digits in the mpmath calculation (default: 15)
    K : int, optional
        the order of the Chebyshev approximation (default: 75)
    nf : int, optional
        the size of the FFT used (default: 1024)

    Returns
    -------
    zk :ndarray
        1D array of mp.mpc type containing the poles of the PFD
    ck: ndarray
        1D array of mp.mpc type containing the coefficients of the PFD
    rinf: mp.mpf
        Asymptotic error of the approximation
    """
    def polyval(coeffs, x):
        x = np.asarray(x)
        res = np.asarray(x)
        res = coeffs[0]
        for c in coeffs[1:]:
            res *= x
            res += c
        return res

    mpexp = np.vectorize(mp.exp)
    mpreal = np.vectorize(mp.re)
    mpimag = np.vectorize(mp.im)
    mpfabs = np.vectorize(mp.fabs)
    mppower = np.vectorize(mp.power)

    with mp.workdps(dps):

        nf=mp.mpf(nf)

        twopij = mp.mpc('0','2')*mp.pi
        one = mp.mpf('1')
        w = mpexp(twopij/nf*np.array(mp.arange(0,nf)))
        t = mpreal(w)
        scale = mp.mpf('9')
        if verbose: print('1. Calculating Chebyshev nodes')
        F = mpexp(scale*(t-one)/(t+one+mp.mp.eps))
        c = mpreal(fft(F))/nf
        if verbose: print('2. Building Hankel matrix')
        f = polyval(c[:K+1][::-1],w)
        h = hankel(c[1:K+1])
        if verbose: print('3. Do SVD')
        U, S, V = mp.svd_r(mp.matrix(h), full_matrices=True, compute_uv=True)
        s = S[n]
        u = U[::-1,n]
        v = V[n,:]
        if verbose: print('4. Do FFTs')
        zz = int(nf-K)*[mp.mpf('0')]
        b = fft(np.concatenate((u, zz)))/fft(np.concatenate((v, zz)))
        rt = f-s*w**K*b
        rtc = mpreal(fft(rt))/nf
        if verbose: print('5. Start root finding')
        zr = np.array(mp.polyroots(v, maxsteps=20000))
        qk = zr[mpfabs(zr)>one]
        qc = np.poly(qk)
        pt = rt*polyval(qc,w)
        ptc1 = mpreal(fft(pt)/nf)
        ptc = ptc1[n::-1]
        ck = 0*qk
        if verbose: print('6. Start poles/residu')
        for k in range(n):
            q = qk[k]
            q2 = np.poly(qk[qk!=q]);
            ck[k]= np.polyval(ptc,q)/polyval(q2,q)
        zk = scale*(qk-one)**2./(qk+one)**2
        ck = mp.mpf('4')*ck*zk/(qk**2-one)
        idx = np.argsort(mpimag(zk))
        zk = zk[idx]
        ck = ck[idx]

        rinf = mpreal(one/mp.mpf('2')*(one + np.sum(ck/zk))).item()

    return zk, ck, rinf
