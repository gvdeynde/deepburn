# -*- coding: utf-8 -*-
"""
This module provides tools to generate and use Chebyshev Rational
Approximations (CRAs) for the exponential function on the negative real axis.

CRAs published in literature are available:

Origins
-------
Pusa2012:
     M. Pusa, “Correction to Partial Fraction Decomposition Coefficients
     for Chebyshev Rational Approximation on the Negative Real Axis,” arXiv,
     2012, [Online]. Available: https://arxiv.org/abs/1206.2880v1.

Calvin2021:
    O. Calvin, S. Schunert, and B. Ganapol, “Global error analysis of the
    Chebyshev rational approximation method,” Annals of Nuclear Energy, vol.
    150, p. 107828, 2021, doi: 10.1016/j.anucene.2020.107828.

Zhang2020:
     B. Zhang, X. Yuan, Y. Zhang, H. Tang, and L. Cao, “Development of
     a versatile depletion code AMAC,” Annals of Nuclear Energy, vol. 143,
     p. 107446, 2020, doi: 10.1016/j.anucene.2020.107446.

VandenEynde2021:
    G. Van den Eynde, "Validated CRAM coefficients for depletion calculations",
    Journal of Nuclear Engineering, to be submitted.
"""


from dataclasses import dataclass, field, asdict
import re
import importlib.resources as pkg_resources
import mpmath as mp
import numpy as np
from scipy.linalg import hankel
from scipy.optimize import fsolve
from . import data


@dataclass
class CRA:
    """unmutable data class to hold information on a CRA

    Attributes
    ----------
    origin: str
        The origin of the approximation, for example "Pusa"
    order: int
        The order of the approximation
    rinf: float
        The absolute error of the approximation at $-\\infty$
    alpha: ndarray
        1D array of size `order` containing data with `complex` type
    theta: ndarray
        1D array of size `order` containing data with `complex` type
    """

    origin: str
    order: int
    rinf: float
    alpha: np.ndarray
    theta: np.ndarray
    _extrema: np.ndarray = field(init=False, repr=False, default=None)
    _extremavals: np.ndarray = field(init=False, repr=False, default=None)

    def __call__(self, x, dps=None):
        """Evaluates the CRA in x.

        Parameters
        ----------
        x: ndarray
            the array where the CRA should be evaluated
        dps: int or None
            if None, double precision (complex) arithmetic is used
            if int, mpmath arbitrary precision arithmetic is used with working
            dps set to dps

        Returns
        -------
        ndarray
            the resulting evaluation as an array of floats or mpf
        """

        if dps:
            mpreal = np.vectorize(mp.re)

            with mp.workdps(dps):
                res = mp.mpc("0") * np.ones_like(x)
                xx = mp.mpf("1") * x

                for i in range(self.order // 2):
                    res += mp.mpc(self.alpha[i]) / (xx - mp.mpc(self.theta[i]))

                res *= mp.mpf("2")

                if self.order % 2:
                    res += mp.mpc(self.alpha[-1]) / (xx - mp.mpc(self.theta[-1]))

                res += mp.mpc(self.rinf) * np.ones_like(x)

                res = mpreal(res)

        else:
            res = np.zeros_like(x, dtype=complex)

            for i in range(self.order // 2):
                res += self.alpha[i] / (x - self.theta[i])

            res *= 2

            if self.order % 2:
                res += self.alpha[-1] / (x - self.theta[-1])

            res += self.rinf

            res = np.real(res)

        return res

    def derivative(self, x, dps=None):
        """Evaluates the derivative of the CRA in x.

        Parameters
        ----------
        x: ndarray
            the array where the CRA should be evaluated
        dps: int or None
            if None, double precision (complex) arithmetic is used
            if int, mpmath arbitrary precision arithmetic is used with working
            dps set to dps

        Returns
        -------
        ndarray
            the resulting evaluation as an array of floats or mpf
        """

        if dps:
            mpreal = np.vectorize(mp.re)

            with mp.workdps(dps):
                res = mp.mpc("0") * np.ones_like(x)
                xx = mp.mpf("1") * x

                for i in range(self.order // 2):
                    res -= mp.mpc(self.alpha[i]) / (
                        (xx - mp.mpc(self.theta[i])) * (xx - mp.mpc(self.theta[i]))
                    )

                res *= mp.mpf("2")

                if self.order % 2:
                    res -= mp.mpc(self.alpha[-1]) / (
                        (xx - mp.mpc(self.theta[-1])) * (xx - mp.mpc(self.theta[-1]))
                    )

                res = mpreal(res)

        else:
            res = np.zeros_like(x, dtype=complex)

            for i in range(self.order // 2):
                res -= self.alpha[i] / ((x - self.theta[i]) * (x - self.theta[i]))

            res *= 2

            if self.order % 2:
                res -= self.alpha[-1] / ((x - self.theta[-1]) * (x - self.theta[-1]))

            res = np.real(res)

        return res

    def error(self, x, dps=None):
        """Evaluates the error between the CRA and exp in x.

        Parameters
        ----------
        x: ndarray
            the array where the error should be evaluated
        dps: int or None
            if None, double precision (complex) arithmetic is used
            if int, mpmath arbitrary precision arithmetic is used with working
            dps set to dps

        Returns
        -------
        ndarray
            the resulting error as an array of floats or mpf
        """

        if dps:
            mpexp = np.vectorize(mp.exp)

            with mp.workdps(dps):
                error = self(x, dps) - mpexp(x)

        else:
            error = self(x) - np.exp(x)

        return error

    def errorderivative(self, x, dps=None):
        """Evaluates the derivative of the error between the CRA and exp in x.

        Parameters
        ----------
        x: ndarray
            the array where the error should be evaluated
        dps: int or None
            if None, double precision (complex) arithmetic is used
            if int, mpmath arbitrary precision arithmetic is used with working
            dps set to dps

        Returns
        -------
        ndarray
            the resulting derivative of the error as an array of floats or mpf
        """

        if dps:
            mpexp = np.vectorize(mp.exp)
            mpre = np.vectorize(mp.re)

            with mp.workdps(dps):
                errorderiv = mpre(self.derivative(x, dps) - mpexp(x))

        else:
            errorderiv = np.real(self.derivative(x) - np.exp(x))

        return errorderiv

    def _calculate_extrema(self):
        """This function calculates the extrema of the CRA error function.

        This function calculates the extrema of the CRA error function.
        A property of the CRA is the equioscillatory behaviour of the error
        function. Because the CRA is derived using the transformation to map
        $[-1,+1]$ on $(-\\infty, 0]$, the equioscillatory behaviour is visible
        in a logarithmic scale.

        The code uses a three step approach:
            1. Get some initial guesses from a logarithmic spread of starting
            values using fsolve on derivative of the error function
            2. Refine the solutions using arbitrary precision arithmetic
            3. Prune the results to be left with 2*order extrema

        Parameters
        ----------
        None

        Returns
        -------
        ndarray holding the extrema in double precision
        """

        dps = 100  # 100 is rather arbitray but should be sufficient
        with mp.workdps(dps):
            mppower = np.vectorize(mp.power)
            tofloat = lambda x: np.array(x, dtype=float)

            # Step 1: initial guesses
            startvals = -mppower(mp.mpf("10"), np.linspace(5, -5, 101))
            sols = np.zeros_like(startvals, dtype=float)
            for i, sv in enumerate(startvals):
                sol = fsolve(
                    lambda x: tofloat(self.errorderivative(x, dps)),
                    float(sv),
                    xtol=1e-5,
                    maxfev=100,
                )[0]
                if sol > -10000:  # arbitrary boundary
                    sols[i] = sol

            # Step 2: refine
            startvalsmp = np.unique(sols.round(decimals=10))[:-1]

            tol = mp.mpf("1e-20")
            delta = mp.mpf("2e-2")
            one = mp.mpf("1")

            extrema = []
            for sv in startvalsmp:
                svmp = mp.mpf(sv)
                left = svmp * (one - delta)
                right = svmp * (one + delta)
                if (
                    self.errorderivative(left, dps).item()
                    * self.errorderivative(right, dps).item()
                    < 0
                ):
                    refined = mp.findroot(
                        lambda x: self.errorderivative(x, dps),
                        (svmp * (one - delta), svmp * (one + delta)),
                        solver="anderson",
                        tol=tol,
                    )

                    extrema.append(refined)

            # Step 3: prune
            result = []
            for ext in extrema:
                present = False
                for r in result:
                    present = mp.almosteq(ext, r, rel_eps=mp.mpf("1e-14"))
                    if present:
                        break
                if not present:
                    result.append(float(ext))

        return np.array(result, dtype=float)

    @property
    def extrema(self):
        if self._extrema is None:
            self._extrema = self._calculate_extrema()
        return self._extrema

    @property
    def extremavals(self):
        if self._extremavals is None:
            self._extremavals = self.error(self.extrema)
        return self._extremavals

    def ODEsolver(A, t):
        print("to implement")


class CRAC:
    """class to thold a collection of Chebyshev Rational Approximations.

    This class can hold a collection of CRAs to the exponential function on the
    negative real axis, held in a CRA dataclass. It provides a
    constructor, an append function and a dump to and real from file using the
    dict representation of a dataclass.
    """

    def __init__(self, cras=None):
        """Initialize a CRA collection using a list of CRA items.

        Parameters
        ----------
        cras : :obj:`list` of :obj:`CRA`, optional
            list of instances of CRA to be added to the collection at
            its creation.
        """
        self._origins = dict()
        self._orders = dict()
        self.approx = list()

        if cras:
            if isinstance(cras, list):
                for cra in cras:
                    self.append(cra)
            elif isinstance(cras, CRA):
                self.append(cras)
            else:
                raise TypeError("CRAC requires CRA object or a list of CRA objects")

    @property
    def origins(self):
        """list: Returns a list of the different CRA origins in the
        collection."""
        return list(self._origins.keys())

    @property
    def orders(self):
        """list: Returns a list of the different CRA orders in the
        collection."""
        return list(self._orders.keys())

    @property
    def collection(self):
        """list[tuple]: Returns a list of all CRA in the collection

        Each item in the list is a tuple, (origin, order)
        """
        return [(a.origin, a.order) for a in self.approx]

    def append(self, cra):
        """None: Appends a CRA to the collection."""
        if isinstance(cra, CRA):
            origin = cra.origin
            order = cra.order

            # Add the new cra to the approx list
            self.approx.append(cra)

            # Add the origin of the case to the dictionary, with link to the object
            if origin in self._origins:
                self._origins[origin].append(cra)
            else:
                self._origins[origin] = [cra]

            # Add the order of the case to the dictionary, with link to the object
            if order in self._orders:
                self._orders[order].append(cra)
            else:
                self._orders[order] = [cra]

        else:
            raise TypeError("I can only handle objects of type CRA")

    def __call__(self, origin, order):
        """Function that returns a CRA from a certain origin and with a certain
        order

        Parameters
        ----------
        origin: str
            name of the origin (like Pusa, Calvin)
        order: int
            order of the CRA

        Returns
        -------
        Matching CRA
        """

        cras = self._origins[origin]

        res = None
        for cra in cras:
            if cra.order == order:
                res = cra
                break

        return res

    def __len__(self):
        """int: Return the number of items in the collection."""
        return len(self.approx)

    def tofile(self, fname, mode="w"):
        """None: Dump the collection to a text file.

        This function dumps the collection to a text file by converting the
        dataclass CRA instances to a corresponding `dict`.

        Parameters
        ----------
        fname: str
            Filename for the output file.
        mode: str, default 'w'
            Single character string, either 'w' (default) or 'a'
        """
        modestr = mode + "t"
        with open(fname, modestr) as fh:
            for cra in self.approx:
                fh.write(str(asdict(cra)) + "\n")

    def fromstring(self, crastr):
        """None: Read a collection from a text file.

        This function reads a collection from a text file by converting the
        dictionaries (separated by a newline) to instances of the dataclass
        CRA.

        Parameters
        ----------
        crastr: str
            string containing dicts of the CRAs
        """

        cralist = re.findall(r"\{[^}]*\}", crastr)

        for cra in cralist:
            self.append(CRA(**eval(cra)))

    def fromfile(self, fname):
        """None: Read a collection from a text file.

        This function reads a collection from a text file by converting the
        dictionaries (separated by a newline) to instances of the dataclass
        CRA.

        Parameters
        ----------
        fname: str
            Filename for the input file.
        """
        with open(fname, "rt") as fh:
            crastrs = fh.read()
            self.fromstring(crastrs)


class CRA_literature:
    """Class that only has a __call__ function to return the CRAs values from
    literature.
    """

    def __init__(self):
        self.cras_literature = CRAC()
        crastr = pkg_resources.read_text(data, "cras_literature.dat")
        self.cras_literature.fromstring(crastr)

    def __call__(self):
        return self.cras_literature


cras_literature = CRA_literature()()


def _fft(yg, inverse=False):
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
    A = ifac * mp.mpc(2j) * mp.pi * n * k / mp.mpf(N_min)
    M = exp(A)
    X = M.dot(yg.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, : X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2 :]
        A = (
            ifac
            * mp.mpc(1j)
            * mp.pi
            * np.array(mp.arange(X.shape[0]))
            / mp.mpf(X.shape[0])
        )
        factor = exp(A)[:, None]
        X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])
    # build-up each level of the recursive calculation all at once
    if inverse:
        return X.ravel() / mp.mpf(N)
    else:
        return X.ravel()


def CaratheodoryFejer(n, verbose=False, dps=30, K=75, nf=1024):
    """Wrapper around mpCaratheodoryFejer to return ndarrays of complex type.

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


def mpCaratheodoryFejer(n, verbose=False, dps=30, K=75, nf=1024):
    """Calculates the best rational approxmation to exp(x) on negative real
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

        nf = mp.mpf(nf)

        twopij = mp.mpc("0", "2") * mp.pi
        one = mp.mpf("1")
        w = mpexp(twopij / nf * np.array(mp.arange(0, nf)))
        t = mpreal(w)
        scale = mp.mpf("9")
        if verbose:
            print("1. Calculating Chebyshev nodes")
        F = mpexp(scale * (t - one) / (t + one + mp.mp.eps))
        c = mpreal(_fft(F)) / nf
        if verbose:
            print("2. Building Hankel matrix")
        f = polyval(c[: K + 1][::-1], w)
        h = hankel(c[1 : K + 1])
        if verbose:
            print("3. Do SVD")
        U, S, V = mp.svd_r(mp.matrix(h), full_matrices=True, compute_uv=True)
        s = S[n]
        u = U[::-1, n]
        v = V[n, :]
        if verbose:
            print("4. Do FFTs")
        zz = int(nf - K) * [mp.mpf("0")]
        b = _fft(np.concatenate((u, zz))) / _fft(np.concatenate((v, zz)))
        rt = f - s * w ** K * b
        rtc = mpreal(_fft(rt)) / nf
        if verbose:
            print("5. Start root finding")
        zr = np.array(mp.polyroots(v, maxsteps=20000))
        qk = zr[mpfabs(zr) > one]
        qc = np.poly(qk)
        pt = rt * polyval(qc, w)
        ptc1 = mpreal(_fft(pt) / nf)
        ptc = ptc1[n::-1]
        ck = 0 * qk
        if verbose:
            print("6. Start poles/residu")
        for k in range(n):
            q = qk[k]
            q2 = np.poly(qk[qk != q])
            ck[k] = np.polyval(ptc, q) / polyval(q2, q)
        zk = scale * (qk - one) ** 2.0 / (qk + one) ** 2
        ck = mp.mpf("4") * ck * zk / (qk ** 2 - one)
        idx = np.argsort(mpimag(zk))
        zk = zk[idx]
        ck = ck[idx]

        rinf = mpreal(one / mp.mpf("2") * (one + np.sum(ck / zk))).item()

    return zk, ck, rinf
