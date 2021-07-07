# -*- coding: utf-8 -*-
from bisect import bisect_left
import numpy as np
import mpmath as mp
from scipy.sparse import dok_matrix, csr_matrix
from .tools import lazyreadonlyproperty
from deepburn.isotope import Isotope


class Transitions:
    """Object that holds a list of isotopic transitions."""

    def __init__(self):
        self._trans = list()
        self._isotopes = set()

    def add_transition(self, fromiso, rate, toiso=None):
        self._trans.append((fromiso, rate, toiso))
        self._isotopes.add(fromiso)

        if toiso:
            self._isotopes.add(toiso)

        return self

    def __call__(self):
        return self._trans

    @property
    def isotopes(self):
        return list(self._isotopes)

    @isotopes.setter
    def isotopes(self, dummy):
        raise ValueError("Read only value")


class IsotopicComposition:
    """Object that holds an isotopic composition"""

    def __init__(self):
        self._ics = dict()

    def add_value(self, iso, value):
        self._ics[iso] = value
        return self

    def __call__(self, isotope_list):
        ics = np.zeros_like(isotope_list, dtype=float)

        for ic_iso, ic_value in self._ics.items():
            idx = isotope_list.index(ic_iso)
            ics[idx] = ic_value

        return ics


class BUP:
    """Objects of the BurnUpProblem class hold depletion problems.

    Attributes
    ----------
    matrix: scipy.sparse.dok_matrix
        sparse matrix holding the transition matrix
    initial_condition: scipy.sparse.dok_matrix
        sparse vector holding the initial condition values
    isotopes: list of Isotope
        list that maps the rows/columns of the transition matrix to actual
        isotopes
    time_stamps: list of float
        list that contains the time stamps at which solution of the burnup
        problem is required
    ref_sols: dictionary
        dictionary that maps a reference solution (dok_matrix) to the index of
        the time stamp in time_stamps. If a reference solution is provided,
        there has to be an associated time stamp. But not all time_stamps
        should have a reference solution

    """

    def __init__(
        self,
        matrix=None,
        initial_condition=None,
        isotopes=[],
        time_stamps=[],
        ref_sols={},
        name="deepburn problem",
    ):
        """Initialize a BurnUpProblem"""
        sizem, sizen = matrix.shape

        if sizem != sizen:
            raise ValueError("System matrix has to be square")

        if isinstance(initial_condition, list):
            inisize = len(initial_condition)
        else:
            inisize = initial_condition.shape[0]

        if inisize != sizem:
            raise ValueError(
                "Size of initial condition vector not compatible with system matrix size"
            )

        self._matrix = matrix.tocsr()

        self._initial_condition = np.array(initial_condition)

        # self._initial_condition = dok_matrix((inisize[0], 1), dtype=float)
        # for i in range(inisize[0]):
        # self._initial_condition[i, 0] = initial_condition[i]
        # self._initial_condition = self._initial_condition.tocsr()

        self._isotopes = isotopes
        self._time_stamps = time_stamps

        self._ref_sols = dict()

        for ref_time, ref_sol in ref_sols.items():
            self.add_reference_solution(ref_time, ref_sol)

        self.name = name

    @classmethod
    def fromTransitions(
        cls,
        trans=None,
        initial_condition=None,
        time_stamps=[],
        ref_sols={},
        name="deepburn problem",
    ):
        """Classmethod to create a BUP from a set of transitions and initial
        conditions.
        Reorders the isotopes in increasing order and reorders the initial
        condition accordingly.
        """

        problem_size = len(trans.isotopes)

        permutation = np.argsort(trans.isotopes)
        sorted_isotopes = np.take(trans.isotopes, permutation).tolist()

        dok = dok_matrix((problem_size, problem_size))

        for t in trans():
            fromisoidx = bisect_left(sorted_isotopes, t[0])
            dok[fromisoidx, fromisoidx] -= t[1]

            if t[2]:
                toisoidx = bisect_left(sorted_isotopes, t[2])
                dok[toisoidx, fromisoidx] += t[1]

        matrix = dok.tocsr()

        sorted_initial_condition = initial_condition(sorted_isotopes)

        for timestamp, isocomp in ref_sols.items():
            ref_sols[timestamp] = isocomp(sorted_isotopes)

        return cls(
            matrix,
            sorted_initial_condition,
            sorted_isotopes,
            time_stamps,
            ref_sols,
            name,
        )

    def __call__(self, t, tol=1e-12):
        """Find a reference solution"""
        raise ValueError("Not implemented")

    @property
    def size(self):
        """Returns the size of the problem matrix"""
        return self._initial_condition.shape[0]

    @size.setter
    def size(self, newsize):
        raise ValueError("Sorry, read-only property")

    @property
    def initial_condition(self):
        return self._initial_condition

    @initial_condition.setter
    def initial_condition(self, initial_condition):

        if isinstance(initial_condition, list):
            inisize = len(initial_condition)
        else:
            inisize = initial_condition.shape[0]

        if inisize[0] != self.size:
            raise ValueError(
                "Size of inital condition vector not compatible with system matrix size"
            )

        self._initial_condition = np.array(initial_condition)

        # self._initial_condition = dok_matrix((inisize[0], 1), dtype=float)
        # for i in range(inisize[0]):
        # self._initial_condition[i, 0] = initial_condition[i]
        # self._initial_condition.tocsr()
        return self

    @property
    def sparsematrix(self):
        return self._matrix

    @sparsematrix.setter
    def sparsematrix(self, dummy):
        raise ValueError("Sorry. read-only property")

    @property
    def densematrix(self):
        fullmatrix = self._matrix.toarray()
        return fullmatrix

    @densematrix.setter
    def densematrix(self, dummy):
        raise ValueError("Sorry. read-only property")

    @property
    def time_stamps(self):
        """Returns a copy of the time_stamps for inspection."""
        return list(self._time_stamps)

    @time_stamps.setter
    def time_stamps(self, time_stamps):
        """Sets the time stamps, ensuring sorted and uniqueness."""
        self._time_stamps = sorted(set(time_stamps))
        return self

    def add_time_stamps(self, time_stamps):
        """Add time stamps to the list if it is not yet present.

        Args:
            time_stamps (list): list of time stamps to be added

        Returns:
            postion (int): the position in the time stamp list where it was
            added
        """
        for time_stamp in time_stamps:
            position = None
            idx = bisect_left(self._time_stamps, time_stamp)
            if self._time_stamps[idx] != time_stamp:
                self._time_stamps.insert(idx, time_stamp)
        return idx

    def remove_time_stamps(self, time_stamps):
        """Remove time stamp from the list based on value.

        Args:
            time_stamps (list): list of time stamps to be removed
        """
        for time_stamp in time_stamps:
            self._time_stamps.remove(time_stamp)

    def remove_time_stamps_idx(self, ts_indices):
        """Remove time stamp from the list based on index.

        Args:
            ts_indices (list): list of indices of time stamps to be removed
        """
        for idx in ts_indices:
            del self._time_stamps[idx]

    def add_reference_solution(self, time, ref_sol):
        """Add a reference solution to the problem at a time stamp ``time''

        Args:
            time (float): time at which the reference solution is valid
            ref_sol (numpy array or list): array containing the reference
            solution
        """

        sz = ref_sol.shape

        if sz[0] != self.size or len(sz) > 1:
            raise ValueError(
                f"Reference solution should have same dimensions as"
                f"problem size. Got {szi} x {szj}."
                f"Expected {self.size}"
            )

        idx = self.add_time_stamps([time])

        self._ref_sols[idx] = ref_sol

    @property
    def ref_sols(self):
        return self._ref_sols

    @ref_sols.setter
    def ref_sols(self, ref_sols):
        raise ValueError("Read only property. Use methods")

    @property
    def isotopes(self):
        return self._isotopes

    @isotopes.setter
    def isotopes(self, isotopes):
        # """Add a list of Isotopes for pretty printing and interpretation

        # Args:
        # isoptopes (list of Isotopes): list of isotopes
        # """
        # if len(isotopes) != self.size:
        # raise ValueError(
        # f"Length of list of Isotopes ({len(isotopes)} is not conforming to the size of the burnup problem ({self._size})"
        # )

        # self._isotopes = list(isotopes)
        raise ValueError(
            "Read-only property! Isotopes should be set at object construction"
        )

    def __str__(self):
        res = self.name + "\n"
        res += "Isotopes: "
        for iso in self._isotopes:
            res += f"{iso} "
        res += "\nTransition matrix\n"
        res += f"{self._matrix}"
        res += "\nInitial condition\n"
        res += f"{self._initial_condition}"
        res += "\nTime stamps\n"
        for idx, time in enumerate(self._time_stamps):
            res += f"{time}"
            if idx in self._ref_sols.keys():
                res += f"    {self._ref_sols[idx]}"
            else:
                res += "     No reference solution provided"
            res += "\n"
        return res


def Polonium():
    isotopes = [Isotope("Bi209"), Isotope("Bi210"), Isotope("Po210")]

    trans = Transitions()
    trans.add_transition(isotopes[0], 1.83163e-12, isotopes[1])
    trans.add_transition(isotopes[1], 1.60035e-6, isotopes[2])
    trans.add_transition(isotopes[2], 5.79764e-8)

    ICs = IsotopicComposition()
    ICs.add_value(isotopes[0], 6.95896e-4)

    t = [20 * 24 * 3600, 180 * 24 * 3600]
    rst = 180 * 24 * 3600
    rs1 = IsotopicComposition()
    rs1.add_value(isotopes[0], 6.9587617733003087e-04)
    rs1.add_value(isotopes[1], 7.451824950503656e-09)
    rs1.add_value(isotopes[2], 1.2725788327617256e-08)

    ref_sols = {rst: rs1}

    pol = BUP.fromTransitions(trans, ICs, t, ref_sols, "Po210")

    return pol


def LagoRahnema_1():
    with mp.workdps(1000):
        isotopes = [Isotope("U238"), Isotope("Th-234")]

        trans = Transitions()

        lambda1 = mp.log(mp.mpf("2")) / mp.mpf("1.4099935680e+17")
        lambda2 = mp.log(mp.mpf("2")) / mp.mpf("2.082240e+06")

        trans.add_transition(isotopes[0], float(lambda1), isotopes[1])
        trans.add_transition(isotopes[1], float(lambda2))

        ICs = IsotopicComposition()
        ICs.add_value(isotopes[0], 1e10)

        t = [5e17]

        # Analytical
        N1t = mp.mpf("1e10") * mp.exp(-lambda1 * mp.mpf("5e17"))
        N2t = (
            lambda1
            / (lambda2 - lambda1)
            * mp.mpf("1e10")
            * (mp.exp(-lambda1 * mp.mpf("5e17")) - mp.exp(-lambda2 * mp.mpf("5e17")))
        )

        rs = IsotopicComposition()
        rs.add_value(isotopes[0], float(N1t))
        rs.add_value(isotopes[1], float(N2t))

        ref_sols = {5e17: rs}

    return BUP.fromTransitions(trans, ICs, t, ref_sols, "Lago & Rahnema #1 2017")


def LagoRahnema_2():
    with mp.workdps(1000):
        isotopes = [Isotope("Np237"), Isotope("Pa233"), Isotope("U233")]

        trans = Transitions()
        lambda1 = mp.log(mp.mpf("2")) / mp.mpf("6.7659494310E+13")
        lambda2 = mp.log(mp.mpf("2")) / mp.mpf("2.330640E+06")
        lambda3 = mp.log(mp.mpf("2")) / mp.mpf("5.023969920E+12")

        trans.add_transition(isotopes[0], lambda1, isotopes[1])
        trans.add_transition(isotopes[1], lambda2, isotopes[2])
        trans.add_transition(isotopes[2], lambda3)

        ICs = IsotopicComposition()
        ICs.add_value(isotopes[0], 1e12)

        t = mp.mpf("1e12")

        # Analyticak
        N1t = mp.mpf("1e12") * mp.exp(-lambda1 * mp.mpf("1e12"))
        N2t = (
            -lambda1
            * mp.mpf("1e12")
            * (mp.exp(-lambda1 * t) - mp.exp(-lambda2 * t))
            / (lambda1 - lambda2)
        )

        N3t = (
            -mp.exp(-lambda3 * t)
            * (
                (-lambda2 + lambda3) * mp.exp(-t * (lambda1 - lambda3))
                + (lambda1 - lambda3) * mp.exp(-t * (lambda2 - lambda3))
                - lambda1
                + lambda2
            )
            * mp.mpf("1e12")
            * lambda2
            * lambda1
            / (lambda2 - lambda3)
            / (lambda1 - lambda3)
            / (lambda1 - lambda2)
        )

        rs = IsotopicComposition()
        rs.add_value(isotopes[0], float(N1t))
        rs.add_value(isotopes[1], float(N2t))
        rs.add_value(isotopes[2], float(N3t))

        ref_sols = {1e12: rs}

    return BUP.fromTransitions(
        trans, ICs, [float(t)], ref_sols, "Lago & Rahnema #2 2017"
    )


def LagoRahnema_3():
    with mp.workdps(1000):
        isotopes = [
            Isotope("Pb211"),
            Isotope("Bi211"),
            Isotope("Tl207"),
            Isotope("Pb207"),
        ]

        trans = Transitions()
        lambdas = []
        lambdas.append(mp.log(mp.mpf("2")) / mp.mpf("2.1660E+03"))
        lambdas.append(mp.log(mp.mpf("2")) / mp.mpf("1.2840E+02"))
        lambdas.append(mp.log(mp.mpf("2")) / mp.mpf("2.8620E+02"))

        for i in range(3):
            trans.add_transition(isotopes[i], lambdas[i], isotopes[i + 1])

        ICs = IsotopicComposition()
        ICs.add_value(isotopes[0], 1.00e10)
        ICs.add_value(isotopes[1], 1.00e04)
        ICs.add_value(isotopes[2], 1.00e01)

        t = mp.mpf("1e4")

        # Analytical solution (code generated by Maple)
        N10 = mp.mpf("1e10")
        N20 = mp.mpf("1e4")
        N30 = mp.mpf("1e1")
        N40 = mp.mpf("0")

        N1t = N10 * mp.exp(-lambdas[0] * t)

        N2t = (
            ((N10 + N20) * lambdas[0] - N20 * lambdas[1]) * mp.exp(-lambdas[1] * t)
            - lambdas[0] * mp.exp(-lambdas[0] * t) * N10
        ) / (lambdas[0] - lambdas[1])

        N3t = (
            -mp.exp(-lambdas[2] * t)
            * (
                lambdas[1]
                * (lambdas[0] - lambdas[2])
                * ((N10 + N20) * lambdas[0] - N20 * lambdas[1])
                * mp.exp(-t * (lambdas[1] - lambdas[2]))
                - lambdas[0]
                * lambdas[1]
                * N10
                * (lambdas[1] - lambdas[2])
                * mp.exp(-t * (lambdas[0] - lambdas[2]))
                - (
                    ((N10 + N20 + N30) * lambdas[0] - lambdas[2] * (N20 + N30))
                    * lambdas[1]
                    - lambdas[2] * N30 * (lambdas[0] - lambdas[2])
                )
                * (lambdas[0] - lambdas[1])
            )
            / (lambdas[1] - lambdas[2])
            / (lambdas[0] - lambdas[2])
            / (lambdas[0] - lambdas[1])
        )

        N4t = (
            (
                -(
                    ((N10 + N20 + N30) * lambdas[1] - lambdas[2] * N30) * lambdas[0]
                    - ((N20 + N30) * lambdas[1] - lambdas[2] * N30) * lambdas[2]
                )
                * (lambdas[0] - lambdas[1])
                * mp.exp(-lambdas[2] * t)
                + lambdas[2]
                * (lambdas[0] - lambdas[2])
                * ((N10 + N20) * lambdas[0] - N20 * lambdas[1])
                * mp.exp(-lambdas[1] * t)
                + (lambdas[1] - lambdas[2])
                * (
                    -lambdas[2] * lambdas[1] * N10 * mp.exp(-lambdas[0] * t)
                    + (N10 + N20 + N30 + N40)
                    * (lambdas[0] - lambdas[2])
                    * (lambdas[0] - lambdas[1])
                )
            )
            / (lambdas[1] - lambdas[2])
            / (lambdas[0] - lambdas[2])
            / (lambdas[0] - lambdas[1])
        )

        rs = IsotopicComposition()
        rs.add_value(isotopes[0], float(N1t))
        rs.add_value(isotopes[1], float(N2t))
        rs.add_value(isotopes[2], float(N3t))
        rs.add_value(isotopes[3], float(N4t))

        ref_sols = {float(t): rs}

    return BUP.fromTransitions(
        trans, ICs, [float(t)], ref_sols, "Lago & Rahnema #3 2017"
    )

