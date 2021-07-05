# -*- coding: utf-8 -*-
from bisect import bisect_left
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from .tools import lazyreadonlyproperty
from deepburn.isotope import Isotope


class Transitions:
    """Object that holds all transitions.
    Generates a transition matrix for use in a BurnUpProblem
    """

    def __init__(self):
        self._trans = list()
        self._isotopes = set()

    def add_transition(self, fromiso, rate, toiso=None):
        self._isotopes = self._isotopes.union({fromiso})

        if toiso:
            self._isotopes = self._isotopes.union({toiso})

        self._trans.append((fromiso, rate, toiso))

        return self

    @property
    def transition_matrix(self):
        """Property that returns the transition matrix in csr_matrix format"""

        problem_size = len(self._isotopes)
        isolist = sorted(list(self._isotopes))
        dok = dok_matrix((problem_size, problem_size))

        for t in self._trans:
            fromisoidx = bisect_left(isolist, t[0])
            dok[fromisoidx, fromisoidx] -= t[1]

            if t[2]:
                toisoidx = bisect_left(isolist, t[2])
                dok[toisoidx, fromisoidx] += t[1]

        return dok.tocsr()

    @transition_matrix.setter
    def transition_matrix(self):
        """Read-only property"""
        raise ValueError("transition_matrix is a readonly property")


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
                "Size of inital condition vector not compatible with system matrix size"
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
        isotopes=[],
        time_stamps=[],
        ref_sols={},
        name="deepburn problem",
    ):
        """Classmethod to convert a Transitions object to a matrix object"""
        return cls(
            trans.transition_matrix,
            initial_condition,
            isotopes,
            time_stamps,
            ref_sols,
            name,
        )

    def __cal__(self, t, tol=1e-12):
        """Find a reference solution"""

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

        self._ref_sols[idx] = np.asarray(ref_sol)

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
        """Add a list of Isotopes for pretty printing and interpretation

        Args:
            isoptopes (list of Isotopes): list of isotopes
        """
        if len(isotopes) != self.size:
            raise ValueError(
                f"Length of list of Isotopes ({len(isotopes)} is not conforming to the size of the burnup problem ({self._size})"
            )

        self._isotopes = list(isotopes)

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

    N0 = [6.95896e-4, 0, 0]
    t = [20 * 24 * 3600, 180 * 24 * 3600]

    ref_sols = {
        90
        * 24
        * 3600: np.array(
            [6.958860885944331e-04, 7.964521968307441e-10, 7.9644399128560947e-10]
        ),
        180
        * 24
        * 3600: np.array(
            [6.9587617733003087e-04, 7.451824950503656e-09, 1.2725788327617256e-08]
        ),
    }

    pol = BUP.fromTransitions(trans, N0, isotopes, t, ref_sols, "Po210")

    return pol
