# -*- coding: utf-8 -*-
from bisect import bisect
import numpy as np
from scipy.sparse import dok_matrix
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
        """Property that returns the transition matrix in dok_matrix format"""

        problem_size = len(self._isotopes)
        isolist = sorted(list(self._isotopes))
        dok = dok_matrix((problem_size, problem_size))

        for t in self._trans:
            fromisoidx = bisect(t[0], isolist)
            dok[fromisoidx, fromisoidx] -= t[1]

            if t[2]:
                toisoidx = bisect(t[2], isolist)
                dok[fromisoidx, fromisoidx] += t[1]

        return dok

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
        refsols={},
        name="deepburn problem",
    ):
        """Initialize a BurnUpProblem"""
        sizem, sizen = matrix.shape

        if sizem != sizen:
            raise ValueError("System matrix has to be square")

        (inisize,) = initial_condition.shape

        if inisize != sizem:
            raise ValueError(
                "Size of inital condition vector not compatible with system matrix size"
            )

        self._dok_matrix = matrix
        self._initial_condition = initial_condition

        self._isotopes = isotopes
        self._time_stamps = time_stamps
        self._refsols = {}
        self._name = name

    def __cal__(self, t, tol=1e-12):
        """Find a reference solution"""

    @property
    def size(self):
        """Returns the size of the problem matrix"""
        (size,) = self.initial_condition.shape
        return size

    @size.setter
    def size(self, newsize):
        raise ValueError("Sorry, read-only property")

    @property
    def sparsematrix(self):
        cscmatrix = self.dok_matrix.tocsc()
        return cscmatrix

    @sparsematrix.setter
    def sparsematrix(self, dummy):
        raise ValueError("Sorry. read-only property")

    @property
    def densematrix(self):
        fullmatrix = self.dok_matrix.toarray()
        return fullmatrix

    @densematrix.setter
    def densematrix(self, dummy):
        raise ValueError("Sorry. read-only property")

    @property
    def time_stamps(self):
        """Returns a copy of the timestamps for inspection."""
        return list(self.time_stamps)

    @time_stamps.setter
    def time_stamps(self, time_stamps):
        """Sets the time stamps, ensuring sorted and uniqueness."""
        self._time_stamps = sorted(set(time_stamps))

    def AddTimeStamps(self, time_stamps):
        """Add time stamps to the list if it is not yet present.

        Args:
            time_stamps (list): list of time stamps to be added

        Returns:
            postion (int): the position in the time stamp list where it was
            added
        """
        for time_stamp in time_stamps:
            position = None
            idx = bisect(self._time_stamps, time_stamp)
            try:
                if self._time_stamps[idx] != time_stamp:
                    self._time_stamps.insert(idx, time_stamp)
                    position = idx
            except ValueError:
                self._time_stamps.insert(idx, time_stamp)
                position = idx

            return position

    def RemoveTimeStamps(self, time_stamps):
        """Remove time stamp from the list based on value.

        Args:
            time_stamps (list): list of time stamps to be removed
        """
        for time_stamp in time_stamps:
            self._time_stamps.remove(time_stamp)

    def RemoveTimeStampsIdx(self, ts_indices):
        """Remove time stamp from the list based on index.

        Args:
            ts_indices (list): list of indices of time stamps to be removed
        """
        for idx in ts_indices:
            del self._time_stamps[idx]

    def AddReferenceSolution(self, time, refsol):
        """Add a reference solution to the problem at a time stamp ``time''

        Args:
            time (float): time at which the reference solution is valid
            refsol (numpy array or list): array containing the reference
            solution
        """
        if len(refsol) != self.size:
            raise ValueError(
                f"Reference solution should have same dimensions as"
                f"problem size. Got {len(refsol)}."
                f"Expected {self.size}"
            )

        idx = self.AddTimeStamps(time)

        self._refsols[idx] = np.asarray(refsol)

    @property
    def isotopes(self):
        return self._isotopes

    @isotopes.setter
    def isotopes(self, isotopes):
        """Add a list of Isotopes for pretty printing and interpretation

        Args:
            isoptopes (list of Isotopes): list of isotopes
        """
        if len(isotopes) != self._size:
            raise ValueError(
                f"Length of list of Isotopes ({len(isotopes)} is not conforming to the size of the burnup problem ({self._size})"
            )

        self._isotopes = list(isotopes)


def Polonium():
    lbi209 = 1.83163e-12
    lbi210 = 1.60035e-6
    lpo210 = 5.79764e-8

    val = [-lbi209, lbi209, -lbi210, lbi210, -lpo210]
    N0 = [6.95896e-4, 0, 0]
    t = [90 * 24 * 3600, 135 * 24 * 3600, 180 * 24 * 3600]

    refsols = {
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

    isotopes = [Isotope("Bi209"), Isotope("Bi210"), Isotope("Po210")]

    pol = BurnUpProblem(
        3, (iindex, jindex, val), N0, t, name="Polonium-210", refsols=refsols
    )
    return pol
