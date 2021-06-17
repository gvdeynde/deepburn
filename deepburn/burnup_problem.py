# -*- coding: utf-8 -*-
import bisect
import numpy as np
from scipy.sparse import dok_matrix
from .tools import lazyreadonlyproperty
import isotope as iso


class BurnUpProblem:
    """Objects of the BurnUpProblem class hold depletion problems.

    Attributes:

    """

    def __init__(
        self,
        size=0,
        matrix_data=([], [], []),
        initial_condition=[],
        time_stamps=[],
        name="DeepBurnProblem",
        isotopes=[],
        refsols={},
    ):
        """Initializes a DeepBurn problem

        Args:
            name (str): name of the problem
            size (int): size of the transition matrix
            matrix_data (tuple of lists): tuple of transition matrix elements
            using three lists: row_index, col_index, value
            initial_condition (list): list of initial concentrations
            time_stamps (list): list of time stamps at which solution should be
            computed
        """
        self.name = name
        self._size = size
        self.dokmatrix = None

        if size > 0:
            self.dokmatrix = dok_matrix((size, size), dtype=np.float)

        row_idx = matrix_data[0]
        col_idx = matrix_data[1]
        value = matrix_data[2]

        if row_idx and col_idx and value:
            if size == 0:
                size = np.max([np.max(row_idx), np.max(col_idx)])
                self._size = size
                self.dokmatrix = dok_matrix((size, size), dtype=np.float)

            for (i, j, v) in zip(row_idx, col_idx, value):
                self.dokmatrix[i, j] = v

        if self._size != len(initial_condition):
            raise ValueError(
                "Size of matrix doesn't correpsond to length of initial condition vector"
            )

        self.initial_condition = initial_condition
        self.time_stamps = time_stamps

        self._reference_solutions = refsols

        self._isotopes = isotopes

    @property
    def size(self):
        """Returns the size of the problem matrix"""
        return self._size

    @size.setter
    def size(self, newsize):
        """Sets the size of the problem matrix

        Args:
            newsize (int): the new size of the problem. It cannot be smaller
            than the size before (we cannot shrink problems). If the size is
            larger than before, extra zero rows and columns are added to the
            matrix
        """
        if newsize < self._size:
            raise ValueError("I cannot shrink a problem")
        else:
            self._size = newsize
            if self.dokmatrix:
                self.dokmatrix.resize(newsize, newsize)
            else:
                self.dokmatrix = dok_matrix((newsize, newsize), dtype=np.float)

    @property
    def sparsematrix(self):
        cscmatrix = self.dokmatrix.tocsc()
        return cscmatrix

    @sparsematrix.setter
    def sparsematrix(self, dummy):
        raise ValueError("Sorry. Read-only property")

    @property
    def densematrix(self):
        fullmatrix = self.dokmatrix.toarray()
        return fullmatrix

    @densematrix.setter
    def densematrix(self, dummy):
        raise ValueError("Sorry. Read-only property")

    @property
    def time_stamps(self):
        """Returns a copy of the timestamps for inspection."""
        return list(self._time_stamps)

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
            idx = bisect.bisect(self._time_stamps, time_stamp)
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

        self._reference_solutions[idx] = np.array(refsol)

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
    iindex = [0, 1, 1, 2, 2]
    jindex = [0, 0, 1, 1, 2]
    lbi209 = 1.83163e-12
    lbi210 = 1.60035e-6
    lpo210 = 5.79764e-8
    val = [-lbi209, lbi209, -lbi210, lbi210, -lpo210]
    N0 = [6.95896e-4, 0, 0]
    t = 90 * 24 * 3600
    Nref = [0.0006958860886, 7.964521967e-10, 7.451824964e-9]
    isotopes = [iso.Isotope("Bi209"), iso.Isotope("Bi210"), iso.Isotope("Po210")]

    pol = BurnUpProblem(
        3, (iindex, jindex, val), t, name="Polonium-210", refsols={t: Nref}
    )
    return pol
