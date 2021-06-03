# -*- coding: utf-8 -*-
"""
This module provides a data structure to hold a Chebyshev Rational
Approximation (CRA) for the exponential function.
and a data structure to keep a collection of ChebRatExps, ChebRatExpCollection, that
can be accessed by origin and order of the approximation
"""


import numpy as np
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ChebRatExp:
    """ unmutable data class to hold information on a CRA

        Attributes
        ----------
        origin: str
            The origin of the approximation, for example "Pusa"
        order: int
            The order of the approximation
        rinf: float
            The absolute error of the approximation at $-\infty$
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


class ChebRatExpCollection:
    """ class to thold a collection of Chebyshev Rational Approximations.

        This class can hold a collection of CRAs to the exponential function on the
        negative real axis, held in a ChebRatExp dataclass. It provides a
        constructor, an append function and a dump to and real from file using the
        dict representation of a dataclass.
    """

    def __init__(self, cras=None):
        """ Initialize a CRA collection using a list of ChebRatExp items.

        Parameters
        ----------
        cras : :obj:`list` of :obj:`ChebRatExp`, optional
            list of instances of ChebRatExp to be added to the collection at
            its creation.
        """
        self._origins = dict()
        self._orders = dict()
        self._approx = list()

        if cras:
            for cra in cras:
                self.append(cra)

    @property
    def origins(self):
        """ list: Returns a list of the different CRA origins in the
        collection."""
        return list(self._origins.keys())

    @property
    def orders(self):
        """ list: Returns a list of the different CRA orders in the
        collection."""
        return list(self._orders.keys())

    def append(self, cra):
        """ None: Appends a ChebRatExp to the collection."""
        if isinstance(cra, ChebRatExp):
            origin = cra.origin
            order = cra.order

            # Add the new cra to the _approx list
            self._approx.append(cra)

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
            raise TypeError("I can only handle objects of type ChebRatExp")

    def __len__(self):
        """ int: Return the number of items in the collection."""
        return len(self._approx)

    def tofile(self, fname, mode="w"):
        """ None: Dump the collection to a text file.

        This function dumps the collection to a text file by converting the
        dataclass ChebRatExp instances to a corresponding `dict`.

        Parameters
        ----------
        fname: str
            Filename for the output file.
        mode: str, default 'w'
            Single character string, either 'w' (default) or 'a'
        """
        modestr = mode + "t"
        with open(fname, modestr) as fh:
            for cra in self._approx:
                fh.write(str(asdict(cra)) + "\n")

    def fromfile(self, fname):
        """ None: Read a collection from a text file.

        This function reads a collection from a text file by converting the
        dictionaries (separated by a newline) to instances of the dataclass
        ChebRatExp.

        Parameters
        ----------
        fname: str
            Filename for the input file.
        """
        with open(fname, "rt") as fh:
            crastrs = fh.readlines()
            for crastr in crastrs:
                self.append(ChebRatExp(**eval(crastr)))
