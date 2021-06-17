# -*- coding: utf-8 -*-

""" Tests for the `CRA` module."""

import pytest
from pytest import approx

import numpy as np
import scipy.sparse as sp

from deepburn.CRAM import CRA, CRAC, cras_literature, CRA_ODEsolver

def test_init():
    crasolver = CRA_ODEsolver()
    assert isinstance(crasolver._cra, CRA)

def test_basicsolve_zerodim():
    A = np.zeros((0,0))
    N0 = np.zeros(0)
    crasolver = CRA_ODEsolver()
    N = crasolver._solveCRA(A, N0)

def test_basicsolve_trivial():
    A = np.asarray([[-1.0]])
    N0 = np.asarray([1.0])
    crasolver = CRA_ODEsolver()
    N = crasolver._solveCRA(A, N0)
    assert N == approx(0.367879441)

def test_basicsolve_polonium_dense():
    oneY = 365.25*24*3600
    A = np.array([[-1.83163e-12,0,0],[+1.83163e-12,-1.60035e-6,0],[0,+1.60035e-6,-5.79764e-8]]) * oneY
    N0 = np.array([6.95896e-4,0,0])
    crasolver = CRA_ODEsolver()
    Y = crasolver._solveCRA(A, N0)
    print(Y)
    assert Y[0] == approx(6.958557771e-04)
    assert Y[1] == approx(7.964206428e-10)
    assert Y[2] == approx(1.832378200e-08)

def test_basicsolve_polonium_sparse():
    oneY = 365.25*24*3600
    A = np.array([[-1.83163e-12,0,0],[+1.83163e-12,-1.60035e-6,0],[0,+1.60035e-6,-5.79764e-8]]) * oneY
    Asp = sp.csc_matrix(A)
    N0 = np.array([6.95896e-4,0,0])
    crasolver = CRA_ODEsolver()
    Y = crasolver._solveCRA(Asp, N0)
    assert Y[0] == approx(6.9585577708845012e-04)
    assert Y[1] == approx(7.9642064281967071e-10)
    assert Y[2] == approx(1.8323781965287107e-08)
