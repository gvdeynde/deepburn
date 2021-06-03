# -*- coding: utf-8 -*-

""" Tests for the `ChebRatExp` module."""

import pytest
from pytest import approx

from deepburn.CRAM.ChebRatExp import ChebRatExp, ChebRatExpCollection


def test_ChebRatExp_init():
    c = ChebRatExp(
        "CFT",
        3,
        -0.0008014597072714745,
        [0.691122195041816 - 0.043143728357214896j, -1.4837496454512307 + 0.0j],
        [0.1981697296161005 - 2.410667766188531j, 1.3688034212107467 + 0.0j],
    )
    assert isinstance(c, ChebRatExp)
    assert c.origin == "CFT"
    assert c.order == 3
    assert c.rinf == approx(-0.000801459)


def test_ChebRatExpCollection_defaultinit():
    col = ChebRatExpCollection()
    assert isinstance(col, ChebRatExpCollection)


def test_ChebRatExpCollection_init_onecra():
    c1 = ChebRatExp(
        "CFT",
        3,
        -0.0008014597072714745,
        [0.691122195041816 - 0.043143728357214896j, -1.4837496454512307 + 0.0j],
        [0.1981697296161005 - 2.410667766188531j, 1.3688034212107467 + 0.0j],
    )

    col = ChebRatExpCollection(c1)

    assert col.approx[0] is c1


def test_ChebRatExpCollection_init_cras():
    c1 = ChebRatExp(
        "CFT",
        3,
        -0.0008014597072714745,
        [0.691122195041816 - 0.043143728357214896j, -1.4837496454512307 + 0.0j],
        [0.1981697296161005 - 2.410667766188531j, 1.3688034212107467 + 0.0j],
    )

    c2 = ChebRatExp("CFT", 2, -1, [1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j])

    c3 = ChebRatExp("Dummy", 3, -2, [2 + 1j, 1 + 2j], [4 + 3j, 3 + 4j])

    col = ChebRatExpCollection([c1, c2, c3])

    assert col.approx[0] is c1
    assert col.approx[1] is c2
    assert col.approx[2] is c3

    assert len(col) == 3

    assert set(col.origins) == set(["CFT", "Dummy"])
    assert set(col.orders) == set([3, 2])
