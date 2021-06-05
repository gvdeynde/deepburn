# -*- coding: utf-8 -*-

""" Tests for the `CRA` module."""

import pytest
from pytest import approx

from deepburn.CRAM import CRA, CRAC


def test_CRA_init():
    c = CRA(
        "CFT",
        3,
        -0.0008014597072714745,
        [0.691122195041816 - 0.043143728357214896j, -1.4837496454512307 + 0.0j],
        [0.1981697296161005 - 2.410667766188531j, 1.3688034212107467 + 0.0j],
    )
    assert isinstance(c, CRA)
    assert c.origin == "CFT"
    assert c.order == 3
    assert c.rinf == approx(-0.000801459)


def test_CRAC_defaultinit():
    col = CRAC()
    assert isinstance(col, CRAC)


def test_CRAC_init_onecra():
    c1 = CRA(
        "CFT",
        3,
        -0.0008014597072714745,
        [0.691122195041816 - 0.043143728357214896j, -1.4837496454512307 + 0.0j],
        [0.1981697296161005 - 2.410667766188531j, 1.3688034212107467 + 0.0j],
    )

    col = CRAC(c1)

    assert col.approx[0] is c1


def test_CRAC_init_cras():
    c1 = CRA(
        "CFT",
        3,
        -0.0008014597072714745,
        [0.691122195041816 - 0.043143728357214896j, -1.4837496454512307 + 0.0j],
        [0.1981697296161005 - 2.410667766188531j, 1.3688034212107467 + 0.0j],
    )

    c2 = CRA("CFT", 2, -1, [1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j])

    c3 = CRA("Dummy", 3, -2, [2 + 1j, 1 + 2j], [4 + 3j, 3 + 4j])

    col = CRAC([c1, c2, c3])

    assert col.approx[0] is c1
    assert col.approx[1] is c2
    assert col.approx[2] is c3

    assert len(col) == 3

    assert set(col.origins) == set(["CFT", "Dummy"])
    assert set(col.orders) == set([3, 2])
