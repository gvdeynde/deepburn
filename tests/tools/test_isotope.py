#!/usr/bin/env python

"""Tests for `isotopes` module."""

import pytest

from deepburn import isotope


@pytest.mark.parametrize(
    "zzzaaam, name",
    [
        ("952410", "Am-241"),
        ("952411", "Am-241m"),
        ("952412", "Am-241mm"),
        (952410, "Am-241"),
        (952411, "Am-241m"),
        (952412, "Am-241mm"),
    ],
)
def test_zzzaaam2str_defaults(zzzaaam, name):
    result = isotope._zzzaaam2str(zzzaaam)
    assert result == name


@pytest.mark.parametrize(
    "zzzaaam, name",
    [
        ("952410", "Americium-241"),
        ("952411", "Americium-241m"),
        ("952412", "Americium-241mm"),
        (952410, "Americium-241"),
        (952411, "Americium-241m"),
        (952412, "Americium-241mm"),
    ],
)
def test_zzzaaam2str_element(zzzaaam, name):
    result = isotope._zzzaaam2str(zzzaaam, symbol=False)
    assert result == name


@pytest.mark.parametrize(
    "zzzaaam, name",
    [
        ("952410", "241-Am"),
        ("952411", "241m-Am"),
        ("952412", "241mm-Am"),
        (952410, "241-Am"),
        (952411, "241m-Am"),
        (952412, "241mm-Am"),
    ],
)
def test_zzzaaam2str_zfirst(zzzaaam, name):
    result = isotope._zzzaaam2str(zzzaaam, zfirst=False)
    assert result == name


@pytest.mark.parametrize(
    "zzzaaam, name",
    [
        ("952410", "Am-241"),
        ("952411", "Am-241"),
        ("952412", "Am-241"),
        (952410, "Am-241"),
        (952411, "Am-241"),
        (952412, "Am-241"),
    ],
)
def test_zzzaaam2str_nometa(zzzaaam, name):
    result = isotope._zzzaaam2str(zzzaaam, meta=False)
    assert result == name


@pytest.mark.parametrize(
    "zzzaaam, name",
    [
        ("952410", "Am+241"),
        ("952411", "Am+241m"),
        ("952412", "Am+241mm"),
        (952410, "Am+241"),
        (952411, "Am+241m"),
        (952412, "Am+241mm"),
    ],
)
def test_zzzaaam2str_separator(zzzaaam, name):
    result = isotope._zzzaaam2str(zzzaaam, separator="+")
    assert result == name


def test_zzzaam2str_exceptions_1():
    with pytest.raises(ValueError):
        isotope._zzzaaam2str("123")


def test_zzzaam2str_exceptions_2():
    with pytest.raises(ValueError):
        isotope._zzzaaam2str("12345678")


# tests for str2zzzaaam
@pytest.mark.parametrize(
    "name, zzzaaam",
    [
        ("Am241", (95, 241, 0)),
        ("Am241m", (95, 241, 1)),
        ("Am241mm", (95, 241, 2)),
        ("Am-241", (95, 241, 0)),
        ("Am-241m", (95, 241, 1)),
        ("Am-241mm", (95, 241, 2)),
        ("Americium241", (95, 241, 0)),
        ("Americium241m", (95, 241, 1)),
        ("Americium241mm", (95, 241, 2)),
        ("Americium-241", (95, 241, 0)),
        ("Americium-241m", (95, 241, 1)),
        ("Americium-241mm", (95, 241, 2)),
        ("241Am", (95, 241, 0)),
        ("241mAm", (95, 241, 1)),
        ("241mmAm", (95, 241, 2)),
        ("241-Am", (95, 241, 0)),
        ("241m-Am", (95, 241, 1)),
        ("241mm-Am", (95, 241, 2)),
        ("241Americium", (95, 241, 0)),
        ("241mAmericium", (95, 241, 1)),
        ("241mmAmericium", (95, 241, 2)),
        ("241-Americium", (95, 241, 0)),
        ("241m-Americium", (95, 241, 1)),
        ("241mm-Americium", (95, 241, 2)),
    ],
)
def test_str2zzzaam(name, zzzaaam):
    result = isotope._str2zzzaaam(name)
    assert result == zzzaaam


def test_str2zzzaam_exceptions_1():
    with pytest.raises(ValueError):
        isotope._str2zzzaaam("Ap241")


def test_str2zzzaam_exceptions_3():
    with pytest.raises(ValueError):
        isotope._str2zzzaaam("241Alhambra")


def test_str2zzzaam_exceptions_4():
    with pytest.raises(ValueError):
        isotope._str2zzzaaam("241Ap")


def test_str2zzzaam_exceptions_2():
    with pytest.raises(ValueError):
        isotope._str2zzzaaam("Alhambra241")


def test_Isotope_constructor():
    iso = isotope.Isotope((1, 1, 0))

    assert iso.zzz == 1
    assert iso.aaa == 1
    assert iso.meta == 0


def test_Isotope_constructorfromname():
    iso = isotope.Isotope("H1")

    assert iso.zzz == 1
    assert iso.aaa == 1
    assert iso.meta == 0


def test_Isotope_constructorfromname():
    iso = isotope.Isotope("239mPu")

    assert iso.zzz == 94
    assert iso.aaa == 239
    assert iso.meta == 1


@pytest.mark.parametrize(
    "name, zzzaaam",
    [
        ("Am-241", (95, 241, 0)),
        ("Am-241m", (95, 241, 1)),
        ("Xe-135", (54, 135)),
        ("H-1", (1, 1)),
        ("Cf-252", (98, 252, 0)),
    ],
)
def test_Isotope_namegenerator(name, zzzaaam):
    iso = isotope.Isotope(zzzaaam)
    assert iso.name == name


def test_Isotope_exceptions():
    iso = isotope.Isotope((1, 1, 1))

    with pytest.raises(ValueError):
        iso.zzz = 2

    with pytest.raises(ValueError):
        iso.aaa = 2

    with pytest.raises(ValueError):
        iso.meta = 2

    with pytest.raises(ValueError):
        iso.name = "test"


def test_Isotope_fmt():
    iso = isotope.Isotope((54, 135))

    assert f"{iso:zam}" == "54-135"
    assert f"{iso:Sam}" == "Xe-135"
    assert f"{iso:Eam}" == "Xenon-135"

    iso = isotope.Isotope((95, 241, 1))

    assert f"{iso:zam}" == "95-241m"
    assert f"{iso:Sam}" == "Am-241m"
    assert f"{iso:Eam}" == "Americium-241m"


def test_Isotope_equality():
    iso = isotope.Isotope((54, 135))
    iso1 = isotope.Isotope((54, 135, 1))
    iso2 = isotope.Isotope((54, 136))
    iso3 = isotope.Isotope((52, 135))

    assert iso == iso
    assert iso != iso1
    assert iso != iso2
    assert iso != iso3

@pytest.mark.parametrize(
    "iso1, iso2, lt",
    [
        ((1, 1), (2, 4), True),
        ((1, 1), (1, 1), False),
        ((1, 1), (1, 2), True),
        ((1, 1), (1, 1, 0), False),
        ((1, 1), (1, 1, 1), True),
    ],
)
def test_Isotope_inequality(iso1, iso2, lt):
    assert((isotope.Isotope(iso1) < isotope.Isotope(iso2)) == lt)
