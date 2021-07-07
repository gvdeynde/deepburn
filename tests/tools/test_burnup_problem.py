import pytest

from numpy import allclose
from scipy.sparse import dok_matrix
from deepburn.isotope import Isotope
from deepburn.burnup_problem import IsotopicComposition, Transitions, BUP, Polonium


def test_isotopic_composition_init():
    ics = IsotopicComposition()
    assert ics._ics == {}


def test_isotopic_composition_add():
    ics = IsotopicComposition()
    iso1 = Isotope("Bi209")
    iso2 = Isotope("Bi210")
    iso3 = Isotope("Po210")

    isolist = [iso1, iso2, iso3]

    ics.add_value(iso1, 1e17)

    assert allclose(ics(isolist), [1e17, 0, 0])


def test_isotopic_composition_add_two():
    ics = IsotopicComposition()
    iso1 = Isotope("Bi209")
    iso2 = Isotope("Bi210")
    iso3 = Isotope("Po210")

    isolist = [iso1, iso2, iso3]

    ics.add_value(iso1, 1e17)
    ics.add_value(iso3, 1e5)

    assert allclose(ics(isolist), [1e17, 0, 1e5])


def test_isotopic_composition_overwrite():
    ics = IsotopicComposition()
    iso1 = Isotope("Bi209")
    iso2 = Isotope("Bi210")
    iso3 = Isotope("Po210")

    isolist = [iso1, iso2, iso3]

    ics.add_value(iso1, 1e17)
    ics.add_value(iso1, 1e5)

    assert allclose(ics(isolist), [1e5, 0, 0])


def test_isotopic_composition_reorder():
    ics = IsotopicComposition()
    iso1 = Isotope("Bi209")
    iso2 = Isotope("Bi210")
    iso3 = Isotope("Po210")

    isolist = [iso3, iso2, iso1]

    ics.add_value(iso1, 1e17)

    assert allclose(ics(isolist), [0, 0, 1e17])


def test_transitions_init():
    t = Transitions()
    assert isinstance(t, Transitions)
    assert isinstance(t._trans, list) and not t._trans


@pytest.fixture
def Po210_example():
    iso = []
    iso.append(Isotope("Bi209"))
    iso.append(Isotope("Bi210"))
    iso.append(Isotope("Po210"))

    trans = []
    trans.append(1.83163e-12)
    trans.append(1.60035e-6)
    trans.append(5.79764e-8)

    return (iso, trans)


def test_transition_iso(Po210_example):
    t = Transitions()
    iso, trans = Po210_example
    t.add_transition(iso[0], trans[0], iso[1])

    assert set(t.isotopes) == set([iso[0], iso[1]])


def test_transition(Po210_example):
    t = Transitions()
    iso, trans = Po210_example
    t.add_transition(iso[0], trans[0], iso[1])

    assert t._trans == [(iso[0], trans[0], iso[1])]


def test_transition_null(Po210_example):
    t = Transitions()
    iso, trans = Po210_example
    t.add_transition(iso[0], trans[0])
    assert t._trans == [(iso[0], trans[0], None)]


@pytest.fixture
def Po210_example_trans(Po210_example):
    iso = []
    iso.append(Isotope("Bi209"))
    iso.append(Isotope("Bi210"))
    iso.append(Isotope("Po210"))

    trans = []
    trans.append(1.83163e-12)
    trans.append(1.60035e-6)
    trans.append(5.79764e-8)

    t = Transitions()
    iso, trans = Po210_example
    t.add_transition(iso[0], trans[0], iso[1])
    t.add_transition(iso[1], trans[1], iso[2])
    t.add_transition(iso[2], trans[2])

    return iso, trans, t


def test_transition_full(Po210_example_trans):
    iso, trans, transition = Po210_example_trans

    assert transition._trans == [
        (iso[0], trans[0], iso[1]),
        (iso[1], trans[1], iso[2]),
        (iso[2], trans[2], None),
    ]


def test_Polonium(Po210_example_trans):
    iso, trans, transition = Po210_example_trans

    dok = dok_matrix((3, 3))
    dok[0, 0] = -trans[0]
    dok[1, 0] = +trans[0]
    dok[1, 1] = -trans[1]
    dok[2, 1] = +trans[1]
    dok[2, 2] = -trans[2]

    pol = Polonium()

    assert pol.name == "Po210"
    assert allclose(dok.A, pol.sparsematrix.A)
    assert pol.time_stamps == [20 * 24 * 3600, 180 * 24 * 3600]
