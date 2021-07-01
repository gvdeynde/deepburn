import pytest

from numpy import allclose
from scipy.sparse import dok_matrix
from deepburn.isotope import Isotope
from deepburn.burnup_problem import Transitions, BUP, Polonium

def test_transitions_init():
    t = Transitions()
    assert isinstance(t, Transitions)
    assert isinstance(t._trans, list) and not t._trans
    assert isinstance(t._isotopes, set) and not t._isotopes

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

def test_transition(Po210_example):
    t = Transitions()
    iso, trans = Po210_example
    t.add_transition(iso[0], trans[0], iso[1])

    assert t._isotopes == {iso[0], iso[1]}
    assert t._trans == [(iso[0], trans[0], iso[1])]

def test_transition_null(Po210_example):
    t = Transitions()
    iso, trans = Po210_example
    t.add_transition(iso[0], trans[0])
    assert t._isotopes == {iso[0]}
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

    assert transition._isotopes == set(iso)
    assert transition._trans == [(iso[0], trans[0], iso[1]),
            (iso[1], trans[1], iso[2]),
            (iso[2], trans[2], None)]


def test_transition_Podok(Po210_example_trans):
    iso, trans, transition = Po210_example_trans

    dok = dok_matrix((3,3))
    dok[0,0] = -trans[0]
    dok[1,0] = +trans[0]
    dok[1,1] = -trans[1]
    dok[2,1] = +trans[1]
    dok[2,2] = -trans[2]

    assert allclose(dok.A, transition.transition_matrix.A)

def test_Polonium(Po210_example_trans):
    iso, trans, transition = Po210_example_trans

    dok = dok_matrix((3,3))
    dok[0,0] = -trans[0]
    dok[1,0] = +trans[0]
    dok[1,1] = -trans[1]
    dok[2,1] = +trans[1]
    dok[2,2] = -trans[2]

    pol = Polonium()

    assert (pol.name=="Po210")
    assert allclose(dok.A, pol.sparsematrix.A)
    assert pol.time_stamps == [20*24*3600, 90*24*3600, 180*24*3600]


