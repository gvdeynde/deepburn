# -*- coding: utf-8 -*-

""" Tests for the `ChebRatExp` module."""

import pytest

from deepburn.CRAM.ChebRatExp import ChebRatExp, ChebRatExpCollection

def test_init():
    c = ChebRatExp()
    assert(isinstance(c, ChebRatExp))

