# -*- coding: utf-8 -*-
"""单元测试：Cavity 和 CavityMode"""
import numpy as np
import pytest
from nqeddft import Cavity, CavityMode


def test_add_mode_chain():
    cav = Cavity().add_mode(0.1, 0.05, [0,0,1]).add_mode(0.2, 0.02, [1,0,0])
    assert cav.n_modes == 2
    assert cav.modes[0].name == "mode_0"

def test_polarization_normalization():
    cav = Cavity().add_mode(0.1, 0.05, [3, 4, 0])
    np.testing.assert_allclose(np.linalg.norm(cav.modes[0].polarization), 1.0)

def test_validate_empty():
    with pytest.raises(ValueError, match="为空"):
        Cavity().validate()

def test_validate_negative_omega():
    with pytest.raises(ValueError):
        Cavity().add_mode(-0.1, 0.05, [0,0,1]).validate()

def test_lambda_vec():
    m = CavityMode(0.1, 0.5, [0, 0, 1])
    np.testing.assert_allclose(m.lambda_vec, [0, 0, 0.5])

def test_coupling_prefactor():
    m = CavityMode(0.2, 0.1, [0, 1, 0])
    expected = np.sqrt(0.2/2) * 0.1 * np.array([0, 1, 0])
    np.testing.assert_allclose(m.coupling_prefactor, expected)

def test_regime():
    assert CavityMode(0.1, 0.005, [0,0,1]).regime() == "弱耦合"
    assert CavityMode(0.1, 0.05, [0,0,1]).regime() == "中等耦合"
    assert CavityMode(0.1, 0.2, [0,0,1]).regime() == "强耦合"
    assert CavityMode(0.1, 0.5, [0,0,1]).regime() == "超强耦合"

def test_rabi_estimate():
    m = CavityMode(0.1, 0.1, [0,0,1])
    dip = 1.0  # a.u.
    omega_r = m.rabi_estimate(dip)
    assert omega_r > 0
    np.testing.assert_allclose(omega_r, 2 * np.sqrt(0.05) * 0.1, rtol=1e-6)
