# -*- coding: utf-8 -*-
"""QED-CCSD 单元测试"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto
from nqeddft import Cavity, QEDRHF, QEDCCSD


@pytest.fixture(scope="module")
def h2_qedrhf():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    cav = Cavity().add_mode(0.1, 0.05, [0, 0, 1])
    mf  = QEDRHF(mol, cav)
    mf.verbose = 0
    mf.kernel()
    return mf, cav


def test_qedccsd_runs(h2_qedrhf):
    mf, cav = h2_qedrhf
    cc = QEDCCSD(mf, cav)
    cc.verbose = 0
    e, _, _ = cc.kernel()
    assert np.isfinite(e)
    assert e < mf.e_tot   # CCSD 能量应低于 HF 能量


def test_qedccsd_breakdown(h2_qedrhf):
    mf, cav = h2_qedrhf
    cc = QEDCCSD(mf, cav); cc.verbose = 0; cc.kernel()
    bd = cc.qed_breakdown()
    for mode_name, d in bd.items():
        assert 'E_DSE' in d
        assert d['E_DSE'] >= 0   # DSE 能量非负
