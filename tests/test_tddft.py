# -*- coding: utf-8 -*-
"""QED-TDDFT 单元测试"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto
from nqeddft import Cavity, QEDRKS, QEDTDA


@pytest.fixture(scope="module")
def hf_qed():
    mol = gto.M(atom="H 0 0 0; F 0 0 1.733", basis="sto-3g",
                unit="Bohr", verbose=0)
    cav = Cavity().add_mode(0.4, 0.05, [0,0,1])
    mf  = QEDRKS(mol, cav); mf.xc = 'lda,vwn'; mf.verbose = 0
    mf.kernel()
    return mf, cav


def test_tda_returns_energies(hf_qed):
    mf, cav = hf_qed
    td = QEDTDA(mf, cav)
    td.nstates = 3
    e, _ = td.kernel()
    assert len(e) == 3
    assert all(ei > 0 for ei in e), "激发能必须为正"


def test_tda_weights_sum_to_one(hf_qed):
    mf, cav = hf_qed
    td = QEDTDA(mf, cav); td.nstates = 3; td.kernel()
    for ph, ex in zip(td.photon_weight, td.exciton_weight):
        np.testing.assert_allclose(ph + ex, 1.0, atol=1e-4,
                                    err_msg="光子+激子权重之和应为 1")


def test_oscillator_strength_nonneg(hf_qed):
    mf, cav = hf_qed
    td = QEDTDA(mf, cav); td.nstates = 3; td.kernel()
    f = td.oscillator_strength()
    assert all(fi >= 0 for fi in f), "振子强度必须非负"


def test_rabi_splitting_positive(hf_qed):
    mf, cav = hf_qed
    td = QEDTDA(mf, cav); td.nstates = 5; td.kernel()
    from nqeddft.analysis.polariton import PolaritonAnalysis
    pol = PolaritonAnalysis(td).compute()
    assert pol.rabi_splitting > 0
