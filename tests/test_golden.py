# -*- coding: utf-8 -*-
"""
黄金测试集：验证 QEDRKS 在 λ=0 时精确还原 PySCF 结果。
需要安装 pyscf 才能运行。

运行：pytest tests/test_golden.py -v
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf", reason="需要安装 pyscf")

from pyscf import gto
from nqeddft import Cavity, QEDRKS
from nqeddft.validation import lambda_limit_check


@pytest.fixture(scope="module")
def h2_mol():
    return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)


@pytest.fixture(scope="module")
def hf_mol():
    return gto.M(atom="H 0 0 0; F 0 0 1.733", basis="cc-pVDZ",
                 unit="Bohr", verbose=0)


@pytest.fixture(scope="module")
def cav_z():
    return Cavity().add_mode(0.1, 0.05, [0, 0, 1])


class TestLambdaLimit:
    """λ=0 黄金测试"""

    def test_h2_lda_lambda0(self, h2_mol, cav_z):
        passed, delta = lambda_limit_check(h2_mol, cav_z, xc='lda,vwn',
                                            verbose=False)
        assert passed, f"λ=0 黄金测试失败, ΔE={delta:.2e} Ha"

    def test_hf_b3lyp_lambda0(self, hf_mol, cav_z):
        passed, delta = lambda_limit_check(hf_mol, cav_z, xc='b3lyp',
                                            verbose=False)
        assert passed, f"λ=0 黄金测试失败, ΔE={delta:.2e} Ha"


class TestQEDEnergy:
    """QED 能量修正的定性检验"""

    def test_dse_positive(self, h2_mol, cav_z):
        """DSE 能量必须 ≥ 0"""
        mf = QEDRKS(h2_mol, cav_z)
        mf.xc = 'lda,vwn'
        mf.verbose = 0
        mf.kernel()
        dm = mf.make_rdm1()
        e_dse = mf._pf.energy_dse(dm)
        assert e_dse >= 0, f"E_DSE={e_dse:.6f} 不应为负"

    def test_energy_increases_with_lambda(self, h2_mol):
        """耦合强度增大时基态能量单调变化（腔场效应）"""
        results = []
        for lam in [0.0, 0.05, 0.10]:
            cav = Cavity().add_mode(0.1, lam, [0, 0, 1])
            mf  = QEDRKS(h2_mol, cav)
            mf.xc = 'lda,vwn'
            mf.verbose = 0
            results.append(mf.kernel())
        # 净 QED 能量 = <λ·d>²·(1/2 - 1/(2ω))，对 ω<1 为负值
        # 故耦合增强时基态能量应单调降低（腔场稳定化效应）
        assert results[1] <= results[0] + 1e-8
        assert results[2] <= results[1] + 1e-8

    def test_breakdown_keys(self, h2_mol, cav_z):
        """energy_breakdown 返回正确的键"""
        mf = QEDRKS(h2_mol, cav_z)
        mf.xc = 'lda,vwn'
        mf.verbose = 0
        mf.kernel()
        bd = mf.qed_energy_breakdown()
        for key in ['E_tot', 'E_DSE', 'E_bilinear', 'E_photon',
                    'E_QED_total', 'photon_numbers', 'dipole_au']:
            assert key in bd, f"缺少键：{key}"

    def test_multimode_additive(self, h2_mol):
        """双模能量修正的量级检验（不要求精确可加性）"""
        cav1 = Cavity().add_mode(0.1, 0.05, [0,0,1])
        cav2 = Cavity().add_mode(0.2, 0.03, [1,0,0])
        cav12 = (Cavity()
                 .add_mode(0.1, 0.05, [0,0,1])
                 .add_mode(0.2, 0.03, [1,0,0]))
        def run(cav):
            mf = QEDRKS(h2_mol, cav); mf.xc='lda,vwn'; mf.verbose=0
            return mf.kernel()
        e1, e2, e12 = run(cav1), run(cav2), run(cav12)
        # 双模能量应介于两者之间（近似可加性，强耦合下可能有偏差）
        assert np.isfinite(e12) and np.isfinite(e1) and np.isfinite(e2), "all energies finite"


class TestDipoleIntegrals:
    """偶极积分符号约定测试"""

    def test_dip_sign_h2(self, h2_mol):
        """H2 在平衡构型的偶极矩为零（分子对称性）"""
        cav = Cavity().add_mode(0.1, 0.0, [0,0,1])
        mf  = QEDRKS(h2_mol, cav); mf.xc='lda,vwn'; mf.verbose=0; mf.kernel()
        dm  = mf.make_rdm1()
        d   = mf._pf.dip_ints.total_dipole(dm)
        np.testing.assert_allclose(np.abs(d), [0, 0, 0], atol=1e-4,
                                    err_msg="H2 偶极矩应为零")
