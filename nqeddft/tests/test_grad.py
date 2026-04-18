# -*- coding: utf-8 -*-
"""QED-DFT 核梯度数值验证测试"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto
from nqeddft import Cavity, QEDRKS
from nqeddft.grad import QEDGradients


@pytest.fixture(scope="module")
def h2_qed():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    cav = Cavity().add_mode(0.1, 0.05, [0, 0, 1])
    mf  = QEDRKS(mol, cav)
    mf.xc = 'lda,vwn'
    mf.verbose = 0
    mf.kernel()
    return mf


def test_grad_shape(h2_qed):
    g = QEDGradients(h2_qed)
    de = g.kernel()
    assert de.shape == (h2_qed.mol.natm, 3)


def test_grad_translational_invariance(h2_qed):
    """核梯度之和应近似为零（平动不变性）"""
    g  = QEDGradients(h2_qed)
    de = g.kernel()
    total = de.sum(axis=0)
    np.testing.assert_allclose(total, [0, 0, 0], atol=2e-4,
                                err_msg="核梯度违反平动不变性")


def numerical_gradient(mf, stepsize=0.002):
    """
    数值梯度（中心差分）用于对比。

    重要：每次位移必须新建 mol 和 mf 对象。
    复用同一个 mol 对象并调用 set_geom_ 会污染 pyscf 内部的积分缓存
    （尤其是 mol._atm/mol._bas/mol._env 等底层数组），导致后续 SCF
    在错误的几何下计算，产生严重错误的数值梯度。
    """
    mol0   = mf.mol
    natm   = mol0.natm
    coords = mol0.atom_coords().copy()   # Bohr
    basis  = mol0.basis
    xc     = mf.xc
    cavity = mf.cavity
    grad_num = np.zeros((natm, 3))

    for i in range(natm):
        for j in range(3):
            # +步长：新建独立的 mol 和 mf
            c_p = coords.copy(); c_p[i, j] += stepsize
            atom_str_p = '; '.join(
                f"{mol0.atom_symbol(k)} {c_p[k,0]} {c_p[k,1]} {c_p[k,2]}"
                for k in range(natm)
            )
            mol_p = gto.M(atom=atom_str_p, basis=basis, unit='Bohr', verbose=0)
            mf_p  = QEDRKS(mol_p, cavity); mf_p.xc = xc; mf_p.verbose = 0
            mf_p.kernel()
            e_p = mf_p.e_tot

            # -步长：新建独立的 mol 和 mf
            c_m = coords.copy(); c_m[i, j] -= stepsize
            atom_str_m = '; '.join(
                f"{mol0.atom_symbol(k)} {c_m[k,0]} {c_m[k,1]} {c_m[k,2]}"
                for k in range(natm)
            )
            mol_m = gto.M(atom=atom_str_m, basis=basis, unit='Bohr', verbose=0)
            mf_m  = QEDRKS(mol_m, cavity); mf_m.xc = xc; mf_m.verbose = 0
            mf_m.kernel()
            e_m = mf_m.e_tot

            grad_num[i, j] = (e_p - e_m) / (2 * stepsize)

    return grad_num


@pytest.mark.slow
def test_grad_vs_numerical(h2_qed):
    """解析梯度与数值梯度对比（标记为 slow，需显式运行）"""
    g_anal = QEDGradients(h2_qed).kernel()
    g_num  = numerical_gradient(h2_qed)
    np.testing.assert_allclose(g_anal, g_num, atol=1e-5,
                                err_msg="解析梯度与数值梯度不一致")
