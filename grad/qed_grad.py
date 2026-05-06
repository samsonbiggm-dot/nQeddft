# -*- coding: utf-8 -*-
"""
nqeddft.grad.qed_grad  —  QED-DFT 解析核梯度
继承 pyscf.grad.rks.Gradients，在 PySCF 标准 KS 梯度基础上叠加：
  ∂E_DSE/∂R_I      = <λ·d_total> · [Tr[∂(λ·d)_AO/∂R_I · P] + Z_I·λ_l]
  ∂E_bilinear/∂R_I = 同形式，系数为 -√(2/ω)·<λ·d_total>
实现说明：
  pyscf.grad.rks.Gradients.get_veff 直接用 mf._numint 和 mf.xc 计算
  XC+JK 的梯度张量，不调用 mf.get_veff，因此 QEDRKS.get_veff 里的
  QED Fock 修正对 pyscf 的梯度计算完全透明，不产生干扰。
  super().grad_elec() 已经通过含 QED 修正的 mo_energy/mo_coeff 间接包含
  了 QED 对电子结构的影响（即 CP-KS 层面的间接效应）。
  剩余的显式 QED 梯度贡献（dE_DSE/dR, dE_bilinear/dR）由
  grad_dse + grad_bilinear 单独提供。

fix (2026-03): 新增 QEDUKSGradients，处理 UKS（开壳层）体系。
  QEDUKS.mo_occ 是 (2, nmo) 的二维数组（alpha/beta 分开），
  rks_grad.Gradients（继承自 rhf_grad）的 make_rdm1e 只接受一维 mo_occ，
  故开壳层体系必须改继承 pyscf.grad.uks.Gradients。
"""
import numpy as np
from pyscf.grad import rks as rks_grad
from pyscf.grad import uks as uks_grad
from ..cavity import Cavity


# ── 工具：给 as_scanner() 返回的 _S 对象注入 geometric_solver 需要的属性 ──
def _make_scanner(base_scanner, mf):
    """
    包装 PySCF scanner，确保返回对象有 verbose/base 属性。
    geometric_solver 内部调用 logger.note(g_scanner, ...) 需要 g_scanner.verbose，
    PySCF 原生 scanner 有此属性，但 nqeddft 自定义的 _S 没有，故在此统一注入。
    """
    _pf = mf._pf

    class _S:
        def __init__(self, s):
            self._s      = s
            self.verbose = mf.verbose   # fix: geometric_solver 需要
            self.base    = mf           # fix: geometric_solver 需要
            self.mol     = s.mol
            self.e_tot   = None

        def __call__(self, mol):
            _pf.dip_ints.invalidate_cache()
            result = self._s(mol)
            self.mol   = self._s.mol
            # scanner 返回 (energy, grad) 或仅 grad，兼容两种情况
            if isinstance(result, tuple):
                self.e_tot = result[0]
            else:
                self.e_tot = getattr(self._s, 'e_tot', None)
            return result

        @property
        def mol(self):
            return self._s.mol

        @mol.setter
        def mol(self, val):
            pass   # geometric_solver 会设置，忽略即可

    return _S(base_scanner)


# ── RKS（闭壳层）QED 梯度 ────────────────────────────────────────────────
class QEDGradients(rks_grad.Gradients):
    """QED-DFT 解析核梯度（RKS，闭壳层）"""

    def __init__(self, mf):
        super().__init__(mf)
        self.cavity = mf.cavity
        self._pf    = mf._pf

    def grad_elec(self, mo_energy=None, mo_coeff=None,
                  mo_occ=None, atmlst=None):
        de  = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        mol = self.mol
        dm  = self.base.make_rdm1()
        if atmlst is None:
            atmlst = list(range(mol.natm))
        for k, atm_id in enumerate(atmlst):
            de[k] += self._pf.grad_dse(dm, atm_id)
            de[k] += self._pf.grad_bilinear(dm, atm_id)
        return de

    def kernel(self, mo_energy=None, mo_coeff=None,
               mo_occ=None, atmlst=None):
        de = super().kernel(mo_energy, mo_coeff, mo_occ, atmlst)
        if self.verbose >= 3:
            mol = self.mol
            print("QED-DFT 核梯度 (Ha/Bohr):")
            for i in range(mol.natm):
                s = mol.atom_symbol(i)
                print(f"  {s:3s}  {de[i,0]:+12.8f} {de[i,1]:+12.8f} {de[i,2]:+12.8f}")
        return de

    def as_scanner(self):
        return _make_scanner(super().as_scanner(), self.base)


# ── UKS（开壳层）QED 梯度 ────────────────────────────────────────────────
class QEDUKSGradients(uks_grad.Gradients):
    """
    QED-DFT 解析核梯度（UKS，开壳层）。

    fix: QEDUKS.mo_occ 是 (2, nmo) 的二维数组，
    rks_grad（继承自 rhf_grad）的 make_rdm1e 用 mo_occ>0 做一维布尔索引会报
    IndexError，故改继承 uks_grad.Gradients（正确处理 alpha/beta 分离的 mo_occ）。
    QED 额外梯度项（DSE + bilinear）的形式与 RKS 相同，叠加在 UKS 梯度上。
    """

    def __init__(self, mf):
        super().__init__(mf)
        self.cavity = mf.cavity
        self._pf    = mf._pf

    def grad_elec(self, mo_energy=None, mo_coeff=None,
                  mo_occ=None, atmlst=None):
        # UKS 父类正确处理 (2,nmo) 的 mo_occ
        de  = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        mol = self.mol
        # UKS 的 make_rdm1 返回 (2, nao, nao)，对 DSE/bilinear 梯度取和
        dm  = self.base.make_rdm1()   # shape (2, nao, nao)
        dm_total = dm[0] + dm[1]      # 总密度矩阵，与 RKS 接口一致
        if atmlst is None:
            atmlst = list(range(mol.natm))
        for k, atm_id in enumerate(atmlst):
            de[k] += self._pf.grad_dse(dm_total, atm_id)
            de[k] += self._pf.grad_bilinear(dm_total, atm_id)
        return de

    def kernel(self, mo_energy=None, mo_coeff=None,
               mo_occ=None, atmlst=None):
        de = super().kernel(mo_energy, mo_coeff, mo_occ, atmlst)
        if self.verbose >= 3:
            mol = self.mol
            print("QED-DFT 核梯度 UKS (Ha/Bohr):")
            for i in range(mol.natm):
                s = mol.atom_symbol(i)
                print(f"  {s:3s}  {de[i,0]:+12.8f} {de[i,1]:+12.8f} {de[i,2]:+12.8f}")
        return de

    def as_scanner(self):
        return _make_scanner(super().as_scanner(), self.base)
