# -*- coding: utf-8 -*-
"""
nqeddft.cc.qed_ccsd  —  QED-CCSD（微扰级别光子修正）

策略：在收敛的 CCSD 波函数基础上，加入光子二阶微扰修正。
完整振幅方程（Haugland et al. PRL 2020）在 Phase 3 实现。

E_QED-CCSD = E_CCSD + ΔE_DSE[ρ_CCSD] + ΔE_pt2^photon
"""
import numpy as np
from pyscf.cc import ccsd
from ..cavity import Cavity
from ..integrals.dipole import DipoleIntegrals


class QEDCCSD(ccsd.CCSD):
    _keys = ccsd.CCSD._keys | {'cavity'}

    def __init__(self, mf, cavity: Cavity):
        cavity.validate()
        super().__init__(mf)
        self.cavity    = cavity
        self.dip_ints  = DipoleIntegrals(mf.mol)
        self.e_corr_qed = 0.0

    def kernel(self, t1=None, t2=None, eris=None):
        e_ccsd, t1, t2 = super().kernel(t1, t2, eris)
        # CCSD 一阶 RDM → AO 基密度矩阵
        dm1_mo = self.make_rdm1()
        mo     = self._scf.mo_coeff
        dm_ao  = mo @ dm1_mo @ mo.T
        # 光子微扰修正
        de = self._photon_pt2(dm_ao)
        self.e_corr_qed = de
        self.e_tot_qed  = self.e_tot + de
        if self.verbose >= 3:
            print(f"  QED 光子微扰 ΔE = {de:+.8f} Ha")
            print(f"  QED-CCSD 总能量  = {self.e_tot_qed:.10f} Ha")
        return self.e_tot_qed, t1, t2

    def _photon_pt2(self, dm_ao: np.ndarray) -> float:
        """
        光子二阶微扰修正：
          ΔE = Σ_α [ (1/2)<λ·d>²  (DSE)  -  <λ·d>²/(2ω)  (pt2) ]

        其中 <λ·d> = Tr[(λ·d)_AO · P_CCSD] 为电子偶极期望值。

        注意：
        1. 此处 <λ·d> 仅含电子贡献，与 SCF 层面定义一致。
        2. 核偶极 λ·d_nuc 为常数，对能量差（腔中与自由空间之差）无贡献，
           故可安全忽略。
        3. 这是 Haugland et al. PRL 2020 完整 QED-CCSD 的微扰近似，
           完整振幅方程在 Phase 3 实现。适用范围：弱耦合（λ < 0.05）。
        """
        dip_ao = self.dip_ints.dip_ao
        de = 0.0
        for mode in self.cavity.modes:
            ld      = np.einsum('k,kpq->pq', mode.lambda_vec, dip_ao)
            ld_mean = float(np.einsum('pq,qp->', ld, dm_ao))
            de += 0.5 * ld_mean**2                      # DSE
            de -= ld_mean**2 / (2.0 * mode.omega)       # pt2
            # de += 0.5 * mode.omega                   # ZPE removed
        return de

    def qed_breakdown(self) -> dict:
        if not hasattr(self, 'e_corr_qed'):
            raise RuntimeError("先调用 kernel()")
        dm_ao = self._scf.mo_coeff @ self.make_rdm1() @ self._scf.mo_coeff.T
        result = {}
        dip_ao = self.dip_ints.dip_ao
        for mode in self.cavity.modes:
            ld      = np.einsum('k,kpq->pq', mode.lambda_vec, dip_ao)
            ld_mean = float(np.einsum('pq,qp->', ld, dm_ao))
            result[mode.name] = {
                'ld_mean': ld_mean,
                'E_DSE':   0.5 * ld_mean**2,
                'E_pt2':   -ld_mean**2 / (2.0 * mode.omega),
                'E_ZPE':   0.5 * mode.omega,
            }
        return result
