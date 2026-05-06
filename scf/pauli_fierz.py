# -*- coding: utf-8 -*-
"""
nqeddft.scf.pauli_fierz
======================
Pauli-Fierz Hamiltonian 对 Fock 矩阵的全部修正项（平均场级别）。

物理：在偶极规范、长波近似下，
  H_PF = H_el + sum_a[ w_a(a†a+1/2) - sqrt(w/2)(lam·d)(a†+a) + (1/2)(lam·d)^2 ]

平均场（相干态近似）处理光子自由度后，电子 Fock 矩阵增加两项修正：

  DSE 偶极自能项：E_DSE = (1/2) <λ·d>²
    → Fock 导数 V_DSE = δE_DSE/δP = <λ·d> · (λ·d)_μν
      （即 V_J，只有这一项；不存在额外的 exchange-like V_K）

  双线性耦合项（相干态近似 α = <λ·d>/√(2ω)）：
    E_bil = -<λ·d>²/ω
    → Fock 导数 V_bil = -√(ω/2)·<a+a†>·(λ·d) = -<λ·d>/ω · (λ·d)_μν
      （与 V_DSE 合并后净效应为 <λ·d>·(1/2 - 1/ω)·(λ·d)_μν）

  光子场能量：E_ph = ω·|α|² = <λ·d>²/(2ω)

  三项之和：E_tot_QED = <λ·d>²·(1/2 - 1/ω + 1/(2ω)) = <λ·d>²·(1/2 - 1/(2ω))

相干态近似：α = <λ·d>/√(2ω)，光子场跟随电子偶极瞬时弛豫。
适用范围：弱至中等耦合 (lambda < 0.1)。强耦合需截断 Fock 空间方法。

注意：<λ·d> = Tr[(λ·d)·P] 仅含电子偶极贡献；核偶极为常数不影响 SCF
      轨道优化，但在梯度计算中需要加入核贡献。
"""
import numpy as np
from ..integrals.dipole import DipoleIntegrals
from ..cavity import Cavity


class PauliFierzCorrection:
    """
    计算 Pauli-Fierz 修正项，供 QED-RKS/RHF/UKS 的 get_veff 调用。

    Parameters
    ----------
    mol    : pyscf.gto.Mole
    cavity : Cavity
    """

    def __init__(self, mol, cavity: Cavity):
        self.mol      = mol
        self.cavity   = cavity
        self.dip_ints = DipoleIntegrals(mol)

    # ------------------------------------------------------------------
    # Fock 矩阵修正
    # ------------------------------------------------------------------

    def get_vqed(self, dm: np.ndarray) -> np.ndarray:
        """
        返回 QED Fock 矩阵修正 V_QED = V_DSE + V_bilinear，shape (nao, nao)。
        在每次 SCF 迭代的 get_veff 中调用。

        推导（偶极规范，相干态近似）：
          E_DSE     = (1/2)<λ·d>²  → V_DSE = <λ·d>·(λ·d)_μν
          E_bilinear= -<λ·d>²/ω   → V_bil = -(2/ω)·<λ·d>·(λ·d)_μν
                                            = -√(ω/2)·2α·(λ·d)_μν

        净 Fock 修正：V_QED = <λ·d>·(1 - 2/ω)·(λ·d)_μν
                            = <λ·d>·(λ·d)_μν - √(2/ω)·<λ·d>·(λ·d)_μν

        注意：不存在额外的 V_K（exchange-like）项。
        V_DSE 的 Fock 导数对 Hermitian P 只有一项：δ<λ·d>²/δP_μν = 2<λ·d>·(λ·d)_μν，
        乘以 1/2 后即 <λ·d>·(λ·d)_μν。
        """
        vqed = np.zeros_like(dm, dtype=float)
        dm   = np.asarray(dm, dtype=float)

        for mode in self.cavity.modes:
            lam     = mode.lambda_vec
            ld      = self.dip_ints.lambda_dot_d(lam)        # (nao,nao)
            ld_mean = float(np.einsum('pq,qp->', ld, dm))    # <λ·d_elec>

            # V_DSE = <λ·d> · (λ·d)_μν
            # 来自 E_DSE = (1/2)<λ·d>²，对 P 的 Fock 导数
            vqed += ld_mean * ld

            # V_bilinear = -√(ω/2) · <a+a†> · (λ·d)_μν
            # 相干态近似：α = <λ·d>/√(2ω)，<a+a†> = 2α
            # => V_bil = -√(ω/2)·(2α)·ld = -(1/ω)·<λ·d>·ld
            alpha = ld_mean / np.sqrt(2.0 * mode.omega)
            vqed -= mode.sqrt_omega_half * (2.0 * alpha) * ld

        return vqed

    # ------------------------------------------------------------------
    # 能量贡献
    # ------------------------------------------------------------------

    def energy_dse(self, dm: np.ndarray) -> float:
        """
        E_DSE = (1/2) * sum_a <λ·d_total>²，单位 Ha。

        使用总偶极 d_total = d_elec + d_nuc，与 grad_dse 保持一致，
        确保能量和梯度之间的自洽性（∂E/∂R = grad_dse）。
        """
        e  = 0.0
        dm = np.asarray(dm, dtype=float)
        for mode in self.cavity.modes:
            ld      = self.dip_ints.lambda_dot_d(mode.lambda_vec)
            ld_elec = float(np.einsum('pq,qp->', ld, dm))
            ld_nuc  = float(np.dot(mode.lambda_vec, self.dip_ints.nuclear_dipole()))
            ld_mean = ld_elec + ld_nuc
            e      += 0.5 * ld_mean ** 2
        return e

    def energy_bilinear(self, dm: np.ndarray) -> float:
        """
        E_bilinear = sum_a -<λ·d_total>²/ω，单位 Ha。

        使用总偶极，与 grad_bilinear 保持一致。
        """
        e  = 0.0
        dm = np.asarray(dm, dtype=float)
        for mode in self.cavity.modes:
            ld      = self.dip_ints.lambda_dot_d(mode.lambda_vec)
            ld_elec = float(np.einsum('pq,qp->', ld, dm))
            ld_nuc  = float(np.dot(mode.lambda_vec, self.dip_ints.nuclear_dipole()))
            ld_mean = ld_elec + ld_nuc
            e      += -ld_mean ** 2 / mode.omega
        return e

    def energy_photon(self, dm: np.ndarray) -> float:
        """
        E_ph = sum_a <λ·d_total>²/(2ω)，单位 Ha（不含零点能）。

        使用总偶极，与 grad_dse/grad_bilinear 保持一致。
        """
        e  = 0.0
        dm = np.asarray(dm, dtype=float)
        for mode in self.cavity.modes:
            ld      = self.dip_ints.lambda_dot_d(mode.lambda_vec)
            ld_elec = float(np.einsum('pq,qp->', ld, dm))
            ld_nuc  = float(np.dot(mode.lambda_vec, self.dip_ints.nuclear_dipole()))
            ld_mean = ld_elec + ld_nuc
            e      += ld_mean ** 2 / (2.0 * mode.omega)
        return e

    def energy_qed_total(self, dm: np.ndarray) -> float:
        """三项 QED 能量之和，单位 Ha。"""
        return (self.energy_dse(dm)
                + self.energy_bilinear(dm)
                + self.energy_photon(dm))

    # ------------------------------------------------------------------
    # 光子场诊断
    # ------------------------------------------------------------------

    def photon_number_mean(self, dm: np.ndarray) -> dict:
        """
        各模式的光子数期望值 <n_a> = |alpha_a|^2。
        返回 {mode_name: float}。
        """
        dm = np.asarray(dm, dtype=float)
        result = {}
        for mode in self.cavity.modes:
            ld      = self.dip_ints.lambda_dot_d(mode.lambda_vec)
            ld_mean = float(np.einsum('pq,qp->', ld, dm))
            alpha   = ld_mean / np.sqrt(2.0 * mode.omega)
            result[mode.name] = float(alpha ** 2)
        return result

    def photon_field_amplitudes(self, dm: np.ndarray) -> dict:
        """
        各模式的相干态振幅 alpha_a = <lam·d> / sqrt(2*w)。
        |alpha|^2 = 光子数，angle(alpha) = 场相位。
        返回 {mode_name: float}。
        """
        dm = np.asarray(dm, dtype=float)
        result = {}
        for mode in self.cavity.modes:
            ld      = self.dip_ints.lambda_dot_d(mode.lambda_vec)
            ld_mean = float(np.einsum('pq,qp->', ld, dm))
            result[mode.name] = float(ld_mean / np.sqrt(2.0 * mode.omega))
        return result

    def vacuum_fluctuation_energy(self) -> dict:
        """各模式的真空涨落能（零点能）w/2，单位 Ha。"""
        return {m.name: 0.5 * m.omega for m in self.cavity.modes}

    # ------------------------------------------------------------------
    # 核梯度修正（用于 QEDGradients）
    # ------------------------------------------------------------------

    def grad_dse(self, dm: np.ndarray, atm_id: int) -> np.ndarray:
        """
        dE_DSE/dR_I，shape (3,)，单位 Ha/Bohr。

        平动不变性要求 E_DSE 使用总偶极 d_total = d_elec + d_nuc：
          E_DSE = (1/2)<λ·d_total>²

        梯度：dE_DSE/dR_I^l = <λ·d_total> · d<λ·d_total>/dR_I^l

        d<λ·d_total>/dR_I^l 分两部分：
          (1) 电子 AO 积分导数：Tr[dld_elec/dR_I^l · P]
              注：dip_grad_ao 使用 int1e_irp（<μ|r∇|ν>），与真正的偶极积分
              导数差一个 overlap 项 δ_{kl}·S·N_elec。该项被下面的核修正补偿。
          (2) 核偶极导数 + overlap 补偿：+Z_I·λ_l
              包含 d(λ·Σ_J Z_J R_J)/dR_I^l = Z_I·λ_l，
              同时隐式补偿 dip_grad_ao 的 overlap 缺失项。

        对中性分子，Σ_I grad_dse = 0（平动不变性）。
        """
        grad     = np.zeros(3)
        dm       = np.asarray(dm, dtype=float)
        ddip_all = self.dip_ints.dip_grad_ao(atm_id)
        Z_I      = self.mol.atom_charge(atm_id)
        for mode in self.cavity.modes:
            lam     = mode.lambda_vec
            ld      = self.dip_ints.lambda_dot_d(lam)
            ld_elec = float(np.einsum('pq,qp->', ld, dm))
            ld_nuc  = float(np.dot(lam, self.dip_ints.nuclear_dipole()))
            ld_mean = ld_elec + ld_nuc    # <λ·d_total>
            dld = np.einsum('k,klpq->lpq', lam, ddip_all)
            for l in range(3):
                d_ld_mean = float(np.einsum('pq,qp->', dld[l], dm))
                d_ld_mean += Z_I * lam[l]
                grad[l] += ld_mean * d_ld_mean
        return grad

    def grad_bilinear(self, dm: np.ndarray, atm_id: int) -> np.ndarray:
        """
        dE_bilinear/dR_I，shape (3,)，单位 Ha/Bohr。

        E_bilinear = -<λ·d_total>²/ω

        dE_bilinear/dR_I^l = (-2/ω)·<λ·d_total> · d<λ·d_total>/dR_I^l

        与 grad_dse 完全一致：ld_mean 用总偶极，核修正 +Z_I·λ_l。

        注意：系数 c = -2/ω（不是 -√(2/ω)），由能量表达式直接对 R 求导得到。
              旧版代码错误地使用了 -√(2/ω)，导致梯度在 IR 频率下低估约 16 倍。
        """
        grad     = np.zeros(3)
        dm       = np.asarray(dm, dtype=float)
        ddip_all = self.dip_ints.dip_grad_ao(atm_id)
        Z_I      = self.mol.atom_charge(atm_id)
        for mode in self.cavity.modes:
            lam     = mode.lambda_vec
            ld      = self.dip_ints.lambda_dot_d(lam)
            ld_elec = float(np.einsum('pq,qp->', ld, dm))
            ld_nuc  = float(np.dot(lam, self.dip_ints.nuclear_dipole()))
            ld_mean = ld_elec + ld_nuc    # <λ·d_total>
            c   = (-2.0 / mode.omega) * ld_mean
            dld = np.einsum('k,klpq->lpq', lam, ddip_all)
            for l in range(3):
                d_ld_mean = float(np.einsum('pq,qp->', dld[l], dm))
                d_ld_mean += Z_I * lam[l]
                grad[l] += c * d_ld_mean
        return grad

    # ------------------------------------------------------------------
    # 兼容接口
    # ------------------------------------------------------------------

    def update_photon_fields(self, dm: np.ndarray):
        """保留接口（相干态振幅在 get_vqed 内实时计算，无需预存储）。"""
        pass
