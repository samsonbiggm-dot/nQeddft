# -*- coding: utf-8 -*-
"""
nqeddft.phonon.qed_phonon
========================
光子-声子耦合：振动频率、IR/Raman 强度、腔诱导频率移动

物理：在 Born-Oppenheimer 近似下，腔光场通过修改势能面改变振动频率。
      核 Hessian = d^2 E_QED / dR_I dR_J（包含光子修正）。

两个计算层次：
  Level A（本文件，已实现）：
    - 数值 Hessian（对 QED-DFT 能量做中心差分）
    - 质量加权对角化 → 振动频率和正则模
    - 偶极矩导数 → IR 强度（含腔光场修正）
    - 腔诱导频率移动 Δω = ω_cav - ω_free

  Level B（Phase 4，待实现）：
    - Holstein/Jaynes-Cummings 模型的全量子声子-光子耦合
    - DMRG 求解混合光子-振动态
"""
import numpy as np
import warnings


class QEDPhonon:
    """
    腔调制振动分析。

    Parameters
    ----------
    mf : QEDRKS（已收敛的 QED-DFT 基态对象）

    Examples
    --------
    >>> ph = QEDPhonon(mf)
    >>> freqs, modes = ph.harmonic_analysis()   # 需要先调用 numerical_hessian()
    >>> result = ph.cavity_shift()               # 计算腔诱导频率移动
    >>> ir = ph.ir_intensities()                 # IR 强度
    """

    # 单位换算常量
    AMU_TO_AU    = 1822.88848    # 原子质量单位 -> 电子质量
    AU_TO_CM1    = 219474.6      # a.u. 频率 -> cm^-1
    BOHR_TO_ANG  = 0.529177      # Bohr -> Angstrom

    def __init__(self, mf):
        self.mf      = mf
        self.cavity  = mf.cavity
        self._hess   = None
        self._freqs  = None
        self._modes  = None    # 质量加权本征矢 (3N, 3N)

    # ------------------------------------------------------------------
    # Level A-1：数值 Hessian
    # ------------------------------------------------------------------

    def numerical_hessian(self, stepsize: float = 0.001,
                           verbose: bool = True) -> np.ndarray:
        """
        通过中心差分计算 QED-DFT 能量的数值 Hessian 矩阵。

        使用能量差分（而非梯度差分），避免梯度实现中的潜在 bug。
        每个坐标需要 2 次 SCF，共 6*natm 次，计算量较大。

        Parameters
        ----------
        stepsize : 有限差分步长，单位 Bohr（推荐 0.001-0.005）
        verbose  : 是否显示进度

        Returns
        -------
        ndarray, shape (3*natm, 3*natm), 单位 Ha/Bohr^2
        """
        from ..scf.qed_rks import QEDRKS

        mf0    = self.mf
        mol    = mf0.mol
        natm   = mol.natm
        ndim   = 3 * natm
        hess   = np.zeros((ndim, ndim))
        coords0 = mol.atom_coords().copy()   # shape (natm, 3), Bohr

        if verbose:
            print(f"计算 QED-DFT 数值 Hessian（{natm} 个原子，"
                  f"步长 {stepsize} Bohr）...")
            print(f"共需 {ndim * 2} 次 SCF 计算")

        # 中心能量（用于对角项验证）
        e0 = mf0.e_tot

        for i in range(natm):
            for j in range(3):
                idx = 3 * i + j
                if verbose:
                    print(f"  坐标 ({i},{j}) [{idx+1}/{ndim}]", end='\r')

                # 正向扰动
                c_p = coords0.copy()
                c_p[i, j] += stepsize
                e_p = self._single_point(mf0, c_p)

                # 反向扰动
                c_m = coords0.copy()
                c_m[i, j] -= stepsize
                e_m = self._single_point(mf0, c_m)

                # 对角元：d^2E/dR_ij^2 ≈ (E+ - 2E0 + E-) / h^2
                hess[idx, idx] = (e_p - 2.0 * e0 + e_m) / stepsize**2

                # 混合元通过双重差分（更准确）
                for i2 in range(natm):
                    for j2 in range(3):
                        idx2 = 3 * i2 + j2
                        if idx2 <= idx:
                            continue   # 利用对称性，只算上三角
                        c_pp = coords0.copy(); c_pp[i,j]+=stepsize; c_pp[i2,j2]+=stepsize
                        c_pm = coords0.copy(); c_pm[i,j]+=stepsize; c_pm[i2,j2]-=stepsize
                        c_mp = coords0.copy(); c_mp[i,j]-=stepsize; c_mp[i2,j2]+=stepsize
                        c_mm = coords0.copy(); c_mm[i,j]-=stepsize; c_mm[i2,j2]-=stepsize
                        e_pp = self._single_point(mf0, c_pp)
                        e_pm = self._single_point(mf0, c_pm)
                        e_mp = self._single_point(mf0, c_mp)
                        e_mm = self._single_point(mf0, c_mm)
                        h_ij = (e_pp - e_pm - e_mp + e_mm) / (4.0 * stepsize**2)
                        hess[idx, idx2] = h_ij
                        hess[idx2, idx] = h_ij

        # 恢复原始坐标
        self._single_point(mf0, coords0, restore=True)
        # 对称化（消除数值不对称）
        self._hess = 0.5 * (hess + hess.T)
        if verbose:
            print(f"\n  Hessian 计算完成，形状 {self._hess.shape}")
        return self._hess

    def numerical_hessian_fast(self, stepsize: float = 0.001,
                                verbose: bool = True) -> np.ndarray:
        """
        快速数值 Hessian：基于梯度差分（推荐用于大体系）。
        每个坐标需 2 次梯度计算，共 6*natm 次 SCF + 梯度。
        计算量约为完整双重差分的 1/3N，但需要梯度实现正确。
        """
        from ..grad.qed_grad import QEDGradients

        mf0     = self.mf
        mol     = mf0.mol
        natm    = mol.natm
        ndim    = 3 * natm
        hess    = np.zeros((ndim, ndim))
        coords0 = mol.atom_coords().copy()

        if verbose:
            print(f"快速数值 Hessian（梯度差分，{ndim} 自由度）...")

        for i in range(natm):
            for j in range(3):
                idx = 3 * i + j
                if verbose:
                    print(f"  坐标 ({i},{j}) [{idx+1}/{ndim}]", end='\r')

                # 正向扰动 → 梯度
                c_p = coords0.copy(); c_p[i, j] += stepsize
                g_p = self._gradient(mf0, c_p)

                # 反向扰动 → 梯度
                c_m = coords0.copy(); c_m[i, j] -= stepsize
                g_m = self._gradient(mf0, c_m)

                # hess[:, idx] = (g+ - g-) / (2h)
                hess[:, idx] = (g_p.ravel() - g_m.ravel()) / (2.0 * stepsize)

        self._single_point(mf0, coords0, restore=True)
        self._hess = 0.5 * (hess + hess.T)
        if verbose:
            print(f"\n  完成，Hessian shape = {self._hess.shape}")
        return self._hess

    # ------------------------------------------------------------------
    # Level A-2：谐振子分析
    # ------------------------------------------------------------------

    def harmonic_analysis(self, hess: np.ndarray = None) -> tuple:
        """
        对质量加权 Hessian 对角化，获得振动频率和正则模式。

        Parameters
        ----------
        hess : 可选，外部提供的 Hessian（否则自动调用 numerical_hessian_fast）

        Returns
        -------
        freqs : ndarray, shape (3*natm,), 单位 cm^-1
                负值表示虚频（鞍点方向）
        modes : ndarray, shape (3*natm, 3*natm)
                列向量为笛卡尔坐标下的正则模式（质量未加权）
        """
        if hess is None:
            if self._hess is None:
                warnings.warn(
                    "未提供 Hessian，调用 numerical_hessian_fast()。"
                    "如需更高精度请先显式调用 numerical_hessian()。"
                )
                hess = self.numerical_hessian_fast()
            else:
                hess = self._hess

        mol    = self.mf.mol
        masses = mol.atom_mass_list(isotope_avg=True)   # (natm,), amu
        # 质量加权因子 M^{-1/2}，shape (3*natm,)
        m_inv  = np.repeat(1.0 / np.sqrt(masses * self.AMU_TO_AU), 3)

        # 质量加权 Hessian
        mwh              = np.outer(m_inv, m_inv) * hess
        evals, evecs_mw  = np.linalg.eigh(mwh)

        # 频率：sqrt(|eigenvalue|) * sign，转 cm^-1
        freq_au = np.sign(evals) * np.sqrt(np.abs(evals))
        freqs   = freq_au * self.AU_TO_CM1

        # 笛卡尔正则模式 = M^{-1/2} * 质量加权本征矢
        modes   = evecs_mw * m_inv[:, None]  # 每列归一化为笛卡尔位移

        self._freqs      = freqs
        self._modes      = modes
        self._evecs_mw   = evecs_mw
        self._masses     = masses
        return freqs, modes

    def print_frequencies(self, min_freq: float = 50.0):
        """打印振动频率（过滤平动/转动）。"""
        if self._freqs is None:
            raise RuntimeError("请先调用 harmonic_analysis()")
        print("振动频率 (cm^-1)：")
        print(f"{'模式':>5}  {'频率':>10}  {'类型':>8}")
        print("-" * 28)
        for i, f in enumerate(self._freqs):
            if abs(f) < min_freq:
                continue
            ftype = "虚频" if f < 0 else "实频"
            print(f"{i:>5}  {f:>10.2f}  {ftype:>8}")

    # ------------------------------------------------------------------
    # Level A-3：IR 强度
    # ------------------------------------------------------------------

    def ir_intensities(self, stepsize: float = 0.001) -> dict:
        """
        计算 IR 强度（含腔光场修正）。

        I_k ∝ |d<μ>/dQ_k|^2，通过偶极矩对正则模式的导数计算。
        使用有限差分沿每个正则模式方向扰动几何。

        Parameters
        ----------
        stepsize : 正则模式方向的扰动幅度（dimensionless，质量加权坐标）

        Returns
        -------
        dict:
            freqs_cm      : ndarray, 振动频率 (cm^-1)
            ir_intensity  : ndarray, IR 强度（a.u.^2/amu，与实验 km/mol 成正比）
            dmu_dQ        : ndarray, shape (3N, 3), 偶极矩导数
        """
        if self._freqs is None:
            raise RuntimeError("请先调用 harmonic_analysis()")

        mol     = self.mf.mol
        natm    = mol.natm
        ndim    = 3 * natm
        masses  = self._masses
        coords0 = mol.atom_coords().copy()

        # 偶极矩梯度 dmu/dR，shape (3*natm, 3)
        dmu_dR = np.zeros((ndim, 3))
        for i in range(natm):
            for j in range(3):
                idx = 3 * i + j
                c_p = coords0.copy(); c_p[i, j] += stepsize
                mu_p = self._dipole(c_p)
                c_m = coords0.copy(); c_m[i, j] -= stepsize
                mu_m = self._dipole(c_m)
                dmu_dR[idx] = (mu_p - mu_m) / (2.0 * stepsize)

        self._single_point(self.mf, coords0, restore=True)

        # 沿正则模式的偶极矩导数：dmu/dQ_k = sum_il L_il^k * dmu_i/dR_l
        # L = self._modes（笛卡尔正则模式矩阵，列为模式）
        # dmu/dQ = L^T @ dmu_dR
        dmu_dQ   = self._modes.T @ dmu_dR   # (3N, 3)

        # IR 强度 ∝ |dmu/dQ|^2
        ir_int   = np.sum(dmu_dQ ** 2, axis=1)   # (3N,)

        return {
            'freqs_cm':     self._freqs,
            'ir_intensity': ir_int,
            'dmu_dQ':       dmu_dQ,
            'modes':        self._modes,
        }

    def print_ir_spectrum(self, min_freq: float = 50.0):
        """打印 IR 光谱（频率 + 强度）。"""
        result = self.ir_intensities()
        freqs  = result['freqs_cm']
        irs    = result['ir_intensity']
        print("IR 光谱（含腔光场修正）：")
        print(f"{'模式':>5}  {'频率/cm-1':>12}  {'IR 强度':>14}")
        print("-" * 36)
        for i, (f, ir) in enumerate(zip(freqs, irs)):
            if abs(f) < min_freq:
                continue
            print(f"{i:>5}  {f:>12.2f}  {ir:>14.6e}")

    # ------------------------------------------------------------------
    # Level A-4：腔诱导频率移动
    # ------------------------------------------------------------------

    def cavity_shift(self, verbose: bool = True) -> dict:
        """
        计算腔对振动频率的移动：Δω_k = ω_k^cav - ω_k^free。

        原理：在相同分子几何下，分别用 lambda=0（无耦合参考）
              和实际 lambda 做振动分析，对比频率。

        Returns
        -------
        dict:
            freqs_cav  : ndarray, 腔中振动频率 (cm^-1)
            freqs_free : ndarray, 自由空间振动频率 (cm^-1)
            shift_cm   : ndarray, 频率移动量 (cm^-1)
        """
        from ..scf.qed_rks import QEDRKS
        from ..cavity import Cavity

        mol = self.mf.mol

        # 计算腔中频率（如果还没有）
        if self._freqs is None:
            if verbose:
                print("计算腔中振动频率...")
            hess_cav = self.numerical_hessian_fast()
            freqs_cav, _ = self.harmonic_analysis(hess_cav)
        else:
            freqs_cav = self._freqs

        # 构建零耦合腔（参考态）
        cav0 = Cavity()
        for m in self.cavity.modes:
            cav0.add_mode(m.omega, 0.0, m.polarization, m.name + "_ref")

        mf0 = QEDRKS(mol, cav0)
        mf0.xc      = self.mf.xc
        mf0.verbose = 0
        if verbose:
            print("计算自由空间振动频率（lambda=0 参考态）...")
        mf0.kernel()

        ph0 = QEDPhonon(mf0)
        hess_free = ph0.numerical_hessian_fast(verbose=verbose)
        freqs_free, _ = ph0.harmonic_analysis(hess_free)

        shift = freqs_cav - freqs_free

        if verbose:
            print(f"\n{'模式':>5}  {'ω_free/cm-1':>14}  "
                  f"{'ω_cav/cm-1':>14}  {'Δω/cm-1':>10}")
            print("-" * 50)
            for i, (f0, fc, ds) in enumerate(zip(freqs_free, freqs_cav, shift)):
                if abs(f0) < 50 and abs(fc) < 50:
                    continue
                print(f"{i:>5}  {f0:>14.2f}  {fc:>14.2f}  {ds:>+10.3f}")

        return {
            'freqs_cav':   freqs_cav,
            'freqs_free':  freqs_free,
            'shift_cm':    shift,
        }

    # ------------------------------------------------------------------
    # 振动耦合强度
    # ------------------------------------------------------------------

    def vibrational_coupling_strength(self) -> dict:
        """
        计算各振动模式与腔光场的耦合强度（振动 Rabi 频率）。

        g_k^a = sqrt(w_a/2) * lam_a * |d<mu>/dQ_k|   (单位 a.u.)
        即振动跃迁偶极矩在腔极化方向的投影乘以腔耦合强度。

        Returns
        -------
        dict: {
            'g_vib': ndarray shape (n_modes_cav, 3N), 各腔模对各振动模式的耦合,
            'rabi_vib': ndarray, 振动 Rabi 劈裂估算 (cm^-1),
        }
        """
        ir_result = self.ir_intensities()
        dmu_dQ    = ir_result['dmu_dQ']   # (3N, 3)
        freqs     = ir_result['freqs_cm']

        g_vib    = np.zeros((self.cavity.n_modes, len(freqs)))
        rabi_vib = np.zeros((self.cavity.n_modes, len(freqs)))

        for ia, mode in enumerate(self.cavity.modes):
            lam = mode.lambda_vec   # (3,)
            for k in range(len(freqs)):
                # 耦合强度：g_k = sqrt(w/2) * |lam · dmu/dQ_k|
                g = mode.sqrt_omega_half * abs(float(np.dot(lam, dmu_dQ[k])))
                g_vib[ia, k]    = g
                # 振动 Rabi 劈裂：Omega_R = 2*g，转 cm^-1
                rabi_vib[ia, k] = 2.0 * g * self.AU_TO_CM1

        return {
            'g_vib':    g_vib,
            'rabi_vib': rabi_vib,
            'freqs_cm': freqs,
        }

    def print_coupling_strength(self, min_freq: float = 50.0):
        """打印各振动模式的腔耦合强度。"""
        result = self.vibrational_coupling_strength()
        g_vib    = result['g_vib']
        rabi_vib = result['rabi_vib']
        freqs    = result['freqs_cm']

        print("振动-腔耦合强度：")
        for ia, mode in enumerate(self.cavity.modes):
            print(f"\n  腔模 [{mode.name}] ω={mode.omega_ev():.4f} eV，"
                  f"λ={mode.lambda_scalar:.4f}")
            print(f"  {'振动模式':>6}  {'频率/cm-1':>12}  "
                  f"{'g (a.u.)':>12}  {'Ω_R (cm-1)':>12}")
            print("  " + "-" * 46)
            for k, (f, g, rabi) in enumerate(
                    zip(freqs, g_vib[ia], rabi_vib[ia])):
                if abs(f) < min_freq or g < 1e-8:
                    continue
                print(f"  {k:>6}  {f:>12.2f}  {g:>12.6e}  {rabi:>12.4f}")

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _single_point(self, mf, coords: np.ndarray,
                      restore: bool = False) -> float:
        """
        在给定坐标下运行一次 QED-DFT SCF，返回总能量。
        修改 mol 的几何后调用 mf.kernel()，使用已有 MO 作为初猜。
        """
        mol = mf.mol
        mol.set_geom_(coords, unit='Bohr')
        mol.build()
        # 清除积分缓存（几何已改变）
        if hasattr(mf, '_pf'):
            mf._pf.dip_ints.invalidate_cache()
        if not restore:
            mf.verbose = 0
            mf.kernel()
            return mf.e_tot
        else:
            mf.kernel()
            return mf.e_tot

    def _gradient(self, mf, coords: np.ndarray) -> np.ndarray:
        """在给定坐标下计算 QED-DFT 核梯度，shape (natm, 3)。"""
        from ..grad.qed_grad import QEDGradients
        mol = mf.mol
        mol.set_geom_(coords, unit='Bohr')
        mol.build()
        if hasattr(mf, '_pf'):
            mf._pf.dip_ints.invalidate_cache()
        mf.verbose = 0
        mf.kernel()
        g_obj       = QEDGradients(mf)
        g_obj.verbose = 0
        return g_obj.kernel()

    def _dipole(self, coords: np.ndarray) -> np.ndarray:
        """在给定坐标下计算全偶极矩（电子+核），shape (3,)。"""
        mf  = self.mf
        mol = mf.mol
        mol.set_geom_(coords, unit='Bohr')
        mol.build()
        if hasattr(mf, '_pf'):
            mf._pf.dip_ints.invalidate_cache()
        mf.verbose = 0
        mf.kernel()
        dm = mf.make_rdm1()
        return mf._pf.dip_ints.total_dipole(dm)
