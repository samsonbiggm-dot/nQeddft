# -*- coding: utf-8 -*-
"""
nqeddft.cavity_field  —  连续谱光场离散化
==========================================

将具有频率分布的光场（连续谱或多峰谱）表示为
有限个 Pauli-Fierz 腔模的叠加，从而纳入现有
QED-DFT 框架（QEDRKS / QEDUKS / QEDTDA / QEDTDDFT）。

物理基础
--------
连续多模 Pauli-Fierz 哈密顿量（偶极近似）：

  H = H_e + ∫dω [ω a†a - √(ω/2) λ(ω)(ε·μ)(a†+a) + λ²(ω)/2 (ε·μ)²]

谱密度函数定义：

  J(ω) = π λ²(ω)   [原子单位]

使得：

  λ²_total = ∫₀^∞ J(ω)/π dω

离散化（N 个等面积矩形，中点规则）：

  ω_k = ω_min + (k + 0.5)·Δω,   k = 0,…,N-1
  λ_k = √(J_norm(ω_k)/π · Δω)

其中 J_norm 按用户指定的 lambda_total 归一化：

  ∫ J_norm(ω)/π dω = lambda_total²

支持的谱型
----------
  LorentzianField  —— 洛伦兹（单模腔/线宽展宽）
  GaussianField    —— 高斯（非均匀展宽/热辐射）
  OhmicField       —— Ohmic（声子浴/低频涨落）
  FlatbandField    —— 平顶谱（宽带激光/白光）
  MultiPeakField   —— 多峰叠加（任意离散峰）
  CustomField      —— 从数组或 CSV 文件导入实验谱

常用接口
--------
  field = LorentzianField(omega0=0.38, gamma=0.02, lambda_total=0.05)
  cav   = field.to_cavity(N=20, polarization=[0,0,1])
  # cav 是标准 Cavity 对象，可直接传入 QEDRKS / QEDTDA

  # 或者用单行工厂方法：
  cav = Cavity.from_lorentzian(omega0=0.38, gamma=0.02,
                                lambda_total=0.05, N=20)

单位约定
--------
  所有频率均为原子单位（a.u.）：1 a.u. ≈ 27.211 eV ≈ 219474 cm⁻¹
  lambda_total 无量纲（与 CavityMode.lambda_scalar 相同约定）

  换算辅助函数 cm1_to_au / ev_to_au / nm_to_au 在本模块末尾提供。
"""
from __future__ import annotations

import numpy as np

# numpy >= 2.0 移除了 trapz，统一用 trapezoid；兼容旧版
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import warnings
from typing import Sequence, Tuple, Optional, Union
from .cavity import Cavity


# ══════════════════════════════════════════════════════════════════════
# 单位换算辅助
# ══════════════════════════════════════════════════════════════════════

def cm1_to_au(nu_cm: float) -> float:
    """波数（cm⁻¹）→ 原子单位频率。"""
    return nu_cm / 219474.6306

def au_to_cm1(omega_au: float) -> float:
    """原子单位频率 → 波数（cm⁻¹）。"""
    return omega_au * 219474.6306

def ev_to_au(energy_ev: float) -> float:
    """电子伏特 → 原子单位能量/频率。"""
    return energy_ev / 27.2114

def au_to_ev(omega_au: float) -> float:
    """原子单位频率 → 电子伏特。"""
    return omega_au * 27.2114

def nm_to_au(wavelength_nm: float) -> float:
    """
    波长（nm）→ 原子单位角频率。
    ω = 2πc/λ，c = 137.036 a.u.（精细结构常数倒数），
    波长单位转换：1 nm = 18.897 Bohr。
    """
    c_au     = 137.035999084       # 光速（a.u.）
    lambda_au = wavelength_nm * 18.8973  # nm → Bohr
    return 2.0 * np.pi * c_au / lambda_au

def au_to_nm(omega_au: float) -> float:
    """原子单位角频率 → 波长（nm）。"""
    c_au = 137.035999084
    return 2.0 * np.pi * c_au / omega_au / 18.8973


# ══════════════════════════════════════════════════════════════════════
# 基类 SpectralField
# ══════════════════════════════════════════════════════════════════════

class SpectralField:
    """
    光场谱密度的抽象基类。

    子类只需实现 _spectral_shape(omega_array) → ndarray，
    返回未归一化的谱密度形状（在支撑区间外可返回 0）。
    其余归一化、离散化、to_cavity 均由基类提供。

    Parameters
    ----------
    lambda_total : float
        总耦合强度（无量纲），满足 ∫J(ω)/π dω = lambda_total²。
    omega_range  : (float, float)
        积分/离散化频率范围（a.u.）。若为 None，由子类的
        _default_range() 方法确定（通常取中心 ±5σ 或 ±5Γ）。
    """

    def __init__(self, lambda_total: float,
                 omega_range: Optional[Tuple[float, float]] = None):
        if lambda_total <= 0:
            raise ValueError(f"lambda_total={lambda_total} 必须 > 0")
        self.lambda_total = float(lambda_total)
        self._omega_range = omega_range  # None 表示由子类决定

    # ------------------------------------------------------------------
    # 子类接口
    # ------------------------------------------------------------------

    def _spectral_shape(self, omega: np.ndarray) -> np.ndarray:
        """
        未归一化的谱密度形状 J₀(ω)，shape 与 omega 相同。
        子类必须实现。支撑区间外返回 0。
        """
        raise NotImplementedError

    def _default_range(self) -> Tuple[float, float]:
        """子类提供的合理默认积分范围（a.u.）。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 归一化谱密度
    # ------------------------------------------------------------------

    @property
    def omega_range(self) -> Tuple[float, float]:
        if self._omega_range is not None:
            return self._omega_range
        return self._default_range()

    def spectral_density(self, omega: np.ndarray) -> np.ndarray:
        """
        归一化谱密度 J(ω)（a.u.），满足 ∫J(ω)/π dω = lambda_total²。

        Parameters
        ----------
        omega : array-like，频率（a.u.）

        Returns
        -------
        ndarray，J(ω)，shape 与 omega 相同
        """
        omega  = np.asarray(omega, dtype=float)
        J0     = self._spectral_shape(omega)
        J0     = np.where(J0 < 0, 0.0, J0)   # 保证非负

        # 数值积分归一化（高精度格点）
        o_min, o_max = self.omega_range
        o_grid = np.linspace(o_min, o_max, 4096)
        J0_grid = self._spectral_shape(o_grid)
        J0_grid = np.where(J0_grid < 0, 0.0, J0_grid)
        norm   = np.trapezoid(J0_grid / np.pi, o_grid)

        if norm < 1e-30:
            warnings.warn("谱密度积分近零，检查频率范围是否覆盖谱峰。")
            return np.zeros_like(omega)

        scale = self.lambda_total ** 2 / norm
        return J0 * scale

    def lambda_per_mode(self, omega_grid: np.ndarray) -> np.ndarray:
        """
        在给定频率格点上计算每个离散模式的耦合强度 λ_k。

        Parameters
        ----------
        omega_grid : 等间隔频率数组（N 个模式的中心频率）

        Returns
        -------
        lambda_array : shape (N,)
        """
        if len(omega_grid) < 2:
            raise ValueError("需要至少 2 个格点")
        delta_omega = omega_grid[1] - omega_grid[0]
        J = self.spectral_density(omega_grid)
        # λ_k = √(J(ω_k)/π · Δω)，负值截断为 0
        lam_sq = np.maximum(J / np.pi * delta_omega, 0.0)
        return np.sqrt(lam_sq)

    # ------------------------------------------------------------------
    # 转换为 Cavity
    # ------------------------------------------------------------------

    def to_cavity(self, N: int,
                  polarization: Sequence[float] = (0., 0., 1.),
                  name_prefix: str = "") -> Cavity:
        """
        将连续谱离散化为 N 个腔模，返回标准 Cavity 对象。

        Parameters
        ----------
        N            : 离散模式数，推荐 10–50（视谱宽而定）
        polarization : 所有模式共用的极化方向
        name_prefix  : 模式名称前缀（默认为类名缩写）

        Returns
        -------
        Cavity，可直接传入 QEDRKS / QEDTDA / QEDTDDFT

        Notes
        -----
        收敛性指导：
          - 窄谱（Γ < 0.01 a.u.）：N ≥ 20
          - 宽谱（Γ > 0.1 a.u.） ：N ≥ 10
          - 可用 SpectralField.convergence_check() 自动确定
        """
        if N < 1:
            raise ValueError(f"N={N} 必须 ≥ 1")

        o_min, o_max = self.omega_range
        if o_min <= 0:
            o_min = max(o_min, 1e-6)   # 频率必须正

        # 等间隔中点格点
        delta = (o_max - o_min) / N
        omega_grid = np.array([o_min + (k + 0.5) * delta for k in range(N)])
        lam_grid   = self.lambda_per_mode(omega_grid)

        prefix = name_prefix or self.__class__.__name__[:4].lower()
        cav = Cavity()
        for k, (om, lam) in enumerate(zip(omega_grid, lam_grid)):
            if lam > 0:   # 跳过零耦合模式
                cav.add_mode(float(om), float(lam),
                             polarization, name=f"{prefix}_{k}")

        if cav.n_modes == 0:
            raise RuntimeError("所有离散模式耦合强度均为零，请检查频率范围。")

        return cav

    def to_cavity_adaptive(self, N: int,
                            polarization: Sequence[float] = (0., 0., 1.),
                            threshold: float = 0.01) -> Cavity:
        """
        自适应离散化：在谱密度较大的区域加密。
        使用累积分布函数（CDF）的等分位点作为格点。

        Parameters
        ----------
        N         : 总模式数
        threshold : 低于最大谱密度 threshold 倍的模式舍弃
        """
        o_min, o_max = self.omega_range
        o_fine = np.linspace(max(o_min, 1e-6), o_max, 10000)
        J_fine = self.spectral_density(o_fine)
        J_fine = np.maximum(J_fine, 0.0)

        # 构建 CDF
        cdf = np.cumsum(J_fine)
        if cdf[-1] < 1e-30:
            raise RuntimeError("谱密度全零，无法自适应离散化")
        cdf /= cdf[-1]

        # 等分位点
        quantiles = np.linspace(0.5/N, 1 - 0.5/N, N)
        omega_grid = np.interp(quantiles, cdf, o_fine)

        # 在每个点上用局部 Δω 估算 λ_k
        lam_grid = []
        for k, om in enumerate(omega_grid):
            # 局部 Δω = 相邻点间距
            if k == 0:
                dw = omega_grid[1] - omega_grid[0]
            elif k == N - 1:
                dw = omega_grid[-1] - omega_grid[-2]
            else:
                dw = (omega_grid[k+1] - omega_grid[k-1]) / 2.0
            J_k = float(self.spectral_density(np.array([om]))[0])
            lam_grid.append(np.sqrt(max(J_k / np.pi * dw, 0.0)))

        prefix = self.__class__.__name__[:4].lower() + "ad"
        J_max  = J_fine.max()
        cav    = Cavity()
        for k, (om, lam) in enumerate(zip(omega_grid, lam_grid)):
            J_here = float(self.spectral_density(np.array([om]))[0])
            if lam > 0 and J_here >= threshold * J_max:
                cav.add_mode(float(om), float(lam),
                             polarization, name=f"{prefix}_{k}")

        if cav.n_modes == 0:
            raise RuntimeError("自适应离散化后无有效模式，请降低 threshold。")
        return cav

    # ------------------------------------------------------------------
    # 收敛性检验
    # ------------------------------------------------------------------

    def convergence_check(self, N_list: Sequence[int] = (5, 10, 20, 40),
                          polarization: Sequence[float] = (0., 0., 1.),
                          tol: float = 0.01) -> Tuple[int, dict]:
        """
        检验离散化收敛性：对不同 N 比较总耦合强度的误差。
        返回满足 tol 的最小 N 和各 N 的诊断信息。

        Parameters
        ----------
        N_list : 待测试的模式数列表
        tol    : |λ_total_reconstructed - λ_total| / λ_total 的容许误差

        Returns
        -------
        N_opt  : 满足收敛的最小 N（若全不满足，返回最大 N）
        info   : dict，各 N 的诊断结果
        """
        info   = {}
        N_opt  = max(N_list)
        target = self.lambda_total

        for N in sorted(N_list):
            cav = self.to_cavity(N, polarization)
            lam_sq_sum = sum(m.lambda_scalar**2 for m in cav.modes)
            lam_recon  = np.sqrt(lam_sq_sum)
            rel_err    = abs(lam_recon - target) / target
            info[N]    = {'lambda_recon': lam_recon, 'rel_err': rel_err}
            if rel_err < tol and N_opt == max(N_list):
                N_opt = N

        print(f"{'N':>6}  {'λ_recon':>10}  {'相对误差':>10}  {'状态':>6}")
        print("-" * 36)
        for N in sorted(N_list):
            r   = info[N]
            ok  = "✓" if r['rel_err'] < tol else "✗"
            print(f"{N:>6}  {r['lambda_recon']:>10.6f}  "
                  f"{r['rel_err']:>10.4e}  {ok:>6}")
        print(f"\n推荐 N = {N_opt}（误差阈值 {tol:.1e}）")
        return N_opt, info

    # ------------------------------------------------------------------
    # 文本诊断
    # ------------------------------------------------------------------

    def summary(self, N_pts: int = 200) -> str:
        """返回谱密度的文本摘要。"""
        o_min, o_max = self.omega_range
        omega = np.linspace(max(o_min, 1e-6), o_max, N_pts)
        J     = self.spectral_density(omega)
        idx_peak = int(np.argmax(J))
        lines = [
            f"{self.__class__.__name__}",
            f"  lambda_total = {self.lambda_total:.5f}",
            f"  频率范围     = [{o_min:.5f}, {o_max:.5f}] a.u.",
            f"               = [{au_to_cm1(o_min):.0f}, {au_to_cm1(o_max):.0f}] cm⁻¹",
            f"  谱峰位置     = {omega[idx_peak]:.5f} a.u. "
            f"({au_to_cm1(omega[idx_peak]):.0f} cm⁻¹)",
            f"  谱峰 J(ω₀)  = {J[idx_peak]:.4e} a.u.",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 洛伦兹谱
# ══════════════════════════════════════════════════════════════════════

class LorentzianField(SpectralField):
    """
    洛伦兹谱密度（单模腔 + 线宽展宽）。

      J₀(ω) = (Γ/2π) / [(ω - ω₀)² + (Γ/2)²]

    归一化后 ∫J₀/π dω = 1（全频段）。

    Parameters
    ----------
    omega0       : 中心频率（a.u.）
    gamma        : 半高全宽 FWHM（a.u.）
    lambda_total : 总耦合强度
    omega_range  : 积分范围，默认 [ω₀ - 10Γ, ω₀ + 10Γ]
    """

    def __init__(self, omega0: float, gamma: float,
                 lambda_total: float,
                 omega_range: Optional[Tuple[float,float]] = None):
        super().__init__(lambda_total, omega_range)
        if omega0 <= 0:
            raise ValueError(f"omega0={omega0} 必须 > 0")
        if gamma <= 0:
            raise ValueError(f"gamma={gamma} 必须 > 0")
        self.omega0 = float(omega0)
        self.gamma  = float(gamma)

    def _spectral_shape(self, omega: np.ndarray) -> np.ndarray:
        g2 = (self.gamma / 2.0) ** 2
        return (self.gamma / (2.0 * np.pi)) / ((omega - self.omega0)**2 + g2)

    def _default_range(self) -> Tuple[float, float]:
        hw = 10.0 * self.gamma
        return (max(self.omega0 - hw, 1e-6), self.omega0 + hw)

    def summary(self) -> str:
        base = super().summary()
        extra = (f"  ω₀ = {self.omega0:.5f} a.u. "
                 f"({au_to_cm1(self.omega0):.1f} cm⁻¹  "
                 f"/ {au_to_ev(self.omega0):.4f} eV)\n"
                 f"  Γ  = {self.gamma:.5f} a.u. "
                 f"({au_to_cm1(self.gamma):.1f} cm⁻¹)  "
                 f"Q = {self.omega0/self.gamma:.1f}")
        return base + "\n" + extra


# ══════════════════════════════════════════════════════════════════════
# 高斯谱
# ══════════════════════════════════════════════════════════════════════

class GaussianField(SpectralField):
    """
    高斯谱密度（非均匀展宽 / 热辐射）。

      J₀(ω) = exp[-(ω - ω₀)² / (2σ²)] / (σ√(2π))

    Parameters
    ----------
    omega0  : 中心频率（a.u.）
    sigma   : 标准差（a.u.），FWHM = 2.355σ
    lambda_total, omega_range : 同 SpectralField
    """

    def __init__(self, omega0: float, sigma: float,
                 lambda_total: float,
                 omega_range: Optional[Tuple[float,float]] = None):
        super().__init__(lambda_total, omega_range)
        if omega0 <= 0:
            raise ValueError(f"omega0={omega0} 必须 > 0")
        if sigma <= 0:
            raise ValueError(f"sigma={sigma} 必须 > 0")
        self.omega0 = float(omega0)
        self.sigma  = float(sigma)

    def _spectral_shape(self, omega: np.ndarray) -> np.ndarray:
        return (np.exp(-0.5 * ((omega - self.omega0) / self.sigma)**2)
                / (self.sigma * np.sqrt(2.0 * np.pi)))

    def _default_range(self) -> Tuple[float, float]:
        hw = 5.0 * self.sigma
        return (max(self.omega0 - hw, 1e-6), self.omega0 + hw)

    @property
    def fwhm(self) -> float:
        """高斯谱的 FWHM = 2√(2ln2)·σ。"""
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sigma


# ══════════════════════════════════════════════════════════════════════
# Ohmic 谱（声子浴）
# ══════════════════════════════════════════════════════════════════════

class OhmicField(SpectralField):
    """
    Ohmic 谱密度（声子浴 / 低频涨落）。

      J₀(ω) = (ω / ω_c) · exp(-ω / ω_c)

    归一化后 ∫₀^∞ J₀/π dω = ω_c/π。

    Parameters
    ----------
    omega_c      : 截止频率（a.u.）
    lambda_total : 总耦合强度
    s            : 指数（s=1 为 Ohmic，s<1 为 sub-Ohmic，s>1 为 super-Ohmic）
    omega_range  : 默认 [1e-4, 5·ω_c]
    """

    def __init__(self, omega_c: float, lambda_total: float,
                 s: float = 1.0,
                 omega_range: Optional[Tuple[float,float]] = None):
        super().__init__(lambda_total, omega_range)
        if omega_c <= 0:
            raise ValueError(f"omega_c={omega_c} 必须 > 0")
        self.omega_c = float(omega_c)
        self.s       = float(s)

    def _spectral_shape(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        with np.errstate(over='ignore'):
            return np.where(omega > 0,
                            (omega / self.omega_c)**self.s
                            * np.exp(-omega / self.omega_c),
                            0.0)

    def _default_range(self) -> Tuple[float, float]:
        return (1e-5, 8.0 * self.omega_c)


# ══════════════════════════════════════════════════════════════════════
# 平顶谱（宽带激光）
# ══════════════════════════════════════════════════════════════════════

class FlatbandField(SpectralField):
    """
    平顶（矩形）谱密度（宽带激光 / 白光腔）。

      J₀(ω) = 1 / (ω_max - ω_min)  for ω ∈ [ω_min, ω_max]
             = 0                    otherwise

    Parameters
    ----------
    omega_min, omega_max : 频段边界（a.u.）
    lambda_total : 总耦合强度
    """

    def __init__(self, omega_min: float, omega_max: float,
                 lambda_total: float):
        if omega_min <= 0:
            raise ValueError(f"omega_min={omega_min} 必须 > 0")
        if omega_max <= omega_min:
            raise ValueError("omega_max 必须 > omega_min")
        super().__init__(lambda_total, (omega_min, omega_max))
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

    def _spectral_shape(self, omega: np.ndarray) -> np.ndarray:
        dw = self.omega_max - self.omega_min
        return np.where((omega >= self.omega_min) & (omega <= self.omega_max),
                        1.0 / dw, 0.0)

    def _default_range(self) -> Tuple[float, float]:
        return (self.omega_min, self.omega_max)


# ══════════════════════════════════════════════════════════════════════
# 多峰叠加谱
# ══════════════════════════════════════════════════════════════════════

class MultiPeakField(SpectralField):
    """
    多峰叠加谱密度。

    每个峰可以是 Lorentzian 或 Gaussian，权重可以不同。
    适用于：法布里-珀罗腔的多个纵模、振动边带耦合等。

    Parameters
    ----------
    peaks : list of dict，每个峰的参数：
        {'omega0': float, 'width': float, 'weight': float,
         'shape': 'lorentzian'|'gaussian'}
        width 对 Lorentzian 为 FWHM Γ，对 Gaussian 为 σ
    lambda_total : 总耦合强度（分配到所有峰）
    omega_range  : 默认自动覆盖所有峰

    Examples
    --------
    >>> field = MultiPeakField(
    ...     peaks=[
    ...         {'omega0': 0.38, 'width': 0.01, 'weight': 1.0, 'shape': 'lorentzian'},
    ...         {'omega0': 0.42, 'width': 0.01, 'weight': 0.5, 'shape': 'lorentzian'},
    ...     ],
    ...     lambda_total=0.05,
    ... )
    """

    def __init__(self, peaks: list,
                 lambda_total: float,
                 omega_range: Optional[Tuple[float,float]] = None):
        super().__init__(lambda_total, omega_range)
        if not peaks:
            raise ValueError("peaks 列表不能为空")
        self.peaks = peaks
        # 验证
        for i, p in enumerate(peaks):
            for key in ('omega0', 'width', 'weight'):
                if key not in p:
                    raise ValueError(f"峰 {i} 缺少键 '{key}'")
            if p['weight'] <= 0:
                raise ValueError(f"峰 {i} weight 必须 > 0")

    def _spectral_shape(self, omega: np.ndarray) -> np.ndarray:
        J = np.zeros_like(omega, dtype=float)
        total_weight = sum(p['weight'] for p in self.peaks)
        for p in self.peaks:
            w0  = float(p['omega0'])
            w   = float(p['width'])
            wt  = float(p['weight']) / total_weight
            shape = p.get('shape', 'lorentzian').lower()
            if shape == 'lorentzian':
                g2 = (w / 2.0)**2
                J += wt * (w / (2.0 * np.pi)) / ((omega - w0)**2 + g2)
            elif shape == 'gaussian':
                J += wt * (np.exp(-0.5*((omega-w0)/w)**2)
                            / (w * np.sqrt(2.0*np.pi)))
            else:
                raise ValueError(f"未知峰型 '{shape}'，请用 'lorentzian' 或 'gaussian'")
        return J

    def _default_range(self) -> Tuple[float, float]:
        o0_vals = [p['omega0'] for p in self.peaks]
        w_vals  = [p['width']  for p in self.peaks]
        margin  = max(w_vals) * 8.0
        return (max(min(o0_vals) - margin, 1e-6),
                max(o0_vals) + margin)


# ══════════════════════════════════════════════════════════════════════
# 自定义谱（从实验数据导入）
# ══════════════════════════════════════════════════════════════════════

class CustomField(SpectralField):
    """
    从用户提供的数组或 CSV 文件导入实验谱密度。

    数据单位可以是原子单位（a.u.）或 cm⁻¹，通过 unit 参数指定。
    谱形状通过线性插值，在数据范围外为 0。

    Parameters
    ----------
    omega_data : array-like，频率数组
    J_data     : array-like，谱密度数组（形状，无需归一化）
    lambda_total : 总耦合强度
    unit         : 'au'（默认）或 'cm1'，omega_data 的单位
    omega_range  : 若为 None，取 [omega_data.min(), omega_data.max()]

    Examples
    --------
    >>> import numpy as np

# numpy >= 2.0 移除了 trapz，统一用 trapezoid；兼容旧版
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
    >>> nu = np.linspace(1500, 2000, 100)        # cm⁻¹
    >>> J  = np.exp(-((nu-1720)/50)**2)           # 高斯形状
    >>> field = CustomField(nu, J, lambda_total=0.03, unit='cm1')
    >>> cav = field.to_cavity(N=20, polarization=[0,0,1])

    从 CSV 文件导入：
    >>> field = CustomField.from_csv('spectrum.csv',
    ...                               lambda_total=0.03, unit='cm1')
    """

    def __init__(self, omega_data: Sequence[float],
                 J_data: Sequence[float],
                 lambda_total: float,
                 unit: str = 'au',
                 omega_range: Optional[Tuple[float,float]] = None):
        omega_arr = np.asarray(omega_data, dtype=float)
        J_arr     = np.asarray(J_data,     dtype=float)
        if omega_arr.shape != J_arr.shape:
            raise ValueError("omega_data 和 J_data 形状必须相同")
        if len(omega_arr) < 2:
            raise ValueError("至少需要 2 个数据点")

        # 单位转换
        if unit == 'cm1':
            omega_arr = cm1_to_au(omega_arr)
        elif unit == 'nm':
            omega_arr = nm_to_au(omega_arr)
            # nm → au 是单调递减，需要反转
            if omega_arr[0] > omega_arr[-1]:
                omega_arr = omega_arr[::-1]
                J_arr     = J_arr[::-1]
        elif unit != 'au':
            raise ValueError(f"unit='{unit}' 不支持，请用 'au'、'cm1' 或 'nm'")

        # 排序
        idx = np.argsort(omega_arr)
        self._omega_data = omega_arr[idx]
        self._J_data     = np.maximum(J_arr[idx], 0.0)

        r = (float(self._omega_data[0]), float(self._omega_data[-1]))
        super().__init__(lambda_total, omega_range or r)

    def _spectral_shape(self, omega: np.ndarray) -> np.ndarray:
        return np.interp(omega, self._omega_data, self._J_data,
                         left=0.0, right=0.0)

    def _default_range(self) -> Tuple[float, float]:
        return (float(self._omega_data[0]), float(self._omega_data[-1]))

    @classmethod
    def from_csv(cls, filepath: str, lambda_total: float,
                 unit: str = 'au',
                 omega_col: int = 0, J_col: int = 1,
                 skiprows: int = 1,
                 **kwargs) -> "CustomField":
        """
        从 CSV 文件读取谱密度数据。

        Parameters
        ----------
        filepath  : CSV 文件路径
        lambda_total : 总耦合强度
        unit      : 'au'、'cm1' 或 'nm'
        omega_col : 频率列序号（默认 0）
        J_col     : 谱密度列序号（默认 1）
        skiprows  : 跳过的标题行数（默认 1）
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows, **kwargs)
        return cls(data[:, omega_col], data[:, J_col],
                   lambda_total, unit=unit)


# ══════════════════════════════════════════════════════════════════════
# Cavity 工厂方法扩展（猴子补丁）
# ══════════════════════════════════════════════════════════════════════
# 将便捷工厂方法附加到 Cavity 类，使用时无需单独导入 SpectralField 子类。

def _cavity_from_lorentzian(cls, omega0: float, gamma: float,
                             lambda_total: float, N: int = 20,
                             polarization=(0.,0.,1.),
                             omega_range=None) -> "Cavity":
    """
    洛伦兹谱离散化工厂方法。

    Parameters
    ----------
    omega0       : 中心频率（a.u.）
    gamma        : 线宽 FWHM（a.u.）
    lambda_total : 总耦合强度
    N            : 离散模式数
    polarization : 极化方向
    omega_range  : 自定义频率范围（默认 [ω₀±10Γ]）

    Examples
    --------
    >>> cav = Cavity.from_lorentzian(
    ...     omega0 = cm1_to_au(1720),
    ...     gamma  = cm1_to_au(50),
    ...     lambda_total = 0.03,
    ...     N = 20,
    ... )
    """
    return LorentzianField(omega0, gamma, lambda_total,
                           omega_range).to_cavity(N, polarization)

def _cavity_from_gaussian(cls, omega0: float, sigma: float,
                           lambda_total: float, N: int = 20,
                           polarization=(0.,0.,1.),
                           omega_range=None) -> "Cavity":
    """高斯谱离散化工厂方法。sigma 为标准差（FWHM = 2.355σ）。"""
    return GaussianField(omega0, sigma, lambda_total,
                         omega_range).to_cavity(N, polarization)

def _cavity_from_flatband(cls, omega_min: float, omega_max: float,
                           lambda_total: float, N: int = 10,
                           polarization=(0.,0.,1.)) -> "Cavity":
    """平顶（宽带）谱离散化工厂方法。"""
    return FlatbandField(omega_min, omega_max,
                         lambda_total).to_cavity(N, polarization)

def _cavity_from_spectrum(cls, omega_data, J_data: np.ndarray,
                           lambda_total: float, N: int = 20,
                           polarization=(0.,0.,1.),
                           unit: str = 'au') -> "Cavity":
    """从自定义谱数据（数组）构建 Cavity。"""
    return CustomField(omega_data, J_data, lambda_total,
                       unit=unit).to_cavity(N, polarization)

# 附加到 Cavity 类
Cavity.from_lorentzian = classmethod(_cavity_from_lorentzian)
Cavity.from_gaussian   = classmethod(_cavity_from_gaussian)
Cavity.from_flatband   = classmethod(_cavity_from_flatband)
Cavity.from_spectrum   = classmethod(_cavity_from_spectrum)
