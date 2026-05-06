# -*- coding: utf-8 -*-
"""
nqeddft.tst.tunneling
=====================
量子隧穿修正因子 κ(T)。

实现两种近似：
  Wigner :  κ(T) = 1 + (1/24)(ℏω‡/kT)²
            一阶近似，仅依赖虚频，对低 ν‡ 适用。
  Eckart :  完整数值积分，需要正向/反向能垒高度 + 虚频。
            对 H 转移反应在 200-500 K 范围内推荐。

腔依赖
------
当腔修饰反应坐标的虚频 ν‡(λ) 或修饰反应能垒高度时，κ 自动反映。
腔通过两条路径影响隧穿：
  (1) 修改 V_f, V_r       → Eckart 透射几率改变
  (2) 修改 ν‡             → 隧穿"穿透深度"改变

参考：
  Wigner, Z. Phys. Chem. B 19, 203 (1932)
  Eckart, Phys. Rev. 35, 1303 (1930)
  Johnston, J. Chem. Phys. 35, 1854 (1961)
  Garrett & Truhlar, JPC 83, 1052 (1979) eq. (5)-(7)  ← 本实现采用
  Truhlar et al., POLYRATE manual, https://comp.chem.umn.edu/polyrate/
"""
from __future__ import annotations

import numpy as np

# numpy >= 2.0 移除了 trapz，统一用 trapezoid；兼容旧版
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from typing import Optional

from .thermo import K_B_AU, CM1_TO_AU


# ══════════════════════════════════════════════════════════════════════
# Wigner 修正
# ══════════════════════════════════════════════════════════════════════

def wigner_correction(imag_freq_cm: float, T: float) -> float:
    """
    Wigner 一阶隧穿修正：
      κ(T) = 1 + (1/24)(ℏ|ω‡| / kT)²

    Parameters
    ----------
    imag_freq_cm : 反应坐标虚频幅度 (cm⁻¹)，正数
    T            : 温度 (K)

    Returns
    -------
    kappa : 修正因子（无量纲，≥1）

    备注：仅在 ℏω‡/kT < 2π 时合理。对 H 转移反应（ν‡~1500 cm⁻¹）
    在 T < 200 K 时 Wigner 严重低估，需用 Eckart。
    """
    if imag_freq_cm <= 0:
        return 1.0
    if T <= 0:
        return float('inf')
    omega_au = imag_freq_cm * CM1_TO_AU
    kT = K_B_AU * T
    x  = omega_au / kT
    return 1.0 + x * x / 24.0


# ══════════════════════════════════════════════════════════════════════
# Eckart 修正（采用 Garrett-Truhlar 1979 公式）
# ══════════════════════════════════════════════════════════════════════
#
# Eckart 不对称势：
#
#   V(s) = (A·y)/(1 + y) + (B·y)/(1 + y)²,   y = exp(2π s/L)
#
#   A = V_f - V_r              （能差，不对称度）
#   B = (√V_f + √V_r)²         （形状强度）
#   L = 势垒"特征长度"          （由虚频确定）
#
# 关键关系：势垒最大值处的曲率 V''(s*) 与虚频 ω* 关联（μ=1 in mass-weighted）：
#
#   ω*² = 8 V_f V_r (B - A) / (L² · B²)  ... Johnston (1961)
#   ⇒ L² = 8 V_f V_r (B - A) / (B² · ω*²)
#
# 形状参数（Garrett-Truhlar 1979 eq. (6)-(7)）：
#
#   α_f = (2/ℏω*) · √[2 V_f² · B / (B² - A²)]
#   α_r = (2/ℏω*) · √[2 V_r² · B / (B² - A²)]
#
# 注意：B² - A² = (B-A)(B+A) > 0 always (B > |A| 由 (√V_f+√V_r)² > |V_f-V_r| 保证)。
#
# 透射系数（eq. (5)）：
#   2π·a = α_f · √(E/V_f)
#   2π·b = α_r · √((E - A)/V_r),   when E ≥ A; otherwise P = 0
#   2π·d = π · √[(α_f + α_r)² - 1]    if (α_f+α_r) > 1
#         (α_f+α_r) < 1 时 d 为虚数, cosh→cos
#
#   P(E) = [cosh(2π(a+b)) - cosh(2π(a-b))] / [cosh(2π(a+b)) + cosh(2πd)]
#
# κ(T) = (β·exp(βV_f)) · ∫₀^∞ P(E) exp(-βE) dE,  β = 1/kT
# ══════════════════════════════════════════════════════════════════════


def _eckart_alpha(V_f: float, V_r: float,
                  omega_imag_au: float) -> tuple:
    """
    计算 Eckart 形状参数 α_f, α_r（无量纲）。

    **Johnston (1961, JCP 35, 1854) eq. (16)** 标准形式：

      α_f = (2π/ℏω*) · √[2 V_f / (1/√V_f + 1/√V_r)²]
      α_r = (2π/ℏω*) · √[2 V_r / (1/√V_f + 1/√V_r)²]

    令 C = (1/√V_f + 1/√V_r)²，则 α_f = (2π/ω*)·√(2V_f/C)。

    对称势 V_f = V_r = V* 校验：
      C = (2/√V*)² = 4/V*
      α_f = (2π/ω*)·√(2V*·V*/4) = (2π/ω*)·V*/√2 = π√2·V*/ω* ≈ 4.443·V*/ω*
    与 Bell (1980) 对称势 α' = 4V*/ω* 数值上一致（Bell 用 √(8...)/π 略不同惯例）。

    对 H+H₂ 体系 V*=9.7 kcal/mol, ν*=1500i cm⁻¹:
      α' ≈ 9.05（Bell 形式）或 ≈ 10.05（Johnston 2π 形式），数量级相同。
    势垒顶 P(V*) ≈ 0.5。

    返回 (alpha_f, alpha_r, A, B)
    """
    A = V_f - V_r
    B = (np.sqrt(V_f) + np.sqrt(V_r))**2

    # Johnston eq. (16): C = (V_f^{-1/2} + V_r^{-1/2})^2
    inv_sqrt_sum = 1.0 / np.sqrt(V_f) + 1.0 / np.sqrt(V_r)
    C = inv_sqrt_sum * inv_sqrt_sum

    if C <= 0 or omega_imag_au <= 0:
        return 0.0, 0.0, A, B

    fac = 2.0 * np.pi / omega_imag_au
    alpha_f = fac * np.sqrt(2.0 * V_f / C)
    alpha_r = fac * np.sqrt(2.0 * V_r / C)
    return alpha_f, alpha_r, A, B


def _eckart_P(E_au: float, V_f: float, V_r: float,
              omega_imag_au: float) -> float:
    """
    Eckart 不对称势透射概率 P(E)。

    **Johnston (1961, JCP 35, 1854) eq. (15)**:

        P(E) = [cosh(2π(a+b)) - cosh(2π(a-b))] /
               [cosh(2π(a+b)) + cosh(2πd)]

    其中：
        2πa = α_f · √(E/V_f)
        2πb = α_r · √((E - A)/V_r),    A = V_f - V_r, 要求 E ≥ A
        2πd = √[α_f·α_r ·(α_f·α_r/(α_f+α_r)² · ?) ...]

    Johnston eq. (17) d 参数：
        2πd = √[(α_f + α_r)² - π²]      （注意是 -π², 不是 -1）

    对称势 V_f=V_r 时 (α_f+α_r)² = 4α² ≈ 4·100 = 400, 减 π² ≈ 9.87
    很小修正，所以 d ≈ α_f+α_r。

    E < A（反向通道关闭）时 P = 0。
    """
    if V_f <= 0 or V_r <= 0 or omega_imag_au <= 0:
        return 0.0 if E_au < V_f else 1.0
    if E_au < 0:
        return 0.0

    A = V_f - V_r
    if E_au < A:
        # 反向通道阈值未达
        return 0.0

    alpha_f, alpha_r, _, _ = _eckart_alpha(V_f, V_r, omega_imag_au)
    if alpha_f == 0.0 or alpha_r == 0.0:
        return 0.0

    xi_f = E_au / V_f
    xi_r = (E_au - A) / V_r
    two_pi_a = alpha_f * np.sqrt(max(xi_f, 0.0))
    two_pi_b = alpha_r * np.sqrt(max(xi_r, 0.0))

    # Johnston eq. (17): 2πd = √[(α_f + α_r)² - π²]
    sum_alpha_sq = (alpha_f + alpha_r)**2
    pi_sq = np.pi * np.pi
    if sum_alpha_sq >= pi_sq:
        two_pi_d = np.sqrt(sum_alpha_sq - pi_sq)
        cosh_2pid = np.cosh(min(two_pi_d, 700.0))
    else:
        two_pi_d = np.sqrt(pi_sq - sum_alpha_sq)
        cosh_2pid = np.cos(two_pi_d)

    arg_plus  = two_pi_a + two_pi_b
    arg_minus = abs(two_pi_a - two_pi_b)

    # 极大 arg 时用 log-domain 防溢出
    if arg_plus > 700.0:
        diff = arg_minus - arg_plus  # ≤ 0
        cosh_d_scaled = cosh_2pid * np.exp(-arg_plus)
        num = 1.0 - np.exp(diff)
        den = 1.0 + 2.0 * cosh_d_scaled
        if den <= 0:
            return 0.0
        return float(np.clip(num / den, 0.0, 1.0))

    cosh_plus  = np.cosh(arg_plus)
    cosh_minus = np.cosh(arg_minus)
    num = cosh_plus - cosh_minus
    den = cosh_plus + cosh_2pid

    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def eckart_correction(V_f: float, V_r: float,
                      imag_freq_cm: float, T: float,
                      n_quad: int = 400) -> float:
    """
    Eckart 隧穿修正因子（Boltzmann 加权数值积分）：

      κ(T) = β · exp(β V_f) · ∫₀^∞ P(E) · exp(-β E) dE,   β = 1/kT

    Parameters
    ----------
    V_f : 正向势垒高度 (Ha)，反应物 → TS 能量差（正数）
    V_r : 反向势垒高度 (Ha)，产物 → TS 能量差（正数）
    imag_freq_cm : 反应坐标虚频 (cm⁻¹)，正数（绝对值）
    T   : 温度 (K)
    n_quad : 积分格点数

    Returns
    -------
    kappa : Eckart 修正因子（无量纲，通常 > 1）
    """
    if T <= 0 or V_f <= 0 or V_r <= 0 or imag_freq_cm <= 0:
        return 1.0

    omega_au = imag_freq_cm * CM1_TO_AU
    kT = K_B_AU * T
    E_max = max(V_f, V_r) + 30.0 * kT

    # 在 V_f 附近加密格点（透射快速变化区）
    n1 = n_quad // 2
    n2 = n_quad - n1
    E_low  = np.linspace(0.0, V_f * 1.5, n1, endpoint=False)
    E_high = np.linspace(V_f * 1.5, E_max, n2)
    E_grid = np.concatenate([E_low, E_high])

    P_grid = np.array([_eckart_P(E, V_f, V_r, omega_au) for E in E_grid])

    # 用 (1/kT)·exp((V_f - E)/kT)·P 形式避免大数相乘
    log_weight = (V_f - E_grid) / kT
    # 仅取 log_weight 不会溢出的部分 (< ~700)
    safe = log_weight < 700.0
    if not safe.any():
        return 1.0
    integrand = np.zeros_like(E_grid)
    integrand[safe] = P_grid[safe] * np.exp(log_weight[safe])

    integral = np.trapezoid(integrand, E_grid)
    kappa = integral / kT
    return float(kappa)


# ══════════════════════════════════════════════════════════════════════
# 一站式接口
# ══════════════════════════════════════════════════════════════════════

def tunneling_kappa(imag_freq_cm: float, T: float,
                    V_f: Optional[float] = None,
                    V_r: Optional[float] = None,
                    method: str = 'auto') -> dict:
    """
    隧穿修正因子（统一接口）。

    Parameters
    ----------
    imag_freq_cm : 反应坐标虚频幅度 (cm⁻¹)
    T            : 温度 (K)
    V_f, V_r     : 正/反势垒高度 (Ha)，Eckart 必需
    method :
        'wigner' : Wigner 一阶
        'eckart' : 完整 Eckart（需 V_f, V_r）
        'auto'   : 有 V_f, V_r 时用 eckart，否则 wigner

    Returns
    -------
    dict: kappa, method_used, imag_freq, T
    """
    if method == 'auto':
        method = 'eckart' if (V_f is not None and V_r is not None) else 'wigner'

    if method == 'wigner':
        kappa = wigner_correction(imag_freq_cm, T)
    elif method == 'eckart':
        if V_f is None or V_r is None:
            raise ValueError("Eckart 需要 V_f 和 V_r")
        kappa = eckart_correction(V_f, V_r, imag_freq_cm, T)
    else:
        raise ValueError(f"未知 method: {method}")

    return {
        'kappa':       kappa,
        'method_used': method,
        'imag_freq':   imag_freq_cm,
        'T':           T,
    }
