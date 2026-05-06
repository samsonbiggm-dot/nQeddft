# -*- coding: utf-8 -*-
"""
nqeddft.tst.thermo
==================
振动/平动/转动热力学量计算（QED 增强版）。

物理基础
--------
谐振子近似 + 刚性转子近似 + 理想气体平动。
对于过渡态（含一个虚频），虚频被排除在配分函数外（它对应反应坐标）。

腔依赖性
--------
所有"腔依赖"完全通过传入的 freqs 数组体现：
  - 自由空间：freqs = freqs_free（来自 QEDPhonon, λ=0）
  - 腔中：freqs = freqs_cav（来自 QEDPhonon, λ≠0）
本模块对腔模无显式依赖；这种设计让用户可以传入任何来源的频率
（例如实验测量的频率），保持灵活性。

单位约定
--------
  能量：Hartree (Ha)
  温度：Kelvin (K)
  频率输入：cm⁻¹（与 QEDPhonon 输出一致）
  返回：所有热力学量为 Hartree

参考：D. A. McQuarrie, Statistical Mechanics, 2000, Ch. 8
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Optional, Union


# ══════════════════════════════════════════════════════════════════════
# 物理常数（原子单位）
# ══════════════════════════════════════════════════════════════════════

K_B_AU   = 3.166811563e-6     # Boltzmann 常数 (Ha/K)
H_AU     = 2.0 * np.pi        # h = 2π (a.u., ℏ = 1)
HBAR_AU  = 1.0
AMU_TO_AU = 1822.88848        # 原子质量 → 电子质量
CM1_TO_AU = 1.0 / 219474.6306
AU_TO_CM1 = 219474.6306
KCAL_PER_HA = 627.5095
KJ_PER_HA   = 2625.5
EV_PER_HA   = 27.2114


# ══════════════════════════════════════════════════════════════════════
# 频率过滤工具
# ══════════════════════════════════════════════════════════════════════

def filter_vibrational_freqs(freqs_cm: np.ndarray,
                              min_freq_cm: float = 50.0,
                              is_ts: bool = False,
                              verbose: bool = False) -> dict:
    """
    将 QEDPhonon 输出的频率分类为：实振动模、虚频(反应坐标)、平动转动。

    Parameters
    ----------
    freqs_cm    : ndarray, 全部频率 (cm⁻¹)，虚频以负数存储
    min_freq_cm : 低于此值的实频视为平动/转动而过滤掉
    is_ts       : 若为 True（过渡态），分离出最大幅度的虚频作为反应坐标
    verbose     : 打印分类信息

    Returns
    -------
    dict:
        real_freqs   : 用于配分函数的实振动频率（cm⁻¹）
        imag_freq    : 反应坐标虚频的绝对值（cm⁻¹），仅 is_ts=True 时
        small_freqs  : 被过滤掉的小频率/虚频
        n_imag       : 总虚频数量（用于诊断）
    """
    freqs = np.asarray(freqs_cm, dtype=float)

    # 虚频（按 QEDPhonon 约定为负数）
    mask_imag = freqs < -min_freq_cm
    mask_real = freqs > min_freq_cm
    mask_small = ~(mask_imag | mask_real)

    imag_arr  = -freqs[mask_imag]   # 取绝对值
    real_arr  = freqs[mask_real]
    small_arr = freqs[mask_small]

    n_imag = int(mask_imag.sum())

    result = {
        'real_freqs':  real_arr,
        'small_freqs': small_arr,
        'n_imag':      n_imag,
    }

    if is_ts:
        if n_imag == 0:
            raise ValueError("is_ts=True 但未找到虚频，"
                             "可能不是过渡态或几何未优化到鞍点")
        if n_imag > 1 and verbose:
            print(f"  警告：发现 {n_imag} 个虚频，但TS只应有1个。"
                  f"取最大者作为反应坐标，其余视为数值噪声并丢弃。")
        # 取最大幅度的虚频作为反应坐标，其余丢弃
        idx_rxn = int(np.argmax(imag_arr))
        result['imag_freq'] = float(imag_arr[idx_rxn])
        # 其余虚频不计入配分函数（保守处理）
    else:
        if n_imag > 0 and verbose:
            print(f"  警告：极小点应无虚频，但发现 {n_imag} 个，"
                  f"已忽略。请检查几何收敛。")
        result['imag_freq'] = None

    if verbose:
        print(f"  频率分类：{len(real_arr)} 个实振动模, "
              f"{n_imag} 个虚频, {len(small_arr)} 个小频率(平动/转动)")

    return result


# ══════════════════════════════════════════════════════════════════════
# 振动配分函数（核心）
# ══════════════════════════════════════════════════════════════════════

def zero_point_energy(freqs_cm: np.ndarray) -> float:
    """
    谐振子零点能 ZPE = (1/2) Σ ℏω_i。

    Parameters
    ----------
    freqs_cm : ndarray, 实振动频率 (cm⁻¹)，已过滤虚频和小频率

    Returns
    -------
    zpe : Hartree
    """
    freqs_au = np.asarray(freqs_cm) * CM1_TO_AU
    return 0.5 * float(np.sum(freqs_au))


def vibrational_partition_function(freqs_cm: np.ndarray,
                                    T: float,
                                    convention: str = 'bot') -> float:
    """
    谐振子配分函数 Q_vib。

    Parameters
    ----------
    freqs_cm : 实振动频率 (cm⁻¹)
    T        : 温度 (K)
    convention :
        'bot' (bottom of well): Q = Π 1/(1 - exp(-ℏω/kT))
              能量原点在势能极小点（不含 ZPE）。需单独加 ZPE 才得正确 G。
        'v=0' (zero of vibration): Q = Π exp(-ℏω/2kT) / (1 - exp(-ℏω/kT))
              能量原点在 v=0 振动基态（含 ZPE）。

    Returns
    -------
    Q_vib : 无量纲（log Q 用于 G_vib = -kT ln Q）
    """
    if T <= 0:
        raise ValueError(f"T={T} 必须为正")

    freqs_au = np.asarray(freqs_cm) * CM1_TO_AU
    kT = K_B_AU * T
    x  = freqs_au / kT   # ℏω / kT

    # 数值稳定：x 很大时 exp(-x) ≈ 0
    log_q_each = -np.log1p(-np.exp(-x))   # log[1/(1-exp(-x))]
    log_q = float(np.sum(log_q_each))

    if convention == 'bot':
        return np.exp(log_q)
    elif convention == 'v=0':
        # 多一个 exp(-x/2) 因子
        log_q_v0 = log_q - 0.5 * float(np.sum(x))
        return np.exp(log_q_v0)
    else:
        raise ValueError(f"未知 convention: {convention}")


def vibrational_thermo(freqs_cm: np.ndarray, T: float) -> dict:
    """
    振动热力学量（McQuarrie 8-25 至 8-28）。

    使用 'bot' 约定：能量参考点为势能曲面极小，ZPE 单独返回。

    Returns
    -------
    dict:
        ZPE     : 零点能 (Ha)
        U_vib   : 内能（不含 ZPE）= Σ ℏω/(exp(ℏω/kT) - 1)  (Ha)
        S_vib   : 熵 (Ha/K)
        G_vib   : 自由能（含 ZPE）= ZPE + U_vib - T S_vib (Ha)
        Q_vib   : 配分函数（bot 约定，不含 ZPE）
        log_Q   : ln Q_vib
    """
    freqs_au = np.asarray(freqs_cm) * CM1_TO_AU
    kT = K_B_AU * T

    if len(freqs_au) == 0:
        return {'ZPE': 0.0, 'U_vib': 0.0, 'S_vib': 0.0,
                'G_vib': 0.0, 'Q_vib': 1.0, 'log_Q': 0.0}

    x = freqs_au / kT
    # 抑制溢出
    exp_neg_x = np.exp(-np.minimum(x, 700.0))

    # ZPE
    zpe = 0.5 * float(np.sum(freqs_au))

    # 内能（不含 ZPE）：U = Σ ℏω·exp(-x)/(1-exp(-x))
    U = float(np.sum(freqs_au * exp_neg_x / (1.0 - exp_neg_x + 1e-300)))

    # 熵：S/k = Σ [x·exp(-x)/(1-exp(-x)) - ln(1-exp(-x))]
    term1 = x * exp_neg_x / (1.0 - exp_neg_x + 1e-300)
    term2 = -np.log1p(-exp_neg_x)
    S = K_B_AU * float(np.sum(term1 + term2))

    # 配分函数（bot 约定）
    log_Q = float(np.sum(-np.log1p(-exp_neg_x)))
    Q     = np.exp(min(log_Q, 700.0))

    # 自由能：G = ZPE + U - TS
    G = zpe + U - T * S

    return {
        'ZPE':   zpe,
        'U_vib': U,
        'S_vib': S,
        'G_vib': G,
        'Q_vib': Q,
        'log_Q': log_Q,
    }


# ══════════════════════════════════════════════════════════════════════
# 平动配分函数（理想气体）
# ══════════════════════════════════════════════════════════════════════

def translational_thermo(mass_amu: float, T: float,
                          pressure_atm: float = 1.0) -> dict:
    """
    理想气体平动配分函数（Sackur-Tetrode）。

    Parameters
    ----------
    mass_amu     : 总质量 (amu)
    T            : 温度 (K)
    pressure_atm : 压强 (atm)，默认 1 atm

    Returns
    -------
    dict: U_trans, S_trans, G_trans (Ha)
    
    备注：对吸附在表面的物种，平动自由度被冻结，应跳过本函数。
    """
    M = mass_amu * AMU_TO_AU
    kT = K_B_AU * T

    # 体积：理想气体 V = NkT/p，单分子 V = kT/p
    # 1 atm = 3.39882737e-9 Ha/bohr³
    p_au = pressure_atm * 3.39882737e-9
    V = kT / p_au   # bohr³

    # 热de Broglie波长：Λ = h / sqrt(2π M kT)
    # 在原子单位下 h = 2π
    Lambda_3 = (2.0 * np.pi / np.sqrt(2.0 * np.pi * M * kT)) ** 3

    # Q_trans = V / Λ³
    Q = V / Lambda_3
    log_Q = float(np.log(Q))

    # U_trans = (3/2)kT
    U = 1.5 * kT
    # S_trans = k[ln(Q/N) + 5/2]，N=1 单分子
    S = K_B_AU * (log_Q + 2.5)
    G = U - T * S

    return {
        'U_trans': U,
        'S_trans': S,
        'G_trans': G,
        'Q_trans': Q,
        'log_Q':   log_Q,
    }


# ══════════════════════════════════════════════════════════════════════
# 转动配分函数（刚性转子）
# ══════════════════════════════════════════════════════════════════════

def _moments_of_inertia(coords_bohr: np.ndarray,
                         masses_amu: np.ndarray) -> np.ndarray:
    """
    计算主惯性矩（amu·bohr²）。

    Returns
    -------
    eigvals : ndarray, shape (3,), 主轴惯性矩，升序
    """
    coords = np.asarray(coords_bohr)
    masses = np.asarray(masses_amu)

    # 质心
    M_tot = float(np.sum(masses))
    com = (masses[:, None] * coords).sum(axis=0) / M_tot
    coords = coords - com

    # 惯性张量
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    Ixx = float(np.sum(masses * (y**2 + z**2)))
    Iyy = float(np.sum(masses * (x**2 + z**2)))
    Izz = float(np.sum(masses * (x**2 + y**2)))
    Ixy = -float(np.sum(masses * x * y))
    Ixz = -float(np.sum(masses * x * z))
    Iyz = -float(np.sum(masses * y * z))

    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    eigvals = np.linalg.eigvalsh(I)
    return eigvals  # amu·bohr²


def rotational_thermo(coords_bohr: np.ndarray,
                       masses_amu: np.ndarray,
                       symmetry_number: int = 1,
                       T: float = 298.15,
                       linear: Optional[bool] = None) -> dict:
    """
    刚性转子转动热力学量。

    Parameters
    ----------
    coords_bohr : ndarray (natm, 3), 笛卡尔坐标 (Bohr)
    masses_amu  : ndarray (natm,), 原子质量 (amu)
    symmetry_number : 转动对称数 σ
        线形：1 (异核 CO, CO₂?) 或 2 (同核 H₂, N₂, CO₂)
        非线形：1 (CHFClBr) ... 12 (CH₄)
    T          : 温度 (K)
    linear     : 是否为线形分子。None 时自动判定（最小惯性矩 < 1e-3）。

    Returns
    -------
    dict: U_rot, S_rot, G_rot (Ha), 含线性/非线性的判别
    """
    coords = np.asarray(coords_bohr)
    masses = np.asarray(masses_amu)
    natm = len(coords)

    if natm < 2:
        # 单原子：无转动自由度
        return {'U_rot': 0.0, 'S_rot': 0.0, 'G_rot': 0.0,
                'Q_rot': 1.0, 'linear': False, 'natm': 1}

    I_principal = _moments_of_inertia(coords, masses)

    # 自动判定线性
    if linear is None:
        linear = bool(I_principal[0] < 1e-3 * max(I_principal[1], 1e-10))

    kT = K_B_AU * T
    # 转动温度 Θ_rot = ℏ²/(2 I k_B)，I 单位 amu·bohr² → 转 a.u. 质量
    I_au = I_principal * AMU_TO_AU   # 电子质量·bohr²

    if linear:
        # 取 I_yy = I_zz（线形分子）
        I_lin = I_au[2] if I_au[2] > 1e-3 else I_au[1]
        Theta_rot = 1.0 / (2.0 * I_lin)   # ℏ=1 in a.u.
        # Q_rot = T / (σ Θ_rot)（高温近似）
        Q = T / (symmetry_number * Theta_rot / K_B_AU)
        # 等价于 Q = 2 I kT / (σ ℏ²) = 2 I kT / σ
        # 内能：U_rot = kT
        U = kT
        # 熵：S_rot = k(1 + ln Q)
        S = K_B_AU * (1.0 + np.log(Q))
    else:
        # 非线形：Q = sqrt(π) / σ · sqrt(T³ / (Θ_A Θ_B Θ_C))
        Theta = 1.0 / (2.0 * I_au) / K_B_AU   # K
        Q = (np.sqrt(np.pi) / symmetry_number
             * np.sqrt(T**3 / float(np.prod(Theta))))
        # 内能：U_rot = (3/2)kT
        U = 1.5 * kT
        # 熵：S_rot = k(3/2 + ln Q)
        S = K_B_AU * (1.5 + np.log(Q))

    G = U - T * S

    return {
        'U_rot': U,
        'S_rot': S,
        'G_rot': G,
        'Q_rot': Q,
        'log_Q': float(np.log(Q)),
        'linear': linear,
        'I_principal_au': I_au,
    }


# ══════════════════════════════════════════════════════════════════════
# 总热力学量（一站式接口）
# ══════════════════════════════════════════════════════════════════════

def total_thermo(e_elec: float,
                 freqs_cm: np.ndarray,
                 T: float,
                 mass_amu: Optional[float] = None,
                 coords_bohr: Optional[np.ndarray] = None,
                 masses_amu_array: Optional[np.ndarray] = None,
                 symmetry_number: int = 1,
                 pressure_atm: float = 1.0,
                 phase: str = 'adsorbed',
                 is_ts: bool = False,
                 min_freq_cm: float = 50.0,
                 verbose: bool = False) -> dict:
    """
    给定电子能量和振动频率，计算完整的 Gibbs 自由能。

    Parameters
    ----------
    e_elec    : 电子能量 (Ha)，来自 mf.e_tot
    freqs_cm  : 全部振动频率 (cm⁻¹)，QEDPhonon 输出
    T         : 温度 (K)
    phase :
        'gas'      : 包含平动 + 转动 + 振动
        'adsorbed' : 仅振动（吸附物在表面，平动转动冻结）
        'cluster'  : 同 'adsorbed'，催化中心也作整体看待
    mass_amu       : phase='gas' 时必需
    coords_bohr, masses_amu_array : phase='gas' 时必需（用于转动）
    is_ts     : 是否为过渡态（虚频排除在配分函数外）
    min_freq_cm : 低于此值的频率视为非振动模

    Returns
    -------
    dict: 全部热力学量 + 反应坐标信息
    """
    # 1. 频率分类
    classified = filter_vibrational_freqs(
        freqs_cm, min_freq_cm=min_freq_cm,
        is_ts=is_ts, verbose=verbose
    )
    real_freqs = classified['real_freqs']

    # 2. 振动贡献
    vib = vibrational_thermo(real_freqs, T)

    # 3. 平动 + 转动（仅气相）
    if phase == 'gas':
        if mass_amu is None or coords_bohr is None or masses_amu_array is None:
            raise ValueError("phase='gas' 时需提供 mass_amu, coords_bohr, masses_amu_array")
        trans = translational_thermo(mass_amu, T, pressure_atm)
        rot   = rotational_thermo(coords_bohr, masses_amu_array,
                                   symmetry_number, T)
        U_extra = trans['U_trans'] + rot['U_rot']
        S_extra = trans['S_trans'] + rot['S_rot']
        G_extra = trans['G_trans'] + rot['G_rot']
    elif phase in ('adsorbed', 'cluster'):
        trans = rot = None
        U_extra = S_extra = G_extra = 0.0
    else:
        raise ValueError(f"未知 phase={phase}, 应为 'gas'/'adsorbed'/'cluster'")

    # 4. 总 Gibbs 自由能
    # G = E_elec + ZPE + U_vib + U_trans+rot - T(S_vib + S_trans+rot) + pV
    # 对吸附物 pV 项忽略；对气相 pV = kT 已包含在 G_trans 中
    G_total = e_elec + vib['G_vib'] + G_extra

    return {
        'phase':        phase,
        'T':            T,
        'is_ts':        is_ts,
        'E_elec':       e_elec,
        'ZPE':          vib['ZPE'],
        'U_vib':        vib['U_vib'],
        'S_vib':        vib['S_vib'],
        'G_vib':        vib['G_vib'],
        'Q_vib':        vib['Q_vib'],
        'G_trans_rot':  G_extra,
        'S_extra':      S_extra,
        'U_extra':      U_extra,
        'G_total':      G_total,
        'H_total':      e_elec + vib['ZPE'] + vib['U_vib'] + U_extra,
        'S_total':      vib['S_vib'] + S_extra,
        'imag_freq_cm': classified.get('imag_freq'),
        'n_real_modes': len(real_freqs),
        'trans': trans,
        'rot':   rot,
    }


# ══════════════════════════════════════════════════════════════════════
# 辅助：单位转换打印
# ══════════════════════════════════════════════════════════════════════

def format_thermo(thermo: dict, units: str = 'kcal/mol') -> str:
    """将 total_thermo 输出格式化为可读字符串。"""
    if units == 'kcal/mol':
        f = KCAL_PER_HA
        u = 'kcal/mol'
    elif units == 'kJ/mol':
        f = KJ_PER_HA
        u = 'kJ/mol'
    elif units == 'eV':
        f = EV_PER_HA
        u = 'eV'
    elif units == 'meV':
        f = EV_PER_HA * 1000
        u = 'meV'
    else:
        f = 1.0; u = 'Ha'

    lines = [f"  ─ Thermodynamics @ T = {thermo['T']:.2f} K ─",
             f"    phase           : {thermo['phase']}",
             f"    is_ts           : {thermo['is_ts']}",
             f"    E_elec          : {thermo['E_elec']*f:>12.4f} {u}",
             f"    ZPE             : {thermo['ZPE']*f:>+12.4f} {u}",
             f"    U_vib           : {thermo['U_vib']*f:>+12.4f} {u}",
             f"    -T·S_vib        : {-thermo['T']*thermo['S_vib']*f:>+12.4f} {u}",
             f"    G_trans+rot     : {thermo['G_trans_rot']*f:>+12.4f} {u}",
             f"    G_total         : {thermo['G_total']*f:>12.4f} {u}",
             f"    n_real_modes    : {thermo['n_real_modes']}"]
    if thermo['imag_freq_cm'] is not None:
        lines.append(f"    ν‡ (虚频)       : {thermo['imag_freq_cm']:.1f} cm⁻¹")
    return "\n".join(lines)
