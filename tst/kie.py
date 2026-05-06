# -*- coding: utf-8 -*-
"""
nqeddft.tst.kie
===============
动力学同位素效应 (Kinetic Isotope Effect, KIE)。

物理意义
--------
KIE = k_light / k_heavy （通常 k_H / k_D）

同位素替换不改变电子势能面，只改变核质量 → 振动频率 ω = √(k/m)。
对 H/D 替换 ω_D ≈ ω_H / √2，导致：
  - ZPE 下降（D 比 H 低 ~半）
  - 配分函数改变
  - **反应坐标虚频改变** → 隧穿因子改变

主要 KIE 类型：
  Primary   : 同位素直接参与断键   (KIE_H/D ~ 6-8 经典上限)
  Secondary : 同位素邻近反应中心    (KIE_H/D ~ 1.0-1.2)
  Heavy-atom: ¹²C/¹³C 等           (KIE ~ 1.01-1.05)

腔依赖
------
腔通过修改虚频 ν‡ 进而改变 KIE。在 VSC 体系中，腔与某个振动模共振时
KIE 可能出现"突变"（Climent et al., Acc. Chem. Res. 2022）。

接口
----
kie_h_d(tst_h, tst_d, T) → 单温度 KIE 值
kie_temperature_scan(tst_h, tst_d, Ts) → 温度依赖 KIE

替换工具
--------
make_isotopologue(sp, atom_idx, new_mass) → 新 StationaryPoint
  *注意*：同位素替换不能简单"换质量"——还要重做振动分析，
  因为质量加权 Hessian 改变 → 频率改变。本模块提供
  `rescale_freqs_by_mass` 作为近似（适用于已知"待替换原子"
  在哪些模式中占主要贡献的简单情况）。
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Optional

from .thermo import K_B_AU, KCAL_PER_HA
from .qed_tst import QEDTST, StationaryPoint


# ══════════════════════════════════════════════════════════════════════
# KIE 计算
# ══════════════════════════════════════════════════════════════════════

def kie_at_T(tst_light: QEDTST, tst_heavy: QEDTST, T: float,
             tunneling: str = 'auto') -> dict:
    """
    单温度 KIE = k_light / k_heavy。

    Parameters
    ----------
    tst_light : 轻同位素体系（如 H 标记）
    tst_heavy : 重同位素体系（如 D 标记）
    T         : 温度 (K)
    tunneling : 隧穿方法

    Returns
    -------
    dict:
        T               : 温度
        KIE             : k_L / k_H 总同位素效应
        KIE_TST         : 不含隧穿的 TST 同位素效应
        KIE_tunnel      : 隧穿同位素效应 = κ_L / κ_H
        kie_zpe         : ZPE 同位素效应（半经典近似）
                          = exp[-(ZPE_TS_L - ZPE_TS_H - ZPE_R_L + ZPE_R_H) / kT]
        rate_light, rate_heavy : 各自速率 (s⁻¹)
        details_light, details_heavy : compute_rate 完整输出
    """
    r_l = tst_light.compute_rate(T, tunneling=tunneling)
    r_h = tst_heavy.compute_rate(T, tunneling=tunneling)

    KIE        = r_l['k_total'] / r_h['k_total']
    KIE_TST    = r_l['k_TST']   / r_h['k_TST']
    KIE_tunnel = r_l['kappa_tunnel'] / r_h['kappa_tunnel']

    kT = K_B_AU * T
    # ZPE 同位素效应（半定量）
    zpe_R_l, zpe_R_h = r_l['R_thermo']['ZPE'], r_h['R_thermo']['ZPE']
    zpe_TS_l, zpe_TS_h = r_l['TS_thermo']['ZPE'], r_h['TS_thermo']['ZPE']
    delta_zpe = (zpe_TS_l - zpe_TS_h) - (zpe_R_l - zpe_R_h)
    kie_zpe = np.exp(-delta_zpe / kT)

    return {
        'T':              T,
        'KIE':            KIE,
        'KIE_TST':        KIE_TST,
        'KIE_tunnel':     KIE_tunnel,
        'kie_zpe':        kie_zpe,
        'rate_light':     r_l['k_total'],
        'rate_heavy':     r_h['k_total'],
        'imag_freq_l':    r_l['imag_freq_cm'],
        'imag_freq_h':    r_h['imag_freq_cm'],
        'details_light':  r_l,
        'details_heavy':  r_h,
    }


def kie_temperature_scan(tst_light: QEDTST, tst_heavy: QEDTST,
                         T_array: Sequence[float],
                         tunneling: str = 'auto') -> dict:
    """温度依赖的 KIE 扫描"""
    Ts = np.asarray(T_array, dtype=float)
    n = len(Ts)
    KIE        = np.zeros(n)
    KIE_TST    = np.zeros(n)
    KIE_tunnel = np.zeros(n)
    KIE_zpe    = np.zeros(n)
    for i, T in enumerate(Ts):
        d = kie_at_T(tst_light, tst_heavy, T, tunneling)
        KIE[i]        = d['KIE']
        KIE_TST[i]    = d['KIE_TST']
        KIE_tunnel[i] = d['KIE_tunnel']
        KIE_zpe[i]    = d['kie_zpe']

    return {
        'T':          Ts,
        'inv_T':      1.0 / Ts,
        'KIE':        KIE,
        'KIE_TST':    KIE_TST,
        'KIE_tunnel': KIE_tunnel,
        'KIE_zpe':    KIE_zpe,
    }


# ══════════════════════════════════════════════════════════════════════
# 同位素替换工具
# ══════════════════════════════════════════════════════════════════════

# 标准同位素质量 (amu)
ISOTOPE_MASSES = {
    'H':   1.00784,
    'D':   2.01410,
    'T':   3.01605,
    'C12': 12.00000,
    'C13': 13.00336,
    'C14': 14.00324,
    'N14': 14.00307,
    'N15': 15.00011,
    'O16': 15.99491,
    'O17': 16.99913,
    'O18': 17.99916,
    'S32': 31.97207,
    'S34': 33.96787,
}


def rescale_freqs_by_mass(freqs_cm: np.ndarray,
                          mode_mass_ratios: Sequence[float],
                          mode_mass_ratios_imag: Optional[float] = None
                          ) -> np.ndarray:
    """
    按"局域模式质量比"近似缩放频率（一阶 Teller-Redlich 近似）：

      ω_new = ω_old · √(μ_old / μ_new)

    Parameters
    ----------
    freqs_cm : 原始频率数组（虚频以负数）
    mode_mass_ratios : (μ_old/μ_new) for 各实模式（与 freqs 一一对应，
                       但不含虚频；虚频单独处理）
    mode_mass_ratios_imag : 虚频的质量比

    Returns
    -------
    新频率数组（cm⁻¹）

    备注：这是粗略近似，仅适用于"局域同位素替换"且模式分得清楚时。
          严格 KIE 应重新做整套 QEDPhonon 振动分析。
    """
    freqs_cm = np.asarray(freqs_cm, dtype=float)
    new_freqs = np.empty_like(freqs_cm)

    real_idx = np.where(freqs_cm > 0)[0]
    imag_idx = np.where(freqs_cm < 0)[0]

    if len(real_idx) != len(mode_mass_ratios):
        raise ValueError(
            f"实模式数 ({len(real_idx)}) 与提供的质量比数 "
            f"({len(mode_mass_ratios)}) 不一致"
        )

    for k, idx in enumerate(real_idx):
        ratio = mode_mass_ratios[k]
        new_freqs[idx] = freqs_cm[idx] * np.sqrt(ratio)

    for idx in imag_idx:
        if mode_mass_ratios_imag is None:
            # 默认与实模式相同的质量比 = 没指定就保留（不推荐）
            new_freqs[idx] = freqs_cm[idx]
        else:
            new_freqs[idx] = freqs_cm[idx] * np.sqrt(mode_mass_ratios_imag)

    return new_freqs


def make_isotopologue_simple(sp: StationaryPoint,
                              freq_scale_factors: dict,
                              new_name: Optional[str] = None,
                              ) -> StationaryPoint:
    """
    通过频率缩放构造同位素 StationaryPoint（粗略近似）。

    Parameters
    ----------
    sp : 原始 StationaryPoint
    freq_scale_factors : dict, 形如 {'real': [r1,r2,...], 'imag': r_im}
                         各 r 为 √(μ_old/μ_new)；H→D 时 r ≈ 1/√2 ≈ 0.707
    new_name : 新名字

    Returns
    -------
    新 StationaryPoint，e_elec、coords、masses 不变
    （严格说应换 atom_mass_list，但这里只用于模型测试）
    """
    real_factors = freq_scale_factors.get('real', [])
    imag_factor  = freq_scale_factors.get('imag', None)

    freqs = np.asarray(sp.freqs_cm, dtype=float)
    new_freqs = freqs.copy()

    real_mask = freqs > 0
    imag_mask = freqs < 0
    real_indices = np.where(real_mask)[0]
    imag_indices = np.where(imag_mask)[0]

    if len(real_factors) > 0 and len(real_factors) != len(real_indices):
        raise ValueError(
            f"实模式数={len(real_indices)}，scale 因子数={len(real_factors)}，不匹配"
        )

    for k, idx in enumerate(real_indices):
        new_freqs[idx] = freqs[idx] * real_factors[k]
    if imag_factor is not None:
        for idx in imag_indices:
            new_freqs[idx] = freqs[idx] * imag_factor   # 保持虚频符号

    return StationaryPoint(
        name=new_name or (sp.name + '_iso'),
        e_elec=sp.e_elec,        # 同位素不改电子能量
        freqs_cm=new_freqs,
        is_ts=sp.is_ts,
        coords_bohr=sp.coords_bohr,
        masses_amu=sp.masses_amu, # 简化：质量不动（仅频率改）
        symmetry_number=sp.symmetry_number,
        phase=sp.phase,
        cavity=sp.cavity,
    )


# ══════════════════════════════════════════════════════════════════════
# 输出工具
# ══════════════════════════════════════════════════════════════════════

def print_kie_summary(kie_dict: dict, name_light: str = 'H',
                       name_heavy: str = 'D'):
    """打印 KIE 分析摘要"""
    print("=" * 60)
    print(f"  动力学同位素效应 KIE = k_{name_light} / k_{name_heavy}")
    print(f"  T = {kie_dict['T']:.2f} K")
    print("=" * 60)
    print(f"  KIE (total)        : {kie_dict['KIE']:>8.4f}")
    print(f"  KIE (TST only)     : {kie_dict['KIE_TST']:>8.4f}  (无隧穿)")
    print(f"  KIE (tunnel only)  : {kie_dict['KIE_tunnel']:>8.4f}  (隧穿贡献)")
    print(f"  KIE (ZPE estimate) : {kie_dict['kie_zpe']:>8.4f}  (半经典 ZPE)")
    print("-" * 60)
    print(f"  k_{name_light:1s}    = {kie_dict['rate_light']:.4e} s⁻¹")
    print(f"  k_{name_heavy:1s}    = {kie_dict['rate_heavy']:.4e} s⁻¹")
    print(f"  ν‡_{name_light:1s} = {kie_dict['imag_freq_l']:.1f} cm⁻¹")
    print(f"  ν‡_{name_heavy:1s} = {kie_dict['imag_freq_h']:.1f} cm⁻¹")
    print("=" * 60)
