# -*- coding: utf-8 -*-
"""
nqeddft.tst.qed_tst
===================
QED 增强的过渡态理论：将光子腔耦合纳入 Eyring 方程。

物理框架
--------
对于反应 R → TS‡ → P，标准 TST 速率：

    k_TST(T) = (k_B·T / h) · exp(-ΔG‡(T) / k_B T)

含隧穿修正的 generalized TST：

    k(T) = κ(T) · k_TST(T)

QED 影响通过两条路径：
    (A) 势能面修饰（电子层面）：ΔE‡_elec(λ,ω,ε)
    (B) 振动配分函数修饰：腔诱导频移 → ZPE, U_vib, S_vib 全部改变

对一个反应步骤，需要三个 QED-DFT 计算：
    R   ：反应物极小（自由空间或腔中）
    TS‡ ：过渡态（自由空间或腔中）
    P   ：产物极小（仅 Eckart 隧穿需要 V_r 时使用）

对每个计算需要振动分析（QEDPhonon）。

接口设计
--------
QEDTST 接受三种输入模式：
    1. mf_R, mf_TS, mf_P  (PySCF mf 对象，自动调用 QEDPhonon)
    2. 显式传入 e_elec 和 freqs_cm     (跳过 PySCF 步骤，便于断点续算)
    3. 从 JSON 缓存加载（用于 stage1_2 / stage3 已有结果）

主要输出：
    eyring_plot:     ln(k/T) vs 1/T
    arrhenius_plot:  ln(k) vs 1/T
    speedup:         k(腔) / k(自由空间)
    rate_law_table:  ΔE‡, ΔH‡, ΔS‡, ΔG‡, κ_tunnel, k(T)
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Union
from dataclasses import dataclass, field

from .thermo import (
    K_B_AU, AU_TO_CM1, CM1_TO_AU,
    KCAL_PER_HA, KJ_PER_HA, EV_PER_HA,
    total_thermo, format_thermo, filter_vibrational_freqs,
)
from .tunneling import tunneling_kappa, wigner_correction, eckart_correction


# 普朗克常数 h = 2π in atomic units (ℏ=1)
H_AU = 2.0 * np.pi


# ══════════════════════════════════════════════════════════════════════
# 单点状态容器
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StationaryPoint:
    """
    势能面驻点（极小或鞍点）的描述。

    Attributes
    ----------
    name      : 标识符，例如 'CO2*' 或 'TS_CO2_to_COOH'
    e_elec    : 电子能量 (Ha)
    freqs_cm  : 全部振动频率 (cm⁻¹)，QEDPhonon 输出（虚频以负数表示）
    is_ts     : 是否为过渡态
    coords_bohr : 几何 (Bohr)，phase='gas' 时计算转动需要
    masses_amu  : 原子质量 (amu)，phase='gas' 时需要
    symmetry_number : 转动对称数（默认1）
    phase     : 'adsorbed' / 'gas' / 'cluster'
    cavity    : 可选 Cavity 对象，仅用于记录元信息
    """
    name:    str
    e_elec:  float
    freqs_cm: np.ndarray
    is_ts:   bool = False
    coords_bohr: Optional[np.ndarray] = None
    masses_amu: Optional[np.ndarray] = None
    symmetry_number: int = 1
    phase:   str = 'adsorbed'
    cavity:  object = None     # nqeddft.Cavity 或 None

    @classmethod
    def from_mf(cls, mf, name: str, is_ts: bool = False,
                phase: str = 'adsorbed', symmetry_number: int = 1,
                hessian_stepsize: float = 0.001,
                verbose: bool = True) -> 'StationaryPoint':
        """
        从 QEDRKS/QEDUKS mf 对象自动调用 QEDPhonon 进行振动分析。

        要求 mf 已收敛（mf.kernel() 已执行）。
        """
        try:
            from nqeddft.phonon import QEDPhonon
        except ImportError:
            raise ImportError("from_mf 要求安装 nqeddft 软件包")

        if not hasattr(mf, 'e_tot'):
            raise ValueError(f"mf 似乎未收敛 (no e_tot)")

        if verbose:
            print(f"  对 [{name}] 进行振动分析...")
        ph = QEDPhonon(mf)
        hess = ph.numerical_hessian_fast(stepsize=hessian_stepsize,
                                         verbose=False)
        freqs, _ = ph.harmonic_analysis(hess)

        mol = mf.mol
        coords = mol.atom_coords()
        masses = mol.atom_mass_list(isotope_avg=True)

        cavity = getattr(mf, 'cavity', None)

        return cls(
            name=name,
            e_elec=mf.e_tot,
            freqs_cm=np.asarray(freqs),
            is_ts=is_ts,
            coords_bohr=coords,
            masses_amu=masses,
            symmetry_number=symmetry_number,
            phase=phase,
            cavity=cavity,
        )

    def thermo(self, T: float, min_freq_cm: float = 50.0,
               pressure_atm: float = 1.0,
               verbose: bool = False) -> dict:
        """
        在温度 T 下计算热力学量。返回 total_thermo 输出。
        """
        return total_thermo(
            e_elec=self.e_elec,
            freqs_cm=self.freqs_cm,
            T=T,
            mass_amu=(float(np.sum(self.masses_amu))
                      if self.masses_amu is not None else None),
            coords_bohr=self.coords_bohr,
            masses_amu_array=self.masses_amu,
            symmetry_number=self.symmetry_number,
            pressure_atm=pressure_atm,
            phase=self.phase,
            is_ts=self.is_ts,
            min_freq_cm=min_freq_cm,
            verbose=verbose,
        )

    def imag_freq_cm(self, min_freq_cm: float = 50.0) -> Optional[float]:
        """返回反应坐标虚频幅度 (cm⁻¹)，非 TS 返回 None"""
        cls = filter_vibrational_freqs(
            self.freqs_cm, min_freq_cm=min_freq_cm,
            is_ts=self.is_ts, verbose=False
        )
        return cls.get('imag_freq')


# ══════════════════════════════════════════════════════════════════════
# QEDTST 主类
# ══════════════════════════════════════════════════════════════════════

class QEDTST:
    """
    QED 增强的过渡态理论计算。

    Parameters
    ----------
    reactant : StationaryPoint，反应物
    ts       : StationaryPoint，过渡态（is_ts=True）
    product  : StationaryPoint，产物（用于 Eckart 隧穿，可选）

    Examples
    --------
    >>> R  = StationaryPoint.from_mf(mf_R,  'CO2*', is_ts=False)
    >>> TS = StationaryPoint.from_mf(mf_TS, 'TS',   is_ts=True)
    >>> P  = StationaryPoint.from_mf(mf_P,  'COOH*',is_ts=False)
    >>> tst = QEDTST(R, TS, P)
    >>> result = tst.compute_rate(T=300.0, tunneling='eckart')
    >>> tst.print_summary()
    """

    def __init__(self, reactant: StationaryPoint,
                 ts: StationaryPoint,
                 product: Optional[StationaryPoint] = None):
        if not ts.is_ts:
            raise ValueError(f"ts ({ts.name}) 的 is_ts=False；"
                             "请确认这是过渡态")
        if reactant.is_ts:
            raise ValueError(f"reactant ({reactant.name}) 不能为 TS")
        if product is not None and product.is_ts:
            raise ValueError(f"product ({product.name}) 不能为 TS")

        # 检查虚频存在
        nu_imag = ts.imag_freq_cm()
        if nu_imag is None:
            raise ValueError(f"TS [{ts.name}] 未识别到虚频。"
                             "请检查 TS 是否真为鞍点。")

        self.R  = reactant
        self.TS = ts
        self.P  = product
        self._cache = {}   # T → 结果字典

    # ------------------------------------------------------------------
    # 核心：单温度速率计算
    # ------------------------------------------------------------------

    def compute_rate(self, T: float,
                     tunneling: str = 'auto',
                     min_freq_cm: float = 50.0,
                     pressure_atm: float = 1.0,
                     verbose: bool = False) -> dict:
        """
        计算 T 温度下的反应速率常数。

        Parameters
        ----------
        T          : 温度 (K)
        tunneling  : 'none', 'wigner', 'eckart', 'auto'
                     'auto' 在 product 提供时用 eckart，否则 wigner
        min_freq_cm: 频率过滤阈值
        pressure_atm : 气相压强（默认 1 atm）
        verbose    : 打印中间量

        Returns
        -------
        dict:
            T            : 温度
            E_elec_R, E_elec_TS, E_elec_P : 电子能量
            dE_barrier_au : ΔE‡_elec (Ha)
            dG_barrier_au : ΔG‡ (Ha)
            dH_barrier_au : ΔH‡ (Ha)
            dS_barrier_au : ΔS‡ (Ha/K)
            kappa_tunnel  : 隧穿修正因子
            k_TST         : 经典 TST 速率 (s⁻¹)
            k_total       : 含隧穿的总速率 (s⁻¹)
            tunneling_method : 实际使用的隧穿方法
            R_thermo, TS_thermo, P_thermo : 各点热力学量
        """
        kT = K_B_AU * T

        # 1. 热力学量
        th_R  = self.R.thermo(T, min_freq_cm, pressure_atm, verbose=False)
        th_TS = self.TS.thermo(T, min_freq_cm, pressure_atm, verbose=False)
        th_P  = (self.P.thermo(T, min_freq_cm, pressure_atm, verbose=False)
                 if self.P is not None else None)

        # 2. 自由能差
        dE_elec = th_TS['E_elec'] - th_R['E_elec']
        dG = th_TS['G_total']     - th_R['G_total']
        dH = th_TS['H_total']     - th_R['H_total']
        dS = th_TS['S_total']     - th_R['S_total']

        # 反向势垒（Eckart 用）
        dE_elec_rev = None
        if self.P is not None:
            # V_r = E(TS) - E(P)
            dE_elec_rev = th_TS['E_elec'] - th_P['E_elec']

        # 3. 隧穿修正
        nu_imag = self.TS.imag_freq_cm(min_freq_cm)
        kappa = 1.0
        method_used = 'none'
        if tunneling == 'none':
            method_used = 'none'
        elif tunneling == 'auto':
            if dE_elec_rev is not None and dE_elec_rev > 0:
                tun = tunneling_kappa(nu_imag, T,
                                       V_f=dE_elec, V_r=dE_elec_rev,
                                       method='eckart')
                kappa = tun['kappa']
                method_used = 'eckart'
            else:
                tun = tunneling_kappa(nu_imag, T, method='wigner')
                kappa = tun['kappa']
                method_used = 'wigner'
        elif tunneling == 'wigner':
            tun = tunneling_kappa(nu_imag, T, method='wigner')
            kappa = tun['kappa']
            method_used = 'wigner'
        elif tunneling == 'eckart':
            if self.P is None or dE_elec_rev is None or dE_elec_rev <= 0:
                # 无产物或反向势垒非正 → 退化为 wigner
                tun = tunneling_kappa(nu_imag, T, method='wigner')
                kappa = tun['kappa']
                method_used = 'wigner_fallback'
            else:
                tun = tunneling_kappa(nu_imag, T,
                                       V_f=dE_elec, V_r=dE_elec_rev,
                                       method='eckart')
                kappa = tun['kappa']
                method_used = 'eckart'
        else:
            raise ValueError(f"未知 tunneling={tunneling}")

        # 4. Eyring TST 速率
        # k_TST = (kT/h) · exp(-ΔG‡/kT)
        # 在 a.u. 中 h = 2π
        # 单位转换：1 a.u. of (1/time) = 1/τ_au = 4.1341e16 s⁻¹
        TIME_AU_TO_S = 2.4188843e-17
        rate_au = (kT / H_AU) * np.exp(-dG / kT)
        k_TST_per_s = rate_au / TIME_AU_TO_S    # → s⁻¹
        k_total_per_s = kappa * k_TST_per_s

        result = {
            'T':                T,
            'E_elec_R':         th_R['E_elec'],
            'E_elec_TS':        th_TS['E_elec'],
            'E_elec_P':         th_P['E_elec'] if th_P else None,
            'dE_elec_au':       dE_elec,
            'dE_elec_rev_au':   dE_elec_rev,
            'dG_au':            dG,
            'dH_au':            dH,
            'dS_au':            dS,
            'imag_freq_cm':     nu_imag,
            'kappa_tunnel':     kappa,
            'k_TST':            k_TST_per_s,
            'k_total':          k_total_per_s,
            'tunneling_method': method_used,
            'R_thermo':         th_R,
            'TS_thermo':        th_TS,
            'P_thermo':         th_P,
        }
        self._cache[T] = result
        return result

    # ------------------------------------------------------------------
    # 温度扫描
    # ------------------------------------------------------------------

    def temperature_scan(self, T_array: Sequence[float],
                         tunneling: str = 'auto',
                         **kwargs) -> dict:
        """
        在多个温度下计算速率，输出 Eyring/Arrhenius 数据。

        Parameters
        ----------
        T_array  : 温度列表 (K)
        tunneling: 同 compute_rate

        Returns
        -------
        dict:
            T            : np.array
            k_TST        : np.array (s⁻¹)
            k_total      : np.array (s⁻¹)
            kappa_tunnel : np.array
            dG, dH, dS, dE_elec : np.array (Ha 或 Ha/K)
            ln_k_T       : ln(k/T)，用于 Eyring 图
            inv_T        : 1/T
            arrhenius : dict {Ea_au, A_pre, R²} 拟合 ln k = ln A - Ea/(R T)
        """
        T_array = np.asarray(T_array, dtype=float)
        n = len(T_array)
        out = {key: np.zeros(n) for key in
               ['k_TST', 'k_total', 'kappa_tunnel',
                'dG', 'dH', 'dS', 'dE_elec', 'imag_freq']}

        for i, T in enumerate(T_array):
            r = self.compute_rate(T, tunneling=tunneling, **kwargs)
            out['k_TST'][i]        = r['k_TST']
            out['k_total'][i]      = r['k_total']
            out['kappa_tunnel'][i] = r['kappa_tunnel']
            out['dG'][i]           = r['dG_au']
            out['dH'][i]           = r['dH_au']
            out['dS'][i]           = r['dS_au']
            out['dE_elec'][i]      = r['dE_elec_au']
            out['imag_freq'][i]    = r['imag_freq_cm']

        # 派生量
        out['T']      = T_array
        out['inv_T']  = 1.0 / T_array
        out['ln_k']   = np.log(np.maximum(out['k_total'], 1e-300))
        out['ln_k_T'] = out['ln_k'] - np.log(T_array)

        # Arrhenius 拟合 ln k = ln A - Ea/(RT)
        # 在 a.u.: ln k = ln A - Ea/kT  → 斜率 = -Ea/k
        slope, intercept = np.polyfit(out['inv_T'], out['ln_k'], 1)
        Ea_au = -slope * K_B_AU
        ln_A  = intercept
        A_pre = np.exp(ln_A)

        # R² 拟合质量
        y_pred = slope * out['inv_T'] + intercept
        ss_res = float(np.sum((out['ln_k'] - y_pred)**2))
        ss_tot = float(np.sum((out['ln_k'] - out['ln_k'].mean())**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        out['arrhenius'] = {
            'Ea_au':     Ea_au,
            'Ea_kcal':   Ea_au * KCAL_PER_HA,
            'Ea_eV':     Ea_au * EV_PER_HA,
            'A_pre':     A_pre,
            'ln_A':      ln_A,
            'R2':        r2,
        }
        return out

    # ------------------------------------------------------------------
    # 速率比（腔 vs 自由空间）
    # ------------------------------------------------------------------

    @staticmethod
    def speedup(tst_cav: 'QEDTST', tst_free: 'QEDTST',
                T_array: Sequence[float],
                tunneling: str = 'auto') -> dict:
        """
        比较两个 QEDTST 对象（腔中 vs 自由空间）的速率比。

        Returns
        -------
        dict:
            T          : 温度数组
            ratio      : k_cav / k_free
            ratio_TST  : k_TST_cav / k_TST_free  （不含隧穿）
            ratio_kappa: κ_cav / κ_free
            ddG_au     : ΔG‡_cav - ΔG‡_free（自由能差对速率的影响）
        """
        scan_cav  = tst_cav.temperature_scan(T_array, tunneling=tunneling)
        scan_free = tst_free.temperature_scan(T_array, tunneling=tunneling)
        return {
            'T':           scan_cav['T'],
            'ratio':       scan_cav['k_total'] / scan_free['k_total'],
            'ratio_TST':   scan_cav['k_TST']   / scan_free['k_TST'],
            'ratio_kappa': scan_cav['kappa_tunnel'] / scan_free['kappa_tunnel'],
            'ddG_au':      scan_cav['dG'] - scan_free['dG'],
            'ddE_elec_au': scan_cav['dE_elec'] - scan_free['dE_elec'],
        }

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def print_summary(self, T: float = 298.15, tunneling: str = 'auto',
                      units: str = 'kcal/mol'):
        """打印温度 T 下的速率分析摘要"""
        r = self._cache.get(T) or self.compute_rate(T, tunneling=tunneling)

        if units == 'kcal/mol':
            f = KCAL_PER_HA; u = 'kcal/mol'
        elif units == 'kJ/mol':
            f = KJ_PER_HA;   u = 'kJ/mol'
        elif units == 'eV':
            f = EV_PER_HA;   u = 'eV'
        elif units == 'meV':
            f = EV_PER_HA*1000; u = 'meV'
        else:
            f = 1.0; u = 'Ha'

        print("=" * 65)
        print(f"  QED-TST 反应速率分析  @ T = {T:.2f} K")
        print("=" * 65)
        print(f"  反应物 :  {self.R.name}")
        print(f"  过渡态 :  {self.TS.name}")
        if self.P is not None:
            print(f"  产物   :  {self.P.name}")
        if self.TS.cavity is not None:
            print(f"  腔     :  {self.TS.cavity.summary().splitlines()[0]}")
        print("-" * 65)
        print(f"  ΔE‡_elec        : {r['dE_elec_au']*f:>+10.3f} {u}")
        if r['dE_elec_rev_au'] is not None:
            print(f"  ΔE‡_elec (反向) : {r['dE_elec_rev_au']*f:>+10.3f} {u}  (Eckart V_r)")
        print(f"  ΔH‡             : {r['dH_au']*f:>+10.3f} {u}")
        print(f"  T·ΔS‡           : {T*r['dS_au']*f:>+10.3f} {u}")
        print(f"  ΔG‡             : {r['dG_au']*f:>+10.3f} {u}")
        print(f"  ν‡ (虚频)        : {r['imag_freq_cm']:>10.1f} cm⁻¹")
        print(f"  κ_tunnel         : {r['kappa_tunnel']:>10.4f}  ({r['tunneling_method']})")
        print("-" * 65)
        print(f"  k_TST    = {r['k_TST']:.4e}  s⁻¹")
        print(f"  k_total  = {r['k_total']:.4e}  s⁻¹  (含隧穿)")
        print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
# 便利函数
# ══════════════════════════════════════════════════════════════════════

def make_arrhenius_plot_data(scan: dict) -> dict:
    """
    从 temperature_scan 输出生成 Arrhenius/Eyring 图所需的数据。

    Returns
    -------
    dict: 用于 matplotlib 绘图的 x/y 数组
    """
    return {
        'inv_T_per_K':    scan['inv_T'],
        '1000_inv_T':     1000.0 / scan['T'],
        'ln_k':           scan['ln_k'],
        'ln_k_over_T':    scan['ln_k_T'],
        'k':              scan['k_total'],
        'k_TST':          scan['k_TST'],
        'kappa':          scan['kappa_tunnel'],
        'dG_kcal':        scan['dG'] * KCAL_PER_HA,
        'dE_elec_kcal':   scan['dE_elec'] * KCAL_PER_HA,
        'arrhenius_fit':  scan['arrhenius'],
    }
