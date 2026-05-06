# -*- coding: utf-8 -*-
"""
nqeddft.tst.eyring_plot
=======================
Eyring 与 Arrhenius 图的生成与对比工具。

主要功能
--------
1. 单 QEDTST 的 Eyring/Arrhenius 图 (matplotlib)
2. 多 QEDTST 比较图（自由空间 vs 不同腔参数）
3. CSV 导出（用于其他绘图工具）
4. 文本摘要表（不依赖 matplotlib）
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Optional, Dict

from .thermo import K_B_AU, KCAL_PER_HA
from .qed_tst import QEDTST


# ══════════════════════════════════════════════════════════════════════
# 不依赖 matplotlib 的输出
# ══════════════════════════════════════════════════════════════════════

def eyring_table(scan: dict, units: str = 'kcal/mol') -> str:
    """
    生成温度扫描数据的文本表格。

    Returns
    -------
    str : 多行字符串，可直接 print
    """
    if units == 'kcal/mol':
        f = KCAL_PER_HA; u = 'kcal/mol'
    elif units == 'eV':
        f = 27.2114; u = 'eV'
    else:
        f = 1.0; u = 'Ha'

    lines = ["=" * 95]
    lines.append(f"{'T (K)':>8}  {'1000/T':>8}  "
                 f"{'ΔE‡ (' + u + ')':>14}  {'ΔG‡ (' + u + ')':>14}  "
                 f"{'ν‡ (cm⁻¹)':>11}  {'κ_tun':>8}  "
                 f"{'k (s⁻¹)':>12}")
    lines.append("-" * 95)
    for i, T in enumerate(scan['T']):
        lines.append(
            f"{T:>8.1f}  {1000/T:>8.4f}  "
            f"{scan['dE_elec'][i]*f:>14.4f}  "
            f"{scan['dG'][i]*f:>14.4f}  "
            f"{scan['imag_freq'][i]:>11.1f}  "
            f"{scan['kappa_tunnel'][i]:>8.3f}  "
            f"{scan['k_total'][i]:>12.3e}"
        )
    lines.append("-" * 95)
    arr = scan['arrhenius']
    lines.append(f"  Arrhenius 拟合: ln k = ln A - Ea/RT")
    lines.append(f"     E_a  = {arr['Ea_kcal']:.4f} kcal/mol "
                 f"= {arr['Ea_eV']:.4f} eV")
    lines.append(f"     A    = {arr['A_pre']:.4e} s⁻¹")
    lines.append(f"     R²   = {arr['R2']:.6f}")
    lines.append("=" * 95)
    return "\n".join(lines)


def save_scan_csv(scan: dict, filename: str):
    """将温度扫描结果保存为 CSV"""
    header = ("T_K,inv_T_per_K,1000_per_T,"
              "dE_elec_kcal,dG_kcal,dH_kcal,TdS_kcal,"
              "imag_freq_cm,kappa_tunnel,k_TST_per_s,k_total_per_s,"
              "ln_k_per_s,ln_k_over_T")

    T = scan['T']
    data = np.column_stack([
        T,
        1.0 / T,
        1000.0 / T,
        scan['dE_elec'] * KCAL_PER_HA,
        scan['dG']      * KCAL_PER_HA,
        scan['dH']      * KCAL_PER_HA,
        T * scan['dS']  * KCAL_PER_HA,
        scan['imag_freq'],
        scan['kappa_tunnel'],
        scan['k_TST'],
        scan['k_total'],
        scan['ln_k'],
        scan['ln_k_T'],
    ])
    np.savetxt(filename, data, delimiter=',', header=header, comments='')
    print(f"扫描数据已保存至 {filename}")


# ══════════════════════════════════════════════════════════════════════
# matplotlib 绘图（按需）
# ══════════════════════════════════════════════════════════════════════

def plot_eyring(scans: Dict[str, dict], filename: Optional[str] = None,
                title: str = "Eyring plot"):
    """
    Eyring 图：ln(k/T) vs 1/T。

    Parameters
    ----------
    scans    : dict {label: scan_dict}, 各 scan 由 temperature_scan 给出
    filename : 保存路径（None 则只显示）
    title    : 图标题

    斜率 = -ΔH‡/R, 截距 = ln(k_B/h) + ΔS‡/R
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plot_eyring 需要 matplotlib")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, scan in scans.items():
        ax.plot(1000.0 / scan['T'], scan['ln_k_T'], 'o-', label=label, lw=1.5)

    ax.set_xlabel(r'1000/T  (K$^{-1}$)')
    ax.set_ylabel(r'$\ln(k/T)$')
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Eyring 图已保存至 {filename}")
    return fig


def plot_arrhenius(scans: Dict[str, dict], filename: Optional[str] = None,
                   title: str = "Arrhenius plot"):
    """
    Arrhenius 图：ln(k) vs 1/T。
    斜率 = -Ea/R
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plot_arrhenius 需要 matplotlib")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, scan in scans.items():
        Ea = scan['arrhenius']['Ea_kcal']
        ax.plot(1000.0 / scan['T'], scan['ln_k'], 'o-',
                label=f"{label}  (Ea={Ea:.2f} kcal/mol)", lw=1.5)
    ax.set_xlabel(r'1000/T  (K$^{-1}$)')
    ax.set_ylabel(r'$\ln(k\,/\,\mathrm{s}^{-1})$')
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Arrhenius 图已保存至 {filename}")
    return fig


def plot_speedup_vs_T(speedup_dict: dict, filename: Optional[str] = None,
                      title: str = "Cavity-induced speedup"):
    """
    腔诱导速率比 vs T 图。

    speedup_dict : QEDTST.speedup() 的输出
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plot_speedup_vs_T 需要 matplotlib")

    T = speedup_dict['T']
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(T, speedup_dict['ratio'],       'o-', label='total ratio', lw=2)
    ax.plot(T, speedup_dict['ratio_TST'],   's--', label='TST only',   lw=1.5)
    ax.plot(T, speedup_dict['ratio_kappa'], '^:',  label='tunneling only', lw=1.5)
    ax.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$k_{\rm cav}/k_{\rm free}$')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"加速比图已保存至 {filename}")
    return fig


# ══════════════════════════════════════════════════════════════════════
# 综合分析
# ══════════════════════════════════════════════════════════════════════

def comprehensive_analysis(tst_free: QEDTST, tst_cav: QEDTST,
                           T_array: Sequence[float],
                           tunneling: str = 'auto',
                           output_prefix: Optional[str] = None) -> dict:
    """
    一站式综合分析：并排比较 free 和 cavity 体系。

    Parameters
    ----------
    tst_free, tst_cav : QEDTST 对象
    T_array           : 温度数组
    tunneling         : 隧穿方法
    output_prefix     : 文件名前缀（保存 CSV/图），None 则不保存

    Returns
    -------
    dict: 包含 scan_free, scan_cav, speedup, summary_text
    """
    scan_free = tst_free.temperature_scan(T_array, tunneling=tunneling)
    scan_cav  = tst_cav.temperature_scan(T_array, tunneling=tunneling)
    speedup   = QEDTST.speedup(tst_cav, tst_free, T_array, tunneling)

    # 摘要文本
    lines = ["=" * 80]
    lines.append("  QED-TST 综合分析：自由空间 vs 腔中")
    lines.append("=" * 80)
    lines.append("【自由空间】")
    lines.append(eyring_table(scan_free))
    lines.append("\n【腔中】")
    lines.append(eyring_table(scan_cav))
    lines.append("\n【速率比 k_cav/k_free】")
    lines.append(f"{'T (K)':>8}  {'ratio':>10}  {'ratio_TST':>10}  "
                 f"{'ratio_κ':>10}  {'ΔΔG‡ (kcal/mol)':>18}")
    lines.append("-" * 65)
    for i, T in enumerate(speedup['T']):
        lines.append(
            f"{T:>8.1f}  {speedup['ratio'][i]:>10.4f}  "
            f"{speedup['ratio_TST'][i]:>10.4f}  "
            f"{speedup['ratio_kappa'][i]:>10.4f}  "
            f"{speedup['ddG_au'][i]*KCAL_PER_HA:>+18.4f}"
        )
    lines.append("=" * 80)
    summary = "\n".join(lines)

    # 保存文件
    if output_prefix:
        save_scan_csv(scan_free, output_prefix + "_free.csv")
        save_scan_csv(scan_cav,  output_prefix + "_cav.csv")
        with open(output_prefix + "_summary.txt", 'w') as f:
            f.write(summary)
        # 尝试出图
        try:
            plot_arrhenius({'free': scan_free, 'cavity': scan_cav},
                           filename=output_prefix + "_arrhenius.png")
            plot_speedup_vs_T(speedup,
                              filename=output_prefix + "_speedup.png")
        except ImportError:
            print("[info] matplotlib 不可用，跳过出图")

    return {
        'scan_free':   scan_free,
        'scan_cav':    scan_cav,
        'speedup':     speedup,
        'summary':     summary,
    }
