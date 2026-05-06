# -*- coding: utf-8 -*-
"""
nqeddft.tst — QED 增强的过渡态理论模块

提供腔依赖的反应速率计算：
  - 振动配分函数 (thermo)
  - 隧穿修正 Wigner/Eckart (tunneling)
  - Eyring TST + 隧穿 (qed_tst)
  - 同位素效应 (kie)
  - Eyring/Arrhenius 出图 (eyring_plot)

主要接口
--------
    StationaryPoint  : 描述一个驻点（极小或鞍点）
    QEDTST           : 主类，整合三点 (R, TS, P) → k(T)
    kie_at_T         : 单温度 KIE
    comprehensive_analysis : 自由空间 vs 腔中并排对比

物理框架
--------
    k(T) = κ(T) · (k_B T / h) · exp(-ΔG‡(T) / k_B T)

    腔修饰路径：
      (A) 电子势能 → ΔE‡_elec(λ, ω, ε)
      (B) 振动频率 → ZPE, U_vib, S_vib → ΔG‡(T) 改变
      (C) 反应坐标虚频 → κ_tunnel(T) 改变

参考
----
    Eyring, J. Chem. Phys. 3, 107 (1935)
    Eckart, Phys. Rev. 35, 1303 (1930)
    Garrett & Truhlar, JPC 83, 1052 (1979)
    McQuarrie, Statistical Mechanics, 2000
"""

from .thermo import (
    K_B_AU, AU_TO_CM1, CM1_TO_AU,
    KCAL_PER_HA, KJ_PER_HA, EV_PER_HA,
    zero_point_energy,
    vibrational_partition_function,
    vibrational_thermo,
    translational_thermo,
    rotational_thermo,
    total_thermo,
    filter_vibrational_freqs,
    format_thermo,
)

from .tunneling import (
    wigner_correction,
    eckart_correction,
    tunneling_kappa,
)

from .qed_tst import (
    StationaryPoint,
    QEDTST,
    make_arrhenius_plot_data,
)

from .kie import (
    kie_at_T,
    kie_temperature_scan,
    make_isotopologue_simple,
    rescale_freqs_by_mass,
    print_kie_summary,
    ISOTOPE_MASSES,
)

from .eyring_plot import (
    eyring_table,
    save_scan_csv,
    plot_eyring,
    plot_arrhenius,
    plot_speedup_vs_T,
    comprehensive_analysis,
)

__version__ = '0.1.0'

__all__ = [
    'zero_point_energy', 'vibrational_partition_function', 'vibrational_thermo',
    'translational_thermo', 'rotational_thermo', 'total_thermo',
    'filter_vibrational_freqs', 'format_thermo',
    'wigner_correction', 'eckart_correction', 'tunneling_kappa',
    'StationaryPoint', 'QEDTST', 'make_arrhenius_plot_data',
    'kie_at_T', 'kie_temperature_scan',
    'make_isotopologue_simple', 'rescale_freqs_by_mass',
    'print_kie_summary', 'ISOTOPE_MASSES',
    'eyring_table', 'save_scan_csv',
    'plot_eyring', 'plot_arrhenius', 'plot_speedup_vs_T',
    'comprehensive_analysis',
]
