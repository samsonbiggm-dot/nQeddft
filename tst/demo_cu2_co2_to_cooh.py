# -*- coding: utf-8 -*-
"""
demo_cu2_co2_to_cooh.py
========================
QED-TST 模块综合演示：模拟 Cu₂ 团簇上 CO2* + H* → COOH* 反应在自由空间和
振动强耦合 (VSC) 腔中的速率对比。

本脚本不调用 PySCF——使用合成的"已知"驻点数据，目的是展示整个
QED-TST 工作流是如何串起来的。在实际研究中，您会把：

    1. mf_R, mf_TS, mf_P 替换为 stage1_2 / stage3 中的 QEDRKS 收敛对象
    2. 调用 StationaryPoint.from_mf(mf, ...) 自动做 Hessian 和振动分析
    3. 对每个 λ × ω_cav 网格点重复以上分析
    4. 用 comprehensive_analysis 输出 ΔΔG‡(λ, ω_cav, T) 总表

体系参数（合成）：
    Cu₂ 团簇上 CO2* + H* 反应：
        反应物 R  : Cu₂(CO2*)(H*)        E_R  = -3433.5000 Ha
        过渡态 TS : Cu₂(CO2-H‡)          E_TS = -3433.4750 Ha (ΔE‡ = 15.7 kcal/mol)
        产物   P  : Cu₂(COOH*)            E_P  = -3433.4900 Ha (放能 6.3 kcal/mol)
    
    虚频 ν‡ ≈ 1300i cm⁻¹（C-O-H 协同弯曲伸缩）
    
    腔参数：ω_cav = 1720 cm⁻¹（C-O 伸缩共振，您 config.py 中的 nu_target_cm）
            λ = 0.05（标称 VSC 强度）
    
    腔效应（"假设"模型）：
        ΔV‡ = -2.0 kcal/mol   （腔降低势垒）
        ν‡ → 1240 cm⁻¹           （虚频减弱 ~5%）
        TS 实模轻微频移          （+2-3% 平均）
"""
import sys
sys.path.insert(0, '/home/claude/nqeddft_tst')

import numpy as np
from tst import (
    StationaryPoint, QEDTST,
    kie_at_T, kie_temperature_scan,
    make_isotopologue_simple,
    comprehensive_analysis,
    print_kie_summary,
    KCAL_PER_HA,
)


def build_freespace_system():
    """构造自由空间下 Cu₂+CO2*+H* → TS → COOH* 的三个驻点"""
    # 反应物：Cu₂ + CO2* + H*（典型 Cu-O 伸缩 350，C-O 伸缩 1720, ...）
    freqs_R = np.array([
        # Cu-Cu 伸缩
        180.0,
        # Cu-O 伸缩 / Cu-C 伸缩
        320.0, 410.0,
        # CO2* 弯曲（吸附 CO2 通常变形并红移）
        650.0, 720.0,
        # CO2 反对称伸缩
        1280.0,
        # C=O 伸缩
        1660.0,  # 注意吸附后 1720→1660
        # Cu-H 伸缩
        1850.0,
        # 其他低频杂模
        100.0, 230.0, 540.0,
    ])

    # 过渡态：C-O-H 协同弯曲，1 个虚频 + 实模
    freqs_TS = np.array([
        -1300.0,    # 反应坐标虚频（C 转向 H）
        165.0, 280.0, 380.0,
        490.0, 580.0,
        720.0, 880.0,
        1240.0,     # CO2 弯曲（部分破坏）
        1530.0,     # COOH 部分形成的 C-O 伸缩
        2400.0,     # O-H 部分形成
    ])

    # 产物：COOH* 已成型
    freqs_P = np.array([
        180.0, 290.0, 400.0,
        510.0, 620.0,
        750.0, 850.0,
        1080.0, 1230.0,
        1420.0,    # COOH 中 C-O 单键
        1750.0,    # COOH 中 C=O 双键
        3580.0,    # O-H 伸缩（已形成）
    ])

    R  = StationaryPoint('Cu2(CO2*)(H*)',   -3433.5000, freqs_R,
                         is_ts=False, phase='cluster')
    TS = StationaryPoint('Cu2(CO2-H)‡',     -3433.4750, freqs_TS,
                         is_ts=True,  phase='cluster')
    P  = StationaryPoint('Cu2(COOH*)',      -3433.4900, freqs_P,
                         is_ts=False, phase='cluster')
    return R, TS, P


def build_cavity_system(R, TS_free, P, lambda_val=0.05, omega_cav_cm=1720):
    """
    构造腔中体系。在实际研究里，TS 的所有量都来自 QEDRKS+QEDPhonon。
    这里我们用"模型腔效应"近似:
        - C=O 伸缩共振 → 该模式下移 ~3%
        - 反应坐标虚频弱化 ~5%
        - 势能面整体降低 0.05·λ² 的程度
    """
    # TS 的频率响应：
    #   在 ω_cav = 1720 cm⁻¹ 附近的模式被红移（共振）
    #   反应坐标虚频独立处理
    freqs_TS_cav = TS_free.freqs_cm.copy()

    # 红移规则：|ν - ω_cav| < 200 cm⁻¹ 的模式按 1 - 0.03·λ/0.05 调整
    redshift_factor = 1.0 - 0.03 * (lambda_val / 0.05)
    for i, nu in enumerate(freqs_TS_cav):
        if nu > 0 and abs(nu - omega_cav_cm) < 250:
            freqs_TS_cav[i] = nu * redshift_factor

    # 反应坐标虚频弱化 5%（典型腔诱导隧穿减弱）
    for i, nu in enumerate(freqs_TS_cav):
        if nu < 0:
            freqs_TS_cav[i] = nu * (1.0 - 0.05 * (lambda_val / 0.05))

    # 势垒降低（Mandal-Reichman 极化子修正风格）：
    #   ΔV‡ ~ -k·λ²·μ_eff²  这里取 k 使 λ=0.05 时 ΔV=-2 kcal/mol
    delta_V = -2.0 * (lambda_val / 0.05)**2 / KCAL_PER_HA   # in Ha
    E_TS_cav = TS_free.e_elec + delta_V
    
    # 反应物和产物的势能可能也会受影响，但这里假设 TS 受影响最大
    # （这正是"势垒选择性降低"的物理图像）

    TS_cav = StationaryPoint(
        f'Cu2(CO2-H)‡@cav(λ={lambda_val})',
        E_TS_cav, freqs_TS_cav, is_ts=True, phase='cluster'
    )
    return TS_cav


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 80)
    print("█  QED-TST Demo: Cu₂ + CO2* → COOH* 在 VSC 腔中的速率分析")
    print("█" * 80)

    # 1. 构造体系
    print("\n[1] 构造自由空间体系...")
    R, TS_free, P = build_freespace_system()
    tst_free = QEDTST(R, TS_free, P)
    print(f"    R:   {R.name}    E = {R.e_elec:.4f} Ha, "
          f"{len(R.freqs_cm)} 个频率")
    print(f"    TS:  {TS_free.name}   E = {TS_free.e_elec:.4f} Ha, "
          f"虚频 ν‡ = {TS_free.imag_freq_cm():.0f} cm⁻¹")
    print(f"    P:   {P.name}     E = {P.e_elec:.4f} Ha, "
          f"{len(P.freqs_cm)} 个频率")
    print(f"    电子势垒 ΔE‡ = {(TS_free.e_elec - R.e_elec)*KCAL_PER_HA:.2f} kcal/mol")
    print(f"    放能       ΔE = {(P.e_elec - R.e_elec)*KCAL_PER_HA:.2f} kcal/mol")

    # 2. 构造腔中体系
    print("\n[2] 构造腔中体系 (ω_cav=1720 cm⁻¹, λ=0.05)...")
    TS_cav = build_cavity_system(R, TS_free, P, lambda_val=0.05, omega_cav_cm=1720)
    tst_cav = QEDTST(R, TS_cav, P)
    print(f"    TS_cav: ν‡_cav = {TS_cav.imag_freq_cm():.0f} cm⁻¹ "
          f"(自由 {TS_free.imag_freq_cm():.0f}, 减弱 "
          f"{(1 - TS_cav.imag_freq_cm()/TS_free.imag_freq_cm())*100:.1f}%)")
    print(f"    ΔV‡_cav = {(TS_cav.e_elec - R.e_elec)*KCAL_PER_HA:.2f} kcal/mol "
          f"(自由 {(TS_free.e_elec - R.e_elec)*KCAL_PER_HA:.2f}, ΔΔV = "
          f"{(TS_cav.e_elec - TS_free.e_elec)*KCAL_PER_HA:+.2f} kcal/mol)")

    # 3. 单温度速率分析
    print("\n[3] T = 298.15 K 速率对比 (Eckart 隧穿)...")
    print("\n    ━━━ 自由空间 ━━━")
    tst_free.print_summary(T=298.15, tunneling='eckart')
    print("\n    ━━━ 腔中 ━━━")
    tst_cav.print_summary(T=298.15, tunneling='eckart')

    # 4. 温度扫描和综合分析
    print("\n[4] 温度扫描 200-700 K, 综合分析...")
    Ts = np.linspace(200, 700, 11)
    result = comprehensive_analysis(
        tst_free, tst_cav, Ts, tunneling='eckart',
        output_prefix='/tmp/cu2_co2_demo'
    )

    # 5. KIE 分析（H/D 替换）
    print("\n[5] H/D 动力学同位素效应...")
    # 替换 R 中 Cu-H (1850 cm⁻¹) → Cu-D (1850/√2 ≈ 1308 cm⁻¹)
    # 替换 TS 中 O-H 部分形成模 (2400 cm⁻¹) → 1697 cm⁻¹
    # 替换 P 中 O-H (3580) → 2531 cm⁻¹
    # 简化：所有"含 H"的高频模都按 √2 缩放
    
    # R 的 1850 是 Cu-H，索引 = 7
    R_d_factors = [1.0]*len(R.freqs_cm)
    R_d_factors[7] = 1/np.sqrt(2)
    R_d = make_isotopologue_simple(R, {'real': R_d_factors}, 'R_D')
    
    # TS 的 2400 是 O-H 部分形成，索引 = 10（最后一个），
    # 此外虚频也是 H 转移坐标，受影响
    TS_d_real_factors = [1.0]*(len(TS_free.freqs_cm)-1)  # 实模数 = 总-虚
    # 找出哪些模需要 √2 缩放
    for i, idx_in_real in enumerate(np.where(TS_free.freqs_cm > 0)[0]):
        if TS_free.freqs_cm[idx_in_real] >= 2000:  # 高频 H 涉及
            TS_d_real_factors[i] = 1/np.sqrt(2)
    TS_d_imag_factor = 1/np.sqrt(2)  # 反应坐标 H 转移 → D 转移
    TS_d = make_isotopologue_simple(
        TS_free,
        {'real': TS_d_real_factors, 'imag': TS_d_imag_factor},
        'TS_D'
    )
    
    P_d_factors = [1.0]*len(P.freqs_cm)
    P_d_factors[-1] = 1/np.sqrt(2)  # O-H → O-D
    P_d = make_isotopologue_simple(P, {'real': P_d_factors}, 'P_D')
    
    tst_d_free = QEDTST(R_d, TS_d, P_d)
    
    kie = kie_at_T(tst_free, tst_d_free, T=298.15, tunneling='eckart')
    print_kie_summary(kie, name_light='H', name_heavy='D')
    
    # 同样的腔中 KIE
    TS_d_cav = make_isotopologue_simple(
        TS_cav,
        {'real': TS_d_real_factors, 'imag': TS_d_imag_factor},
        'TS_D@cav'
    )
    tst_d_cav = QEDTST(R_d, TS_d_cav, P_d)
    
    kie_cav = kie_at_T(tst_cav, tst_d_cav, T=298.15, tunneling='eckart')
    print(f"\n    腔中 KIE (T=298K) = {kie_cav['KIE']:.3f}")
    print(f"    自由 KIE (T=298K) = {kie['KIE']:.3f}")
    print(f"    KIE 比 (cav/free) = {kie_cav['KIE']/kie['KIE']:.3f}")
    print("    (这个比值是 VSC 实验里的关键诊断量)")

    print("\n" + "█" * 80)
    print("█  Demo 完成。生成的文件位于 /tmp/cu2_co2_demo_*")
    print("█" * 80)


if __name__ == '__main__':
    main()
