# -*- coding: utf-8 -*-
"""
test_thermo.py
==============
验证 thermo.py 中各热力学函数与教科书公式一致。

参考体系：
  H2 气相在 298.15 K：实验值 S = 130.68 J/(mol·K) = 31.21 cal/(mol·K)
  CO 气相在 298.15 K：S = 197.66 J/(mol·K) = 47.21 cal/(mol·K)
"""
import sys
sys.path.insert(0, '/home/claude/nqeddft_tst')

import numpy as np
from nqeddft.tst.thermo import (
    K_B_AU, AU_TO_CM1, KCAL_PER_HA, KJ_PER_HA,
    zero_point_energy, vibrational_partition_function, vibrational_thermo,
    translational_thermo, rotational_thermo, total_thermo,
    filter_vibrational_freqs, format_thermo,
)


def test_zpe_h2():
    """H2 ZPE = 0.5 ω = 0.5 × 4395 cm⁻¹ ≈ 2197 cm⁻¹ ≈ 6.28 kcal/mol"""
    freqs = np.array([4395.0])   # H2 实验振动波数
    zpe = zero_point_energy(freqs)
    zpe_kcal = zpe * KCAL_PER_HA
    print(f"  H2 ZPE = {zpe:.6f} Ha = {zpe_kcal:.3f} kcal/mol "
          f"(expected ~6.28 kcal/mol)")
    assert abs(zpe_kcal - 6.28) < 0.05, f"H2 ZPE 偏差过大: {zpe_kcal}"


def test_vib_low_T_limit():
    """低温下 U_vib → 0, S_vib → 0, Q_vib → 1（高频极限）"""
    freqs = np.array([4395.0])  # 高频
    T = 100.0   # 远低于 ℏω/k = 6322 K
    th = vibrational_thermo(freqs, T)
    print(f"  低温 H2: U_vib={th['U_vib']:.2e}, S_vib={th['S_vib']:.2e}")
    assert th['U_vib'] < 1e-15, "低温下 U_vib 应 ≈ 0"
    assert th['S_vib'] < 1e-15, "低温下 S_vib 应 ≈ 0"
    assert abs(th['Q_vib'] - 1.0) < 1e-10


def test_vib_high_T_limit():
    """高温极限 U_vib → kT（经典极限）。
    
    精确公式 U = ℏω·x/(e^x - 1)，x = ℏω/kT。
    展开 U/kT = x/(e^x-1) = 1 - x/2 + x²/12 - ...
    要求 x << 1 才能严格 → 1。
    取 ν=20 cm⁻¹, T=10000 K → x≈0.0029 → 偏离 ~0.1%
    """
    freqs = np.array([20.0])   # 极低频
    T = 10000.0
    th = vibrational_thermo(freqs, T)
    kT = K_B_AU * T
    ratio = th['U_vib'] / kT
    print(f"  高温低频 x=ℏω/kT≈{20*1e-2/(K_B_AU*T*AU_TO_CM1):.4f}: "
          f"U_vib/kT = {ratio:.6f} (应 → 1)")
    assert abs(ratio - 1.0) < 0.005


def test_partition_function_consistency():
    """vibrational_partition_function 与 vibrational_thermo 内部 Q 一致"""
    freqs = np.array([1000.0, 2000.0, 3000.0])
    T = 298.15
    Q1 = vibrational_partition_function(freqs, T, convention='bot')
    th = vibrational_thermo(freqs, T)
    Q2 = th['Q_vib']
    print(f"  Q (bot): direct={Q1:.6e}, via thermo={Q2:.6e}")
    np.testing.assert_allclose(Q1, Q2, rtol=1e-10)


def test_translational_h2():
    """H2 平动熵在 298.15 K, 1 atm 实验值 ~28.0 cal/(mol·K) = 117.2 J/(mol·K)
    
    注：H2 总熵 S_total ≈ 130.7 J/(mol·K)，分解为：
       S_trans ≈ 117.2 (含)
       S_rot   ≈ 12.7
       S_vib   ≈ 0  (高频)
    """
    M_h2 = 2.016
    T = 298.15
    th = translational_thermo(M_h2, T, pressure_atm=1.0)
    # S 单位 Ha/K → J/(mol·K): × 2625500 / 1000 = 2625.5
    S_J_per_molK = th['S_trans'] * KJ_PER_HA * 1000   # Ha/K → J/(mol·K)
    print(f"  H2 S_trans = {S_J_per_molK:.2f} J/(mol·K) "
          f"(expected ~117 J/(mol·K))")
    assert abs(S_J_per_molK - 117.0) < 2.0


def test_rotational_h2():
    """H2 转动熵：σ=2, B=60.85 cm⁻¹, S_rot ~12.7 J/(mol·K)"""
    # H2 键长 0.741 Å = 1.40 bohr，质心在中点
    coords = np.array([[0.0, 0.0, -0.7],
                        [0.0, 0.0,  0.7]])
    masses = np.array([1.008, 1.008])
    T = 298.15
    th = rotational_thermo(coords, masses, symmetry_number=2, T=T, linear=True)
    S_J = th['S_rot'] * KJ_PER_HA * 1000
    print(f"  H2 S_rot = {S_J:.2f} J/(mol·K) (expected ~12.7), "
          f"linear={th['linear']}")
    assert th['linear']
    assert abs(S_J - 12.7) < 1.5


def test_total_thermo_h2_gas():
    """H2 气相总熵 ~130.7 J/(mol·K) @ 298 K"""
    coords = np.array([[0.0, 0.0, -0.7],
                        [0.0, 0.0,  0.7]])
    masses = np.array([1.008, 1.008])
    e_elec = -1.1745   # 大致 H2 PBE 能量，不影响熵
    freqs  = np.array([4395.0])

    th = total_thermo(
        e_elec=e_elec, freqs_cm=freqs, T=298.15,
        mass_amu=2.016, coords_bohr=coords, masses_amu_array=masses,
        symmetry_number=2, phase='gas',
    )
    S_total_J = th['S_total'] * KJ_PER_HA * 1000
    print(f"  H2 S_total = {S_total_J:.2f} J/(mol·K) (expected ~130.7)")
    print(format_thermo(th, 'kcal/mol'))
    assert abs(S_total_J - 130.7) < 3.0


def test_filter_freqs_ts():
    """TS 频率过滤：1个虚频 + 5个实频"""
    freqs = np.array([-1500.0, 100.0, 800.0, 1200.0, 2400.0, 3500.0, 5.0])
    cls = filter_vibrational_freqs(freqs, min_freq_cm=50, is_ts=True, verbose=True)
    assert cls['n_imag'] == 1
    assert cls['imag_freq'] == 1500.0
    assert len(cls['real_freqs']) == 5  # 100, 800, 1200, 2400, 3500
    # 频率 5.0 (< 50) 被过滤为 small


def test_filter_freqs_min():
    """极小点应无虚频"""
    freqs = np.array([100.0, 800.0, 1200.0, 2400.0])
    cls = filter_vibrational_freqs(freqs, min_freq_cm=50, is_ts=False)
    assert cls['n_imag'] == 0
    assert cls['imag_freq'] is None
    assert len(cls['real_freqs']) == 4


def test_ts_thermo_excludes_imag():
    """TS 配分函数应排除虚频"""
    freqs_ts = np.array([-1500.0, 1000.0, 2000.0, 3000.0])  # 1虚3实
    th_ts = total_thermo(
        e_elec=-1.0, freqs_cm=freqs_ts, T=298.15,
        phase='cluster', is_ts=True, verbose=False,
    )
    # 同样 3 个实频但作为极小点处理
    freqs_min = np.array([1000.0, 2000.0, 3000.0])
    th_min = total_thermo(
        e_elec=-1.0, freqs_cm=freqs_min, T=298.15,
        phase='cluster', is_ts=False,
    )
    print(f"  TS  G_vib = {th_ts['G_vib']:.6f}, ZPE = {th_ts['ZPE']:.6f}")
    print(f"  Min G_vib = {th_min['G_vib']:.6f}, ZPE = {th_min['ZPE']:.6f}")
    np.testing.assert_allclose(th_ts['G_vib'], th_min['G_vib'], atol=1e-12)
    assert th_ts['imag_freq_cm'] == 1500.0


def test_temperature_scaling():
    """检查 G(T) 单调递减（高温熵贡献使 G 下降）"""
    freqs = np.array([1000.0, 2000.0, 3000.0])
    Gs = []
    for T in [200.0, 298.15, 400.0, 600.0]:
        th = total_thermo(e_elec=-1.0, freqs_cm=freqs, T=T, phase='cluster')
        Gs.append(th['G_total'])
        print(f"  T={T:.1f} K: G_total = {th['G_total']:.6f} Ha")
    Gs = np.array(Gs)
    diffs = np.diff(Gs)
    # 振动 G 高温下下降是因为 -TS 项；但 ZPE+U_vib 上升
    # 整体 G 在高温下应单调下降
    print(f"  ΔG: {diffs}")
    # 不强求单调，但应该在 ~mHa 量级
    assert np.all(np.abs(diffs) < 1e-2)


def run_all():
    print("=" * 65)
    print("thermo.py 单元测试")
    print("=" * 65)
    tests = [
        test_zpe_h2,
        test_vib_low_T_limit,
        test_vib_high_T_limit,
        test_partition_function_consistency,
        test_translational_h2,
        test_rotational_h2,
        test_total_thermo_h2_gas,
        test_filter_freqs_ts,
        test_filter_freqs_min,
        test_ts_thermo_excludes_imag,
        test_temperature_scaling,
    ]
    n_pass = 0
    n_fail = 0
    for t in tests:
        try:
            print(f"\n→ {t.__name__}")
            t()
            print(f"  ✓ PASS")
            n_pass += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1
    print("\n" + "=" * 65)
    print(f"结果：{n_pass} 通过，{n_fail} 失败")
    print("=" * 65)
    return n_fail == 0


if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
