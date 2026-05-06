# -*- coding: utf-8 -*-
"""
test_kie.py
===========
KIE (动力学同位素效应) 模块测试。

参考体系：H + H₂ → H₂ + H 对 D + D₂ → D₂ + D 的 KIE。
经典 TST 上限 (无隧穿) 约 6-7（仅 ZPE 贡献），
含隧穿后 KIE 在 T<300K 可达 10-20（经典隧穿强烈依赖质量）。
"""
import sys
sys.path.insert(0, '/home/claude/nqeddft_tst')

import numpy as np

from tst.qed_tst import StationaryPoint, QEDTST
from tst.kie import (
    kie_at_T, kie_temperature_scan,
    make_isotopologue_simple, rescale_freqs_by_mass,
    print_kie_summary, ISOTOPE_MASSES,
)


def make_HHH_system():
    """H + H₂ 模型"""
    R = StationaryPoint('R', -1.6700, np.array([4395.0]),
                         is_ts=False, phase='cluster')
    TS = StationaryPoint('TS', -1.6546,
                          np.array([-1511.0, 2058.0, 870.0, 870.0]),
                          is_ts=True, phase='cluster')
    P = StationaryPoint('P', -1.6700, np.array([4395.0]),
                         is_ts=False, phase='cluster')
    return R, TS, P


def make_DDD_system():
    """D + D₂ 模型：所有 H 频率 × √(1/2) ≈ 0.7071"""
    R, TS, P = make_HHH_system()
    R_d  = make_isotopologue_simple(R,  {'real': [1/np.sqrt(2)]}, 'R_D')
    TS_d = make_isotopologue_simple(TS, {
        'real': [1/np.sqrt(2)] * 3,
        'imag': 1/np.sqrt(2),
    }, 'TS_D')
    P_d  = make_isotopologue_simple(P,  {'real': [1/np.sqrt(2)]}, 'P_D')
    return R_d, TS_d, P_d


def test_isotopologue_creation():
    R, TS, P = make_HHH_system()
    R_d, TS_d, P_d = make_DDD_system()
    
    print(f"  H₂ stretch:  {R.freqs_cm[0]:.0f} cm⁻¹")
    print(f"  D₂ stretch:  {R_d.freqs_cm[0]:.0f} cm⁻¹  (理论 4395/√2 = 3108)")
    
    print(f"  H TS 虚频:   {TS.freqs_cm[0]:.0f} cm⁻¹")
    print(f"  D TS 虚频:   {TS_d.freqs_cm[0]:.0f} cm⁻¹  (理论 -1068)")
    
    np.testing.assert_allclose(R_d.freqs_cm[0], 4395/np.sqrt(2), atol=1.0)
    np.testing.assert_allclose(TS_d.freqs_cm[0], -1511/np.sqrt(2), atol=1.0)


def test_kie_zpe_only():
    """无隧穿 KIE：仅 ZPE 贡献"""
    R, TS, P = make_HHH_system()
    R_d, TS_d, P_d = make_DDD_system()
    
    tst_h = QEDTST(R, TS, P)
    tst_d = QEDTST(R_d, TS_d, P_d)
    
    kie = kie_at_T(tst_h, tst_d, T=300.0, tunneling='none')
    print(f"  T=300K, 无隧穿: KIE = {kie['KIE']:.3f}")
    print(f"    KIE_TST = {kie['KIE_TST']:.3f}")
    print(f"    KIE_zpe = {kie['kie_zpe']:.3f}")
    
    # 模型 H+H₂: R 只有 1 个振动模，TS 有 3 个实模
    # ΔZPE_R(H→D) = 4395·(1 - 1/√2)/2 = 644 cm⁻¹
    # ΔZPE_TS(H→D) = (2058+870+870)·(1 - 1/√2)/2 = 583 cm⁻¹
    # 净 ΔΔZPE = ΔZPE_TS - ΔZPE_R = -61 cm⁻¹
    # 由于 TS ZPE 减少较小（仅 583 < 644），实际 H 反应的 G‡ 差异较小
    # KIE_zpe = exp(-(-61) cm⁻¹ × 1.439 / 300 K) = exp(0.293) = 1.34
    # 加上配分函数比，KIE_TST 应在 1.3-2.0
    assert 1.0 < kie['KIE_TST'] < 5.0
    # 至少 KIE > 1（有方向性）
    assert kie['KIE'] > 1.0


def test_kie_with_tunneling():
    """含隧穿后 KIE 增大"""
    R, TS, P = make_HHH_system()
    R_d, TS_d, P_d = make_DDD_system()
    
    tst_h = QEDTST(R, TS, P)
    tst_d = QEDTST(R_d, TS_d, P_d)
    
    kie_no = kie_at_T(tst_h, tst_d, T=300.0, tunneling='none')
    kie_w  = kie_at_T(tst_h, tst_d, T=300.0, tunneling='wigner')
    kie_e  = kie_at_T(tst_h, tst_d, T=300.0, tunneling='eckart')
    
    print(f"  T=300K KIE:")
    print(f"    无隧穿:    {kie_no['KIE']:.3f}")
    print(f"    Wigner:    {kie_w['KIE']:.3f}  (κ_H/κ_D = {kie_w['KIE_tunnel']:.3f})")
    print(f"    Eckart:    {kie_e['KIE']:.3f}  (κ_H/κ_D = {kie_e['KIE_tunnel']:.3f})")
    
    # 隧穿放大 KIE
    assert kie_w['KIE'] > kie_no['KIE']
    assert kie_e['KIE'] > kie_w['KIE']
    # κ_H > κ_D（H 隧穿更容易）
    assert kie_w['KIE_tunnel'] > 1.0


def test_kie_temperature_dependence():
    """KIE 应随 T 升高而下降（经典极限趋于 1）"""
    R, TS, P = make_HHH_system()
    R_d, TS_d, P_d = make_DDD_system()
    
    tst_h = QEDTST(R, TS, P)
    tst_d = QEDTST(R_d, TS_d, P_d)
    
    Ts = np.array([200., 300., 500., 1000., 2000.])
    scan = kie_temperature_scan(tst_h, tst_d, Ts, tunneling='wigner')
    
    print(f"  KIE vs T (Wigner):")
    for i, T in enumerate(Ts):
        print(f"    T={T:.0f}K: KIE={scan['KIE'][i]:.3f}  "
              f"(zpe={scan['KIE_zpe'][i]:.3f}, tun={scan['KIE_tunnel'][i]:.3f})")
    
    # 单调下降
    assert np.all(np.diff(scan['KIE']) < 0)
    # 高温极限 KIE → 1（弱）
    assert scan['KIE'][-1] < scan['KIE'][0]


def test_secondary_KIE():
    """次级 KIE: 仅替换非反应中心原子 → KIE 接近 1
    
    构造：在 R 和 TS 中都有 2 个振动模——一个反应模（H 直接参与，
    1500 cm⁻¹ 部分），一个旁观者模（要被 D 替换）。
    把"旁观模"放在 1500 cm⁻¹ 处。R 的 4395 是反应中心 H₂ 的振动。
    TS 的反应中心模式不被同位素影响（虚频 + 2058 + 870×2）。
    
    替换方式：
      旁观者模 1500 cm⁻¹ → D化后 1500/√2 = 1061
      反应中心模式 (R 的 4395, TS 的所有) 不变
    
    这模拟"二级原子被 D 替换"——KIE 应近似 1。
    """
    R = StationaryPoint('R', -1.6700, np.array([4395.0, 1500.0]),
                         is_ts=False, phase='cluster')
    TS = StationaryPoint('TS', -1.6546,
                          np.array([-1511.0, 2058.0, 870.0, 870.0, 1500.0]),
                          is_ts=True, phase='cluster')
    P = StationaryPoint('P', -1.6700, np.array([4395.0, 1500.0]),
                         is_ts=False, phase='cluster')
    
    # 旁观者模（最后一个）被 D 替换：缩放 1/√2
    R_iso = make_isotopologue_simple(R, {'real': [1.0, 1/np.sqrt(2)]}, 'R_iso')
    TS_iso = make_isotopologue_simple(TS, {
        'real': [1.0, 1.0, 1.0, 1/np.sqrt(2)],   # 4个实模，最后一个变
        'imag': 1.0,                              # 虚频不变（反应中心）
    }, 'TS_iso')
    P_iso = make_isotopologue_simple(P, {'real': [1.0, 1/np.sqrt(2)]}, 'P_iso')
    
    tst_h = QEDTST(R, TS, P)
    tst_iso = QEDTST(R_iso, TS_iso, P_iso)
    
    kie = kie_at_T(tst_h, tst_iso, T=300.0, tunneling='wigner')
    print(f"  次级 KIE: {kie['KIE']:.4f} (期望接近 1)")
    print(f"    KIE_TST: {kie['KIE_TST']:.4f}, KIE_zpe: {kie['kie_zpe']:.4f}")
    # 次级 KIE 通常 0.8-1.3
    # 由于旁观者模在 R 和 TS 中等同贡献 → 净 ΔZPE 抵消
    assert 0.7 < kie['KIE'] < 1.5


def test_print_summary():
    R, TS, P = make_HHH_system()
    R_d, TS_d, P_d = make_DDD_system()
    tst_h = QEDTST(R, TS, P)
    tst_d = QEDTST(R_d, TS_d, P_d)
    kie = kie_at_T(tst_h, tst_d, T=298.15, tunneling='wigner')
    print_kie_summary(kie)


def run_all():
    print("=" * 65)
    print("kie.py 单元测试")
    print("=" * 65)
    tests = [
        test_isotopologue_creation,
        test_kie_zpe_only,
        test_kie_with_tunneling,
        test_kie_temperature_dependence,
        test_secondary_KIE,
        test_print_summary,
    ]
    n_pass = n_fail = 0
    for t in tests:
        try:
            print(f"\n→ {t.__name__}")
            t()
            print("  ✓ PASS")
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
