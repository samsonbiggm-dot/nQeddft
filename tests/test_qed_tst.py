# -*- coding: utf-8 -*-
"""
test_qed_tst.py
===============
QEDTST 主类单元测试。

使用人造数据（StationaryPoint 直接构造），不依赖 PySCF。
真实 QED-DFT 集成测试见 test_qed_tst_integration.py（Step 6/7）。
"""
import sys
sys.path.insert(0, '/home/claude/nqeddft_tst')

import numpy as np
from nqeddft.tst.thermo import K_B_AU, KCAL_PER_HA, CM1_TO_AU
from nqeddft.tst.qed_tst import StationaryPoint, QEDTST


# ── 人造体系：仿照 H + H₂ → H₂ + H ────────────────────────────────────
# E_R = -1.6700 Ha (H + H₂ 大致)
# E_TS = -1.6546 Ha (V_f = 9.7 kcal/mol = 0.01546 Ha)
# E_P = -1.6700 Ha (对称，V_r = V_f)
# 反应物 H+H₂: 振动模式 ω(H₂) ≈ 4395 cm⁻¹
# TS 线形 H₃，5 个振动模 + 1 虚频
#   ν1 = 2058 cm⁻¹（对称 stretch）
#   ν2,3 = 870 cm⁻¹ × 2（弯曲，简并）
#   ν‡ = 1511i cm⁻¹（反对称 stretch）
# 产物 H₂+H 同 R

E_R  = -1.6700
E_TS = -1.6546   # V_f = 9.7 kcal/mol
E_P  = -1.6700

# 注：用列表展开避免简并模异常；3 个原子 → 9个自由度
# H+H₂线形：3 平动 + 2 转动 + 1 振动(H₂的 stretch)，加上自由 H 的 0 内振动
# 简化为只跟踪振动模式（其余视为 small_freqs 被过滤）
freqs_R  = np.array([4395.0])         # H₂ 振动 + 隐式平动转动
freqs_TS = np.array([-1511.0, 2058.0, 870.0, 870.0])
freqs_P  = np.array([4395.0])

R  = StationaryPoint('R',  E_R,  freqs_R,  is_ts=False, phase='cluster')
TS = StationaryPoint('TS', E_TS, freqs_TS, is_ts=True,  phase='cluster')
P  = StationaryPoint('P',  E_P,  freqs_P,  is_ts=False, phase='cluster')


def test_init_validates_TS():
    """非 TS 不能传给 ts 参数"""
    try:
        QEDTST(R, R, P)   # 用 R 当 TS
        assert False, "应该抛 ValueError"
    except ValueError as e:
        print(f"  ✓ 正确拒绝非TS：{e}")


def test_init_requires_imag_freq():
    """无虚频的"TS"应被拒绝"""
    bad_TS = StationaryPoint('badTS', -1.65, np.array([2000.0]),
                              is_ts=True, phase='cluster')
    try:
        QEDTST(R, bad_TS, P)
        assert False
    except ValueError as e:
        print(f"  ✓ 正确拒绝无虚频TS：{e}")


def test_compute_rate_basic():
    """基本速率计算"""
    tst = QEDTST(R, TS, P)
    r = tst.compute_rate(T=298.15, tunneling='wigner')
    print(f"  T=298.15K:")
    print(f"    ΔE‡ = {r['dE_elec_au']*KCAL_PER_HA:.3f} kcal/mol")
    print(f"    ΔG‡ = {r['dG_au']*KCAL_PER_HA:.3f} kcal/mol")
    print(f"    κ_W = {r['kappa_tunnel']:.3f}")
    print(f"    k_TST = {r['k_TST']:.3e} s⁻¹")
    print(f"    k_total = {r['k_total']:.3e} s⁻¹")
    
    # 物理性检查
    assert 9.0 < r['dE_elec_au']*KCAL_PER_HA < 10.5
    assert r['kappa_tunnel'] > 1.0
    assert r['k_total'] > r['k_TST']
    # 温度因子 kT/h ≈ 6.2e12 s⁻¹ @ 298K
    # exp(-9.7/0.5926) ≈ 8e-8, 故 k_TST ≈ 5e5 s⁻¹
    assert 1e3 < r['k_TST'] < 1e8


def test_compute_rate_eckart():
    """Eckart 隧穿"""
    tst = QEDTST(R, TS, P)
    r = tst.compute_rate(T=300.0, tunneling='eckart')
    print(f"  T=300K Eckart:")
    print(f"    V_f = {r['dE_elec_au']*KCAL_PER_HA:.3f} kcal/mol")
    print(f"    V_r = {r['dE_elec_rev_au']*KCAL_PER_HA:.3f} kcal/mol")
    print(f"    κ_E = {r['kappa_tunnel']:.3f}")
    print(f"    k_total = {r['k_total']:.3e} s⁻¹")
    # 对称势 V_f=V_r ≈ 9.7 kcal/mol, ν=1511 cm⁻¹, T=300
    # κ_E 应 > κ_W
    r_w = tst.compute_rate(T=300.0, tunneling='wigner')
    assert r['kappa_tunnel'] > r_w['kappa_tunnel']


def test_temperature_scan():
    """多温度扫描 + Arrhenius 拟合"""
    tst = QEDTST(R, TS, P)
    Ts = np.linspace(300, 800, 11)
    scan = tst.temperature_scan(Ts, tunneling='wigner')

    print(f"  扫描 T = {Ts[0]:.0f}-{Ts[-1]:.0f} K:")
    print(f"    k_total range: {scan['k_total'][0]:.2e} - {scan['k_total'][-1]:.2e} s⁻¹")
    arr = scan['arrhenius']
    print(f"    Arrhenius E_a = {arr['Ea_kcal']:.3f} kcal/mol")
    print(f"    pre-exp A     = {arr['A_pre']:.3e} s⁻¹")
    print(f"    R²            = {arr['R2']:.6f}")

    # E_a 应接近 ΔE‡ + (1/2)RT 量级（含 ZPE 修正 + 微小温度依赖）
    assert 9.0 < arr['Ea_kcal'] < 12.0
    # R² 应接近 1（线性 Arrhenius 行为）
    assert arr['R2'] > 0.99
    # k 单调递增
    assert np.all(np.diff(scan['k_total']) > 0)


def test_speedup_no_cavity():
    """两个相同的 TST 对象速率比应 = 1"""
    tst1 = QEDTST(R, TS, P)
    tst2 = QEDTST(R, TS, P)
    Ts = [200., 300., 400., 500.]
    sp = QEDTST.speedup(tst1, tst2, Ts, tunneling='wigner')
    print(f"  相同体系: ratio = {sp['ratio']}")
    np.testing.assert_allclose(sp['ratio'], 1.0, rtol=1e-10)


def test_speedup_with_cavity_shift():
    """模拟腔诱导势垒降低 → 速率提升"""
    # 自由空间
    TS_free = StationaryPoint('TS_free', -1.6546,
                               np.array([-1511.0, 2058.0, 870.0, 870.0]),
                               is_ts=True, phase='cluster')
    tst_free = QEDTST(R, TS_free, P)

    # 腔中：势垒降低 1 kcal/mol = 0.001593 Ha，虚频也降低 5%
    # 势垒降低 = TS 能量下降（更接近反应物能量）
    E_TS_cav = -1.6546 - 0.001593   # ΔV‡ = -1 kcal/mol（更稳定）
    TS_cav = StationaryPoint('TS_cav', E_TS_cav,
                              np.array([-1435.0, 2050.0, 868.0, 868.0]),
                              is_ts=True, phase='cluster')
    tst_cav = QEDTST(R, TS_cav, P)

    Ts = [300., 400., 500.]
    sp = QEDTST.speedup(tst_cav, tst_free, Ts, tunneling='wigner')
    print(f"  腔降低势垒 1 kcal/mol:")
    for i, T in enumerate(Ts):
        print(f"    T={T:.0f}K: ratio = {sp['ratio'][i]:.3f}, "
              f"ΔΔG‡ = {sp['ddG_au'][i]*KCAL_PER_HA:.3f} kcal/mol")
    
    # 物理检查：势垒降低 → 速率提升
    assert all(r > 1.0 for r in sp['ratio'])
    # T=300K 时速率比 ≈ exp(1/(kT*627.5)) = exp(1/0.5926) ≈ 5.4
    # 但还有 ν‡变化引起 κ变化，故为大约 4-7
    assert 3.0 < sp['ratio'][0] < 10.0


def test_print_summary():
    """打印输出不报错"""
    tst = QEDTST(R, TS, P)
    tst.print_summary(T=298.15, tunneling='auto')


def test_no_product():
    """无 product 也能算（仅 Wigner 隧穿）"""
    tst = QEDTST(R, TS, product=None)
    r = tst.compute_rate(T=300.0, tunneling='auto')
    print(f"  无product: method={r['tunneling_method']}, κ={r['kappa_tunnel']:.3f}")
    assert r['tunneling_method'] == 'wigner'
    assert r['dE_elec_rev_au'] is None


def run_all():
    print("=" * 65)
    print("qed_tst.py 单元测试")
    print("=" * 65)
    tests = [
        test_init_validates_TS,
        test_init_requires_imag_freq,
        test_compute_rate_basic,
        test_compute_rate_eckart,
        test_temperature_scan,
        test_speedup_no_cavity,
        test_speedup_with_cavity_shift,
        test_print_summary,
        test_no_product,
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
