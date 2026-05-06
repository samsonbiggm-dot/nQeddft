# -*- coding: utf-8 -*-
"""
test_eyring_plot.py
===================
测试 Eyring/Arrhenius 输出和综合分析。
"""
import sys, os
sys.path.insert(0, '/home/claude/nqeddft_tst')

import numpy as np
from nqeddft.tst.qed_tst import StationaryPoint, QEDTST
from nqeddft.tst.eyring_plot import (
    eyring_table, save_scan_csv, comprehensive_analysis,
)

# 自由空间体系
R  = StationaryPoint('R',  -1.6700, np.array([4395.0]),
                      is_ts=False, phase='cluster')
TS_free = StationaryPoint('TS_free', -1.6546,
                           np.array([-1511.0, 2058.0, 870.0, 870.0]),
                           is_ts=True, phase='cluster')
P  = StationaryPoint('P',  -1.6700, np.array([4395.0]),
                      is_ts=False, phase='cluster')

# 腔中：势垒降低 0.5 kcal/mol，虚频降低 3%
TS_cav = StationaryPoint('TS_cav', -1.6546 - 0.000797,
                          np.array([-1465.0, 2055.0, 868.0, 868.0]),
                          is_ts=True, phase='cluster')


def test_eyring_table():
    tst = QEDTST(R, TS_free, P)
    scan = tst.temperature_scan(np.linspace(300, 600, 5), tunneling='wigner')
    text = eyring_table(scan)
    print(text)
    # 表格中应包含 Arrhenius 拟合结果
    assert "Arrhenius" in text
    assert "R²" in text


def test_save_csv():
    tst = QEDTST(R, TS_free, P)
    scan = tst.temperature_scan([300., 400., 500.], tunneling='wigner')
    fn = "/tmp/test_scan.csv"
    save_scan_csv(scan, fn)
    assert os.path.exists(fn)
    
    # 检查行数
    with open(fn) as f:
        lines = f.readlines()
    assert len(lines) == 4   # 1 header + 3 数据
    print(f"  CSV 内容头部:")
    print(f"    {lines[0].strip()}")
    print(f"    {lines[1].strip()}")
    os.remove(fn)


def test_comprehensive_analysis():
    """综合对比 free vs cavity"""
    tst_free = QEDTST(R, TS_free, P)
    tst_cav  = QEDTST(R, TS_cav,  P)
    
    Ts = np.linspace(300, 600, 4)
    result = comprehensive_analysis(tst_free, tst_cav, Ts, tunneling='wigner')
    
    print(result['summary'])
    
    # 物理性检查
    sp = result['speedup']
    # 腔降低势垒 → 速率比 > 1
    assert all(r > 1.0 for r in sp['ratio'])
    # 速率比应随 T 下降（高温熵主导，势垒效应弱）
    assert sp['ratio'][0] > sp['ratio'][-1]


def test_comprehensive_with_output():
    """带文件输出的综合分析"""
    tst_free = QEDTST(R, TS_free, P)
    tst_cav  = QEDTST(R, TS_cav,  P)
    
    prefix = "/tmp/test_comp"
    Ts = np.linspace(300, 500, 3)
    result = comprehensive_analysis(
        tst_free, tst_cav, Ts, tunneling='wigner',
        output_prefix=prefix,
    )
    
    expected_files = [prefix + "_free.csv",
                      prefix + "_cav.csv",
                      prefix + "_summary.txt"]
    for f in expected_files:
        assert os.path.exists(f), f"{f} 未生成"
        print(f"  ✓ {f} 已生成 ({os.path.getsize(f)} bytes)")
        os.remove(f)
    
    # PNG 文件可能因 matplotlib 不可用而缺失——可选
    for f in [prefix + "_arrhenius.png", prefix + "_speedup.png"]:
        if os.path.exists(f):
            print(f"  ✓ {f} 已生成（{os.path.getsize(f)} bytes）")
            os.remove(f)


def run_all():
    print("=" * 65)
    print("eyring_plot.py 单元测试")
    print("=" * 65)
    tests = [
        test_eyring_table,
        test_save_csv,
        test_comprehensive_analysis,
        test_comprehensive_with_output,
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
