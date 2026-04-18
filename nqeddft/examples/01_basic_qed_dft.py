# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""示例 01：基本 QED-DFT 计算 — HF 分子在单模光学腔中的基态"""
from pyscf import gto
from nqeddft import Cavity, QEDRKS
from nqeddft.validation import lambda_limit_check, gauge_invariance_check

mol = gto.M(atom="H 0 0 0; F 0 0 1.733", basis="aug-cc-pVDZ",
            unit="Bohr", verbose=3)
cav = Cavity().add_mode(omega=0.1, lambda_scalar=0.05,
                         polarization=[0, 0, 1], name="IR_mode")
print(cav.summary())

# 黄金测试
passed, delta = lambda_limit_check(mol, cav, xc="b3lyp")
assert passed

# QED-DFT 计算
mf = QEDRKS(mol, cav); mf.xc = "b3lyp"; mf.conv_tol = 1e-9; mf.kernel()
mf.print_qed_summary()
gauge_invariance_check(mol, mf)

# 耦合强度扫描
print(f"\n{'λ':>8}  {'E_tot (Ha)':>18}  {'E_QED (meV)':>14}")
for lam in [0.0, 0.01, 0.05, 0.10, 0.20]:
    ci  = Cavity().add_mode(0.1, lam, [0,0,1])
    mfi = QEDRKS(mol, ci); mfi.xc="b3lyp"; mfi.verbose=0; mfi.kernel()
    bd  = mfi.qed_energy_breakdown()
    print(f"{lam:>8.3f}  {mfi.e_tot:>18.10f}  {bd['E_QED_total']*27211.4:>14.3f}")
