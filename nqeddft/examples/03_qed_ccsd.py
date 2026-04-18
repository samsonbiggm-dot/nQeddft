# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""示例 03：QED-CCSD 高精度能量计算"""
from pyscf import gto
from nqeddft import Cavity, QEDRHF, QEDCCSD

mol = gto.M(atom="H 0 0 0; F 0 0 1.733", basis="cc-pVDZ",
            unit="Bohr", verbose=3)
cav = Cavity().add_mode(0.1, 0.05, [0,0,1])

mf = QEDRHF(mol, cav); mf.kernel()
cc = QEDCCSD(mf, cav); e, _, _ = cc.kernel()
print(f"QED-CCSD 总能量: {e:.10f} Ha")
print(f"QED 光子修正:    {cc.e_corr_qed:.8f} Ha")
for name, d in cc.qed_breakdown().items():
    print(f"  [{name}] E_DSE={d['E_DSE']:+.8f}  E_pt2={d['E_pt2']:+.8f}")
