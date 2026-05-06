# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""示例 05：多模腔计算 — 正交极化双模腔"""
from pyscf import gto
from nqeddft import Cavity, QEDRKS, QEDTDA
from nqeddft.analysis import PolaritonAnalysis

mol = gto.M(atom="H 0 0 0; F 0 0 1.733", basis="cc-pVDZ",
            unit="Bohr", verbose=0)

cav = (Cavity()
       .add_mode(omega=0.38, lambda_scalar=0.04, polarization=[0,0,1], name="TE")
       .add_mode(omega=0.40, lambda_scalar=0.03, polarization=[1,0,0], name="TM"))
print(cav.summary())

mf = QEDRKS(mol, cav); mf.xc="b3lyp"; mf.verbose=0; mf.kernel()
mf.print_qed_summary()

td = QEDTDA(mf, cav); td.nstates=10; td.kernel()
td.print_spectrum()
PolaritonAnalysis(td).compute().print_report()
