# -*- coding: utf-8 -*-
"""
nqeddft  —  QED-DFT software package built on PySCF
版本：0.2.0  |  依赖：pyscf >= 2.3, numpy, scipy
"""
from .cavity              import Cavity, CavityMode
from .cavity_field        import (SpectralField,
                                   LorentzianField, GaussianField,
                                   OhmicField, FlatbandField,
                                   MultiPeakField, CustomField,
                                   cm1_to_au, au_to_cm1,
                                   ev_to_au,  au_to_ev,
                                   nm_to_au,  au_to_nm)
from .scf.qed_rks         import QEDRKS
from .scf.qed_rhf         import QEDRHF
from .scf.qed_uks         import QEDUKS
from .tdscf.qed_tddft     import QEDTDDFT, QEDTDA
from .cc.qed_ccsd         import QEDCCSD
from .grad.qed_grad       import QEDGradients, QEDUKSGradients
from .analysis.polariton  import PolaritonAnalysis
from .analysis.spectrum   import AbsorptionSpectrum
from .phonon.qed_phonon   import QEDPhonon
from .validation.checks   import gauge_invariance_check, convergence_nph

__version__ = "0.2.0"
__all__ = [
    # 腔场
    "Cavity", "CavityMode",
    # 谱密度
    "SpectralField",
    "LorentzianField", "GaussianField", "OhmicField",
    "FlatbandField", "MultiPeakField", "CustomField",
    # 单位换算
    "cm1_to_au", "au_to_cm1", "ev_to_au", "au_to_ev", "nm_to_au", "au_to_nm",
    # SCF
    "QEDRKS", "QEDRHF", "QEDUKS",
    # 激发态
    "QEDTDDFT", "QEDTDA",
    # 耦合簇
    "QEDCCSD",
    # 梯度
    "QEDGradients", "QEDUKSGradients",
    # 分析
    "PolaritonAnalysis", "AbsorptionSpectrum",
    # 声子
    "QEDPhonon",
    # 验证
    "gauge_invariance_check", "convergence_nph",
]
