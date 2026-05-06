# -*- coding: utf-8 -*-
"""
nqeddft  —  QED-DFT software package built on PySCF
版本：0.1.0  |  依赖：pyscf >= 2.3, numpy, scipy
"""
from .cavity              import Cavity, CavityMode
from .scf.qed_rks         import QEDRKS
from .scf.qed_rhf         import QEDRHF
from .scf.qed_uks         import QEDUKS
from .tdscf.qed_tddft     import QEDTDDFT, QEDTDA
from .cc.qed_ccsd         import QEDCCSD
from .grad.qed_grad       import QEDGradients
from .analysis.polariton  import PolaritonAnalysis
from .analysis.spectrum   import AbsorptionSpectrum
from .phonon.qed_phonon   import QEDPhonon
from .tst                 import (StationaryPoint, QEDTST, kie_at_T, comprehensive_analysis)
from .saddle              import (find_ts, CINEB, Dimer, validate_ts, SaddleResult)
from .validation.checks   import gauge_invariance_check, convergence_nph

__version__ = "0.1.0"
__all__ = [
    "Cavity", "CavityMode",
    "QEDRKS", "QEDRHF", "QEDUKS",
    "QEDTDDFT", "QEDTDA",
    "QEDCCSD",
    "QEDGradients",
    "PolaritonAnalysis", "AbsorptionSpectrum",
    "QEDPhonon",
    "gauge_invariance_check", "convergence_nph",
    # TST
    "StationaryPoint", "QEDTST",
    "kie_at_T", "comprehensive_analysis",
    #NEB
    "find_ts, CINEB, Dimer, validate_ts, SaddleResult"
]
