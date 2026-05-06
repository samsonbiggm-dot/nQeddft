# -*- coding: utf-8 -*-
"""nqeddft.scf.qed_uks  —  QED Unrestricted Kohn-Sham（开壳层体系）"""
import numpy as np
from pyscf.dft import uks
from .pauli_fierz import PauliFierzCorrection
from ..cavity import Cavity


class QEDUKS(uks.UKS):
    _keys = uks.UKS._keys | {'cavity'}

    def __init__(self, mol, cavity: Cavity, **kwargs):
        cavity.validate()
        super().__init__(mol, **kwargs)
        self.cavity = cavity
        self._pf    = PauliFierzCorrection(mol, cavity)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm  is None: dm  = self.make_rdm1()
        veff   = super().get_veff(mol, dm, dm_last, vhf_last, hermi)
        dm_tot = dm[0] + dm[1] if (hasattr(dm, 'ndim') and dm.ndim == 3) else dm
        vqed   = self._pf.get_vqed(dm_tot)
        # 用切片赋值保留 NPArrayWithTag 上的 .ecoul 等属性
        if hasattr(veff, 'ndim') and veff.ndim == 3:
            veff_array = np.array(veff)
            veff_array[0] += vqed
            veff_array[1] += vqed
            veff[:] = veff_array
        else:
            veff_array = np.array(veff)
            veff_array += vqed
            veff[:] = veff_array
        return veff

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        e_elec, e_coul = super().energy_elec(dm, h1e, vhf)
        dm_tot = dm[0] + dm[1] if (hasattr(dm, 'ndim') and dm.ndim == 3) else dm
        return e_elec + self._pf.energy_qed_total(dm_tot), e_coul
