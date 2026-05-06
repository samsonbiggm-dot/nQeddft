# -*- coding: utf-8 -*-
"""nqeddft.scf.qed_rhf  —  QED Restricted Hartree-Fock（用于 QED-CCSD 参考态）"""
import numpy as np
from pyscf.scf import hf
from .pauli_fierz import PauliFierzCorrection
from ..cavity import Cavity


class QEDRHF(hf.RHF):
    _keys = hf.RHF._keys | {'cavity'}

    def __init__(self, mol, cavity: Cavity):
        cavity.validate()
        super().__init__(mol)
        self.cavity = cavity
        self._pf    = PauliFierzCorrection(mol, cavity)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm  is None: dm  = self.make_rdm1()
        veff = super().get_veff(mol, dm, dm_last, vhf_last, hermi)
        # 用切片赋值保留 NPArrayWithTag 上的 .ecoul 等属性
        veff_array = np.array(veff)
        veff_array += self._pf.get_vqed(dm)
        veff[:] = veff_array
        return veff

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        e_elec, e_coul = super().energy_elec(dm, h1e, vhf)
        return e_elec + self._pf.energy_qed_total(dm), e_coul

    def CCSD(self):
        from ..cc.qed_ccsd import QEDCCSD
        return QEDCCSD(self, self.cavity)
