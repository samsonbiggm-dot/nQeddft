# -*- coding: utf-8 -*-
"""
nqeddft.scf.qed_rks  —  QED Restricted Kohn-Sham DFT

继承 pyscf.dft.rks.RKS，通过三个 Hook 注入 Pauli-Fierz 修正：
  get_veff()     ← V_dse_J + V_dse_K + V_bilinear
  energy_elec()  ← E_DSE + E_bilinear + E_photon
  （get_hcore 保持原样：密度无关的几何项可按需扩展）

黄金测试：令 lambda_scalar=0 时，QEDRKS 结果精确等于 pyscf.dft.RKS。
"""
import numpy as np
from pyscf.dft import rks
from .pauli_fierz import PauliFierzCorrection
from ..cavity import Cavity


class QEDRKS(rks.RKS):
    """
    QED Restricted Kohn-Sham DFT。

    Parameters
    ----------
    mol     : pyscf.gto.Mole
    cavity  : Cavity
    **kwargs: 透传给 pyscf.dft.rks.RKS（如 xc='b3lyp'）

    Examples
    --------
    >>> mf = QEDRKS(mol, cav)
    >>> mf.xc = 'b3lyp'
    >>> e = mf.kernel()
    >>> mf.print_qed_summary()
    """
    _keys = rks.RKS._keys | {'cavity'}

    def __init__(self, mol, cavity: Cavity, **kwargs):
        cavity.validate()
        super().__init__(mol, **kwargs)
        self.cavity = cavity
        self._pf    = PauliFierzCorrection(mol, cavity)

    # ── Hook 2: 有效势 ─────────────────────────────────────────────────
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm  is None: dm  = self.make_rdm1()
        veff  = super().get_veff(mol, dm, dm_last, vhf_last, hermi)
        veff_array = np.array(veff)           # 先转成普通数组做加法
        veff_array += self._pf.get_vqed(dm)   # 普通数组原地加
        veff[:] = veff_array
        return veff
        
    # ── Hook 3: 电子能量 ───────────────────────────────────────────────
    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        e_elec, e_coul = super().energy_elec(dm, h1e, vhf)
        e_qed = self._pf.energy_qed_total(dm)
        if self.verbose >= 4:
            bd = self._pf
            self.stdout.write(
                f'  E_DSE={bd.energy_dse(dm):+.8f}  '
                f'E_bilin={bd.energy_bilinear(dm):+.8f}  '
                f'E_ph={bd.energy_photon(dm):+.8f}\n')
        return e_elec + e_qed, e_coul

    # ── 分析工具 ───────────────────────────────────────────────────────
    def qed_energy_breakdown(self) -> dict:
        if not hasattr(self, 'mo_energy'):
            raise RuntimeError("请先调用 kernel()")
        dm = self.make_rdm1()
        return {
            'E_tot':       self.e_tot,
            'E_DSE':       self._pf.energy_dse(dm),
            'E_bilinear':  self._pf.energy_bilinear(dm),
            'E_photon':    self._pf.energy_photon(dm),
            'E_QED_total': self._pf.energy_qed_total(dm),
            'photon_numbers': self._pf.photon_number_mean(dm),
            'dipole_au':   self._pf.dip_ints.dipole_moment(dm),
        }

    def print_qed_summary(self):
        bd  = self.qed_energy_breakdown()
        dip = bd['dipole_au']
        print("=" * 55)
        print("QED-DFT 结果摘要")
        print("=" * 55)
        print(f"  总能量:        {bd['E_tot']:.10f} Ha")
        print(f"  E_DSE:         {bd['E_DSE']:+.8f} Ha")
        print(f"  E_bilinear:    {bd['E_bilinear']:+.8f} Ha")
        print(f"  E_photon:      {bd['E_photon']:+.8f} Ha")
        etot_meV = bd['E_QED_total'] * 27211.4
        print(f"  QED 修正总计:  {bd['E_QED_total']:+.8f} Ha  ({etot_meV:+.3f} meV)")
        print(f"  偶极矩 (a.u.): [{dip[0]:.5f}, {dip[1]:.5f}, {dip[2]:.5f}]")
        print("  光子数期望值:")
        for name, n in bd['photon_numbers'].items():
            print(f"    {name}: <n> = {n:.6f}")
        print(self.cavity.summary())

    def Gradients(self):
        """返回 QEDGradients 对象（与 PySCF mf.Gradients() 风格一致）"""
        from ..grad.qed_grad import QEDGradients
        return QEDGradients(self)

    def TDDFT(self):
        """返回 QEDTDDFT 对象"""
        from ..tdscf.qed_tddft import QEDTDDFT
        return QEDTDDFT(self, self.cavity)

    def TDA(self):
        """返回 QEDTDA 对象"""
        from ..tdscf.qed_tddft import QEDTDA
        return QEDTDA(self, self.cavity)
