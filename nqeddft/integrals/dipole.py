# -*- coding: utf-8 -*-
"""
nqeddft.integrals.dipole  —  偶极矩积分封装

封装 PySCF mol.intor 调用，提供缓存和常用派生量。

符号约定：
    mol.intor('int1e_r') 返回 +<μ|r|ν>
    电偶极算符 d = -Σ_i r_i（电子），故 d_μν = -<μ|r|ν>
    核贡献 d_nuc = Σ_I Z_I R_I 在需要全偶极矩时单独加入
"""
import numpy as np


class DipoleIntegrals:
    """
    管理 AO 基偶极矩积分的计算与缓存。

    Attributes
    ----------
    dip_ao : ndarray (3, nao, nao)
        电子偶极算符矩阵元 d_μν^k = -<μ|r_k|ν>，k=x,y,z
    """
    def __init__(self, mol):
        self.mol = mol
        self._dip_ao  = None
        self._quad_ao = None
        self._vel_ao  = None

    def invalidate_cache(self):
        """几何更新后清除所有缓存（势能面扫描必须调用）"""
        self._dip_ao = self._quad_ao = self._vel_ao = None

    @property
    def dip_ao(self) -> np.ndarray:
        if self._dip_ao is None:
            self._dip_ao = -self.mol.intor('int1e_r', comp=3)
        return self._dip_ao

    def lambda_dot_d(self, lambda_vec: np.ndarray) -> np.ndarray:
        """(λ⃗·d̂)_μν = Σ_k λ_k d_μν^k, shape (nao,nao)"""
        return np.einsum('k,kpq->pq', np.asarray(lambda_vec), self.dip_ao)

    def dipole_moment(self, dm: np.ndarray) -> np.ndarray:
        """<d⃗> = Tr[d⃗·P], shape (3,), 单位 a.u."""
        dm_use = (dm[0] + dm[1]) if dm.ndim == 3 else dm
        return np.einsum('kpq,qp->k', self.dip_ao, dm_use)

    def lambda_dot_d_mean(self, lambda_vec: np.ndarray, dm: np.ndarray) -> float:
        """标量 <λ⃗·d̂> = Tr[(λ·d)·P]"""
        return float(np.dot(lambda_vec, self.dipole_moment(dm)))

    def nuclear_dipole(self) -> np.ndarray:
        """核贡献偶极矩 d_nuc = Σ_I Z_I R_I, shape (3,)"""
        mol = self.mol
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        return np.einsum('i,ix->x', charges, coords)

    def total_dipole(self, dm: np.ndarray) -> np.ndarray:
        """全偶极矩（电子+核），shape (3,)"""
        return self.dipole_moment(dm) + self.nuclear_dipole()

    def velocity_ao(self) -> np.ndarray:
        """速度规范动量积分 <μ|∇|ν>, shape (3,nao,nao)（规范验证用）"""
        if self._vel_ao is None:
            self._vel_ao = self.mol.intor('int1e_ipovlp', comp=3)
        return self._vel_ao

    def dip_grad_ao(self, atm_id: int) -> np.ndarray:
        """
        偶极积分对原子 atm_id 坐标的导数, shape (3,3,nao,nao)
        [方向l, 分量k, μ, ν]
        用于核梯度计算。
        """
        mol     = self.mol
        aoslice = mol.aoslice_by_atom()
        shl0, shl1, ao0, ao1 = aoslice[atm_id]
        nao = mol.nao_nr()
        buf = mol.intor('int1e_irp', comp=9,
                         shls_slice=(shl0, shl1, 0, mol.nbas))
        buf = buf.reshape(3, 3, ao1 - ao0, nao)
        ddip = np.zeros((3, 3, nao, nao))
        ddip[:, :, ao0:ao1, :] += buf
        ddip[:, :, :, ao0:ao1] += buf.transpose(0, 1, 3, 2)
        return -ddip   # 与偶极算符符号约定一致
