# -*- coding: utf-8 -*-
"""
nqeddft.saddle.neb
==================
Climbing-Image Nudged Elastic Band (CI-NEB) — 腔感知的过渡态搜索。

物理框架
--------
NEB 把反应路径离散化为 N 个 image（构型）：
    {R_0, R_1, ..., R_{N-1}}  with R_0 = reactant, R_{N-1} = product

每个 intermediate image 受两类力：
  (1) 真实力 F_real_i = -∇E(R_i)   ← 用 QEDGradients 计算
  (2) 弹簧力 F_spring_i = k·(τ_i·(R_{i+1} - R_{i-1}))
其中 τ_i 是切线方向。

NEB 力的关键投影（消除"corner-cutting"）：
    F_NEB_i = F_real_i^⊥ + F_spring_i^∥
          ⊥ = 垂直于 τ_i, ∥ = 沿 τ_i

CI-NEB 的额外修正（爬坡 image）：
    最高 image 的力替换为：F_climb = F_real - 2(F_real·τ)τ
    使其沿反应路径"爬"到鞍点。

QED 修饰
--------
通过 QEDGradients/QEDUKSGradients 计算每个 image 的能量和力，
腔效应自然包含在 (1) 的 ∇E 中。腔不直接进入 (2)，
但腔修饰的 PES 改变路径几何 → spring 力分布间接受腔影响。

参考
----
  Henkelman & Jónsson, JCP 113, 9978 (2000)
  Henkelman, Uberuaga, Jónsson, JCP 113, 9901 (2000) — CI-NEB
  Sheppard, Terrell, Henkelman, JCP 128, 134106 (2008) — improved tangent
"""
from __future__ import annotations

import numpy as np
import time
from typing import Callable, Optional, Sequence
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NEBImage:
    """单个 image（沿反应路径的一个构型）"""
    coords: np.ndarray           # (natm, 3), Bohr
    energy: float = float('nan')
    forces: Optional[np.ndarray] = None    # 真实力 (natm, 3) Ha/Bohr
    is_climb: bool = False                  # 是否爬坡 image
    is_endpoint: bool = False               # 端点（reactant 或 product）
    converged: bool = False                 # 当前 SCF/grad 计算是否成功


@dataclass
class NEBResult:
    """NEB 输出"""
    images: list                          # list[NEBImage]
    n_iter: int = 0
    converged: bool = False
    ts_index: int = -1                    # 鞍点对应的 image 索引
    ts_image: Optional[NEBImage] = None
    history: list = field(default_factory=list)   # 每次迭代的最大力
    elapsed_sec: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# 路径插值（线性 + 中点改进）
# ══════════════════════════════════════════════════════════════════════

def linear_interpolate(R_init: np.ndarray, R_final: np.ndarray,
                       n_images: int) -> np.ndarray:
    """
    在 R_init 和 R_final 之间做线性内插。

    Parameters
    ----------
    R_init, R_final : ndarray (natm, 3), Bohr
    n_images        : 总图数（含端点）

    Returns
    -------
    ndarray, shape (n_images, natm, 3)
    """
    R_init  = np.asarray(R_init)
    R_final = np.asarray(R_final)
    fractions = np.linspace(0.0, 1.0, n_images)
    return np.array([R_init + f * (R_final - R_init) for f in fractions])


def idpp_interpolate(R_init: np.ndarray, R_final: np.ndarray,
                     n_images: int, n_steps: int = 100,
                     verbose: bool = False) -> np.ndarray:
    """
    Image Dependent Pair Potential (IDPP) 插值。

    比线性插值更优：通过最小化"虚拟成对距离势"避免化学键穿插。
    Smidstrup et al., JCP 140, 214106 (2014).

    Parameters
    ----------
    R_init, R_final : 端点几何 (natm, 3), Bohr
    n_images        : 总图数
    n_steps         : IDPP 优化迭代次数

    Returns
    -------
    initial path : ndarray (n_images, natm, 3)
    """
    natm = R_init.shape[0]
    # 起始路径
    path = linear_interpolate(R_init, R_final, n_images)

    # 计算端点的成对距离矩阵
    def pair_dist(R):
        diff = R[:, None, :] - R[None, :, :]    # (natm, natm, 3)
        return np.linalg.norm(diff, axis=-1)    # (natm, natm)

    d_init  = pair_dist(R_init)
    d_final = pair_dist(R_final)

    # 每个 image 的目标距离矩阵 (线性内插)
    fracs = np.linspace(0, 1, n_images)
    d_targets = np.array([d_init + f * (d_final - d_init) for f in fracs])

    # 简单梯度下降优化（仅中间 images）
    lr = 0.01
    for step in range(n_steps):
        for i in range(1, n_images - 1):
            R = path[i]
            d_tar = d_targets[i]
            d_cur = pair_dist(R)
            # 损失 L = sum_kl (d_kl - d_tar_kl)^2 / (d_tar_kl)^4
            # 防止除零
            denom = np.maximum(d_tar, 0.5) ** 4
            err = (d_cur - d_tar) / denom    # (natm, natm)
            # 梯度 dL/dR_k = sum_l 2 (d_kl - d_tar) * (R_k - R_l) / (denom * d_kl)
            d_safe = np.maximum(d_cur, 1e-6)
            diff = R[:, None, :] - R[None, :, :]    # (natm, natm, 3)
            grad = 2.0 * (err / d_safe)[..., None] * diff
            grad = grad.sum(axis=1)    # (natm, 3)
            # 更新（端点保持不动）
            path[i] = R - lr * grad

    if verbose:
        print(f"  IDPP 插值完成（{n_steps} 步）")

    return path


# ══════════════════════════════════════════════════════════════════════
# 切线计算（improved tangent）
# ══════════════════════════════════════════════════════════════════════

def _tangent(images: list, idx: int) -> np.ndarray:
    """
    Henkelman 改进型切线 (JCP 113, 9978).

    根据相邻 image 能量的相对大小选择前向或后向差分：
      若 E[i+1] > E[i] > E[i-1]:  τ = R[i+1] - R[i]
      若 E[i+1] < E[i] < E[i-1]:  τ = R[i] - R[i-1]
      其他情况:                    加权平均
    """
    R_prev = images[idx - 1].coords
    R_curr = images[idx].coords
    R_next = images[idx + 1].coords
    E_prev = images[idx - 1].energy
    E_curr = images[idx].energy
    E_next = images[idx + 1].energy

    tau_plus  = R_next - R_curr
    tau_minus = R_curr - R_prev

    if E_next > E_curr and E_curr > E_prev:
        tau = tau_plus
    elif E_next < E_curr and E_curr < E_prev:
        tau = tau_minus
    else:
        # 极值附近：加权平均
        dE_max = max(abs(E_next - E_curr), abs(E_curr - E_prev))
        dE_min = min(abs(E_next - E_curr), abs(E_curr - E_prev))
        if E_next > E_prev:
            tau = tau_plus * dE_max + tau_minus * dE_min
        else:
            tau = tau_plus * dE_min + tau_minus * dE_max

    norm = np.linalg.norm(tau)
    if norm < 1e-10:
        return np.zeros_like(tau)
    return tau / norm


# ══════════════════════════════════════════════════════════════════════
# NEB 力的计算（含 climbing image）
# ══════════════════════════════════════════════════════════════════════

def _neb_forces(images: list, k_spring: float = 0.1) -> list:
    """
    计算每个 image 的 NEB 力（含 climbing 修饰）。

    Parameters
    ----------
    images   : list[NEBImage]
    k_spring : 弹簧常数 (Ha/Bohr²)，典型 0.05-0.5

    Returns
    -------
    list[ndarray (natm, 3)]：每个 image 的有效力（端点为零）
    """
    n = len(images)
    neb_forces = [np.zeros_like(im.coords) for im in images]

    for i in range(1, n - 1):
        im = images[i]
        F_real = im.forces                 # = -∇E
        tau    = _tangent(images, i)        # 单位切向

        # 投影
        F_real_para = np.sum(F_real * tau) * tau
        F_real_perp = F_real - F_real_para

        if im.is_climb:
            # CI-NEB: F_climb = F_real - 2 (F_real·τ)τ
            F_eff = F_real - 2.0 * F_real_para
        else:
            # 标准 NEB: 真实力的垂直分量 + 弹簧力的平行分量
            R_prev = images[i - 1].coords
            R_curr = images[i].coords
            R_next = images[i + 1].coords
            d_plus  = np.linalg.norm(R_next - R_curr)
            d_minus = np.linalg.norm(R_curr - R_prev)
            F_spring_para = k_spring * (d_plus - d_minus) * tau
            F_eff = F_real_perp + F_spring_para

        neb_forces[i] = F_eff

    return neb_forces


# ══════════════════════════════════════════════════════════════════════
# 主 NEB 类
# ══════════════════════════════════════════════════════════════════════

class CINEB:
    """
    Climbing-Image NEB 计算器。

    Parameters
    ----------
    energy_force_fn : callable(coords_bohr) -> (E, F)
        给定坐标计算能量和力。可由 QEDGradients.as_scanner() 包装而来。
    R_reactant      : (natm, 3), Bohr, 反应物坐标（已优化）
    R_product       : (natm, 3), Bohr, 产物坐标（已优化）
    n_images        : 总 image 数（含端点），推荐 7-12
    k_spring        : 弹簧常数 (Ha/Bohr²)，默认 0.1
    interpolation   : 'linear' 或 'idpp'

    Examples
    --------
    >>> from nqeddft import QEDRKS, Cavity
    >>> from nqeddft.grad import QEDGradients
    >>> 
    >>> mf_R, mf_P = ... 已优化的反应物/产物 mf
    >>> g_R = QEDGradients(mf_R)
    >>> 
    >>> def ef(coords):
    ...     mf_R.mol.set_geom_(coords, unit='Bohr')
    ...     mf_R.mol.build()
    ...     mf_R.kernel()
    ...     g = QEDGradients(mf_R); g.verbose = 0
    ...     return mf_R.e_tot, -g.kernel()
    >>> 
    >>> neb = CINEB(ef, mf_R.mol.atom_coords(), mf_P.mol.atom_coords(),
    ...             n_images=9)
    >>> result = neb.run(max_iter=50)
    >>> print(f"TS at image {result.ts_index}, E = {result.ts_image.energy}")
    """

    def __init__(self,
                 energy_force_fn: Callable,
                 R_reactant: np.ndarray,
                 R_product: np.ndarray,
                 n_images: int = 9,
                 k_spring: float = 0.1,
                 interpolation: str = 'idpp',
                 verbose: bool = True):

        if R_reactant.shape != R_product.shape:
            raise ValueError("反应物/产物原子数不一致")
        self.ef_fn       = energy_force_fn
        self.n_images    = n_images
        self.k_spring    = k_spring
        self.verbose     = verbose

        # 初始路径
        if interpolation == 'idpp':
            path = idpp_interpolate(R_reactant, R_product, n_images, verbose=verbose)
        else:
            path = linear_interpolate(R_reactant, R_product, n_images)

        # 构造 image 列表
        self.images = []
        for i, R in enumerate(path):
            im = NEBImage(coords=R.copy(),
                          is_endpoint=(i == 0 or i == n_images - 1))
            self.images.append(im)

    # ------------------------------------------------------------------
    # 力评估（端点跳过）
    # ------------------------------------------------------------------

    def _evaluate_all(self):
        """对所有非端点 image 计算 E 和 F。"""
        for i, im in enumerate(self.images):
            if im.is_endpoint and im.energy == im.energy:  # 已算过
                continue
            try:
                E, F = self.ef_fn(im.coords)
                im.energy    = float(E)
                im.forces    = np.asarray(F).reshape(im.coords.shape)
                im.converged = True
            except Exception as e:
                if self.verbose:
                    print(f"    image {i} SCF 失败: {e}")
                im.energy    = float('nan')
                im.forces    = np.zeros_like(im.coords)
                im.converged = False

    # ------------------------------------------------------------------
    # 选 climbing image: 最高能量的非端点
    # ------------------------------------------------------------------

    def _set_climber(self):
        # 重置
        for im in self.images:
            im.is_climb = False
        # 找最高
        energies = [im.energy if im.converged else -np.inf
                    for im in self.images[1:-1]]
        if not energies:
            return -1
        idx_in_inner = int(np.argmax(energies))
        idx_global   = idx_in_inner + 1
        self.images[idx_global].is_climb = True
        return idx_global

    # ------------------------------------------------------------------
    # FIRE 优化器（鲁棒，比单纯 steepest descent 好）
    # ------------------------------------------------------------------

    def run(self, max_iter: int = 50,
            f_tol: float = 5e-4,
            climb_after: int = 5,
            dt_max: float = 0.5,
            dt_init: float = 0.1,
            verbose: Optional[bool] = None) -> NEBResult:
        """
        运行 NEB。前 climb_after 步用普通 NEB，之后启用 climbing image。

        Parameters
        ----------
        max_iter   : 最大迭代次数
        f_tol      : 力收敛阈值 (Ha/Bohr) — 所有 image 上的最大力分量
        climb_after: 第几步启用 climbing image
        dt_max     : FIRE 最大步长 (Bohr)
        dt_init    : FIRE 初始步长

        Returns
        -------
        NEBResult
        """
        if verbose is None:
            verbose = self.verbose

        t0 = time.time()

        # FIRE 参数
        N_min = 5
        f_inc = 1.1
        f_dec = 0.5
        alpha_init = 0.1
        f_alpha = 0.99

        dt    = dt_init
        alpha = alpha_init
        velocities = [np.zeros_like(im.coords) for im in self.images]
        N_pos = 0

        history = []
        ts_index = -1

        if verbose:
            print(f"\n══════ CI-NEB 启动 ══════")
            print(f"  n_images = {self.n_images}, k_spring = {self.k_spring}")
            print(f"  收敛阈值 f_tol = {f_tol} Ha/Bohr")

        # 端点初始能量
        if verbose:
            print("\n[初始评估端点]")
        self._evaluate_all()
        E_R = self.images[0].energy
        E_P = self.images[-1].energy
        if verbose:
            print(f"  E(reactant) = {E_R:.6f} Ha")
            print(f"  E(product)  = {E_P:.6f} Ha")

        # 主循环
        for iter_n in range(max_iter):
            iter_t0 = time.time()

            # 评估所有中间 image
            self._evaluate_all()

            # 启用 climbing
            if iter_n >= climb_after:
                ts_index = self._set_climber()

            # 计算 NEB 力
            neb_F = _neb_forces(self.images, k_spring=self.k_spring)

            # 最大力（仅中间 image）
            max_force = max(np.max(np.abs(F)) for F in neb_F[1:-1])
            history.append(max_force)

            if verbose:
                E_arr = [im.energy for im in self.images]
                E_max_idx = int(np.argmax(E_arr[1:-1])) + 1
                E_max = E_arr[E_max_idx]
                E_max_rel = (E_max - E_R) * 27.2114
                climb_tag = f"(climb={ts_index})" if iter_n >= climb_after else ""
                print(f"  iter {iter_n:3d}: max|F| = {max_force:.4e}, "
                      f"E_max = {E_max:.6f} Ha (ΔE = {E_max_rel:.3f} eV) "
                      f"{climb_tag}, dt = {dt:.3f}, "
                      f"用时 {time.time()-iter_t0:.1f}s")

            # 收敛？
            if max_force < f_tol and iter_n >= climb_after + 2:
                if verbose:
                    print(f"  ✓ 收敛！")
                ts_index = self._set_climber()  # 再确定一次
                ts_image = self.images[ts_index] if ts_index >= 0 else None
                return NEBResult(
                    images=self.images,
                    n_iter=iter_n + 1,
                    converged=True,
                    ts_index=ts_index,
                    ts_image=ts_image,
                    history=history,
                    elapsed_sec=time.time() - t0,
                )

            # FIRE 更新
            P = sum(np.sum(v * F) for v, F in zip(velocities[1:-1], neb_F[1:-1]))
            if P > 0:
                N_pos += 1
                if N_pos > N_min:
                    dt = min(dt * f_inc, dt_max)
                    alpha *= f_alpha
            else:
                N_pos = 0
                dt *= f_dec
                alpha = alpha_init
                # 重置速度
                velocities = [np.zeros_like(im.coords) for im in self.images]

            # 速度更新（仅中间 image）
            for i in range(1, len(self.images) - 1):
                F = neb_F[i]
                v = velocities[i]
                F_norm = np.linalg.norm(F)
                v_norm = np.linalg.norm(v)
                if F_norm > 1e-12:
                    F_hat = F / F_norm
                    v_new = (1 - alpha) * v + alpha * v_norm * F_hat
                else:
                    v_new = (1 - alpha) * v
                v_new += dt * F
                # 步长保护（防止单步飞得太远）
                step_max = np.max(np.abs(v_new * dt))
                if step_max > 0.2:   # > 0.2 Bohr 就缩
                    v_new *= 0.2 / step_max
                velocities[i] = v_new

            # 坐标更新
            for i in range(1, len(self.images) - 1):
                self.images[i].coords += dt * velocities[i]

        # 未收敛
        ts_index = self._set_climber()
        ts_image = self.images[ts_index] if ts_index >= 0 else None
        return NEBResult(
            images=self.images,
            n_iter=max_iter,
            converged=False,
            ts_index=ts_index,
            ts_image=ts_image,
            history=history,
            elapsed_sec=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # 状态导出
    # ------------------------------------------------------------------

    def export_path(self) -> dict:
        """导出当前路径（用于断点续算）"""
        return {
            'n_images':  self.n_images,
            'k_spring':  self.k_spring,
            'images': [
                {
                    'coords':    im.coords.tolist(),
                    'energy':    im.energy,
                    'forces':    im.forces.tolist() if im.forces is not None else None,
                    'is_climb':  im.is_climb,
                    'is_endpoint': im.is_endpoint,
                    'converged': im.converged,
                }
                for im in self.images
            ],
        }

    def import_path(self, data: dict):
        """从字典恢复状态"""
        for i, d in enumerate(data['images']):
            self.images[i].coords = np.asarray(d['coords'])
            self.images[i].energy = d['energy']
            if d['forces'] is not None:
                self.images[i].forces = np.asarray(d['forces'])
            self.images[i].is_climb  = d['is_climb']
            self.images[i].converged = d['converged']
