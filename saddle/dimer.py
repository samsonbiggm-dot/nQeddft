# -*- coding: utf-8 -*-
"""
nqeddft.saddle.dimer
====================
Dimer 方法搜索一阶鞍点（过渡态）。

物理框架
--------
两个 image 沿单位向量 N 距离 ΔR/2，构成 "dimer"：
    R_1 = R_center + (ΔR/2) · N
    R_2 = R_center - (ΔR/2) · N

dimer 旋转 → 找最低不稳定模 N（对应最负 Hessian 本征值）
dimer 平移 → 沿最低不稳定模反向 + 其他方向正向最小化

每次 dimer 迭代:
  (a) 旋转：找使 dimer 能量极小的方向 N（即沿最低 Hessian 本征向量）
  (b) 平移：在 N 方向上反转力（"上山"），其他方向用正常梯度（"下山"）

QED 修饰
--------
所有 (E, F) 调用 QEDGradients/QEDUKSGradients，腔效应自动包含。
鞍点本身的虚频应当对应反应坐标，可以用 QEDPhonon 验证。

参考
----
  Henkelman & Jónsson, JCP 111, 7010 (1999)
  Heyden, Bell, Keil, JCP 123, 224101 (2005) — 改进的有限差分
"""
from __future__ import annotations

import numpy as np
import time
from typing import Callable, Optional
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DimerResult:
    """Dimer 优化结果"""
    coords: np.ndarray              # 最终鞍点坐标 (natm, 3) Bohr
    energy: float
    forces: np.ndarray              # 最终力（应当 ≈ 0）
    direction: np.ndarray           # 反应坐标方向 (natm, 3)
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)
    elapsed_sec: float = 0.0
    curvature: float = 0.0          # 沿 N 的曲率（应当 < 0 即不稳定模）


# ══════════════════════════════════════════════════════════════════════
# Dimer 主类
# ══════════════════════════════════════════════════════════════════════

class Dimer:
    """
    Dimer 方法搜索一阶鞍点。

    Parameters
    ----------
    energy_force_fn : callable(coords) -> (E, F)
    R_init          : (natm, 3) Bohr, 初始猜测（通常来自 NEB 的 climbing image）
    N_init          : (natm, 3) 单位向量, 初始猜测的反应坐标方向；
                       若为 None 则随机生成
    dR              : float, dimer 半距离 (Bohr)，默认 0.005

    Examples
    --------
    >>> ts = Dimer(ef_fn, R_neb_top, N_init=neb_tangent).run(max_iter=100)
    """

    def __init__(self,
                 energy_force_fn: Callable,
                 R_init: np.ndarray,
                 N_init: Optional[np.ndarray] = None,
                 dR: float = 0.005,
                 verbose: bool = True):
        self.ef_fn = energy_force_fn
        self.R     = np.asarray(R_init).copy()
        self.dR    = dR
        self.verbose = verbose

        # 反应坐标方向
        if N_init is None:
            # 随机方向
            rng = np.random.RandomState(42)
            N = rng.randn(*self.R.shape)
        else:
            N = np.asarray(N_init).astype(float)
        self.N = N / max(np.linalg.norm(N), 1e-12)

        # 缓存中心点的 (E, F)
        self._E_center = None
        self._F_center = None

    # ------------------------------------------------------------------
    # 旋转：找最低不稳定模方向
    # ------------------------------------------------------------------

    def _rotate(self, max_rot: int = 4, F_rot_tol: float = 0.01):
        """
        旋转 dimer 找最低 Hessian 本征向量。

        使用 Heyden et al. 2005 改进算法：
        旋转力 F_rot = F_⊥(R_1) - F_⊥(R_2)，
        其中 F_⊥ = F - (F·N)N。

        最大旋转曲率公式（dimer 力沿 N 的二阶导）：
          C_N = (F_2 - F_1)·N / dR
        """
        for j in range(max_rot):
            # R_1 = R_center + dR·N, 但只用 R_1 单独算（中心点和 R_1 算 F_2）
            R_1 = self.R + self.dR * self.N
            E_1, F_1 = self.ef_fn(R_1)
            F_2 = 2.0 * self._F_center - F_1   # 用 F_1 + F_2 = 2 F_center 求 F_2
                                                # （线性近似，节省一次 SCF）

            # 沿 N 的 dimer 曲率 C_N（用力的有限差分）
            # dF/dR 沿 N 方向 ≈ (F_2 - F_1)/(2 dR)
            C_N = float(np.sum((F_2 - F_1) * self.N) / (2.0 * self.dR))

            # 旋转力（垂直于 N）
            F_rot = (F_1 - F_2) - np.sum((F_1 - F_2) * self.N) * self.N
            F_rot_norm = np.linalg.norm(F_rot)

            if F_rot_norm < F_rot_tol:
                if self.verbose and j == 0:
                    pass  # 不打印——太啰嗦
                return C_N

            # 旋转方向 Theta = F_rot / |F_rot|
            Theta = F_rot / F_rot_norm

            # 旋转角度 dphi 用 Heyden 公式（二次拟合）
            # 在 phi=0 处计算 N、Theta、F_rot
            # 在 phi=phi_test 处再算一次 dimer，再二次拟合

            phi_test = 0.05  # rad
            cos_t = np.cos(phi_test); sin_t = np.sin(phi_test)
            N_test     = N_orig = self.N
            N_test_new = cos_t * N_test + sin_t * Theta

            R_1_test = self.R + self.dR * N_test_new
            E_1_test, F_1_test = self.ef_fn(R_1_test)
            F_2_test = 2.0 * self._F_center - F_1_test
            F_rot_test = (F_1_test - F_2_test) \
                         - np.sum((F_1_test - F_2_test) * N_test_new) * N_test_new
            F_rot_test_along_Theta = np.sum(F_rot_test
                * (cos_t * Theta - sin_t * N_test))   # 旋转后 Theta 方向

            # 用 F_rot 沿 Theta 的初始值和测试值做二次插值
            F_rot_along_Theta = float(np.sum(F_rot * Theta))
            C0 = F_rot_along_Theta
            C1 = (F_rot_test_along_Theta - F_rot_along_Theta) / phi_test

            if abs(C1) < 1e-12:
                phi_min = 0.0
            else:
                phi_min = -C0 / C1   # 一阶拟合
                # 限制单步旋转
                phi_min = float(np.clip(phi_min, -np.pi/4, np.pi/4))

            # 应用旋转
            cos_p = np.cos(phi_min); sin_p = np.sin(phi_min)
            self.N = cos_p * N_test + sin_p * Theta
            # 归一化
            self.N /= max(np.linalg.norm(self.N), 1e-12)

        return C_N

    # ------------------------------------------------------------------
    # 平移：沿 N 反转力，做"上山"步骤
    # ------------------------------------------------------------------

    def _translation_force(self, F_center: np.ndarray, C_N: float) -> np.ndarray:
        """
        构造修饰后的"鞍点搜索力"：
        若沿 N 方向曲率 < 0（不稳定模）：F_eff = F - 2(F·N)N （沿 N 反向，"上山"）
        若曲率 > 0（不在不稳定区）：     F_eff = -(F·N)N （仅沿 N 上山，其他不动）
        """
        F_para = np.sum(F_center * self.N) * self.N
        F_perp = F_center - F_para

        if C_N < 0:
            # 不稳定模找到，标准 dimer：沿 N 反向 + 垂直方向正常
            F_eff = -F_para + F_perp
        else:
            # 还没到不稳定区：仅沿 N 上山
            F_eff = -F_para

        return F_eff

    # ------------------------------------------------------------------
    # 主运行
    # ------------------------------------------------------------------

    def run(self,
            max_iter: int = 100,
            f_tol: float = 5e-4,
            dt: float = 0.1,
            dt_max: float = 0.5) -> DimerResult:
        """
        运行 dimer 优化。使用 backtracking line search 而非 FIRE，
        以确保在 stiff PES 上的鲁棒性。

        Parameters
        ----------
        max_iter : 最大迭代
        f_tol    : 最大力分量收敛阈值 (Ha/Bohr)
        dt       : 初始步长（兼容参数，实际用 line search 自适应）
        dt_max   : 最大步长（坐标单位）
        """
        t0 = time.time()
        history = []

        # 中心点初始 SCF
        E_center, F_center = self.ef_fn(self.R)
        self._E_center = E_center
        self._F_center = np.asarray(F_center).reshape(self.R.shape)

        if self.verbose:
            print(f"\n══════ Dimer 启动 ══════")
            print(f"  初始 E = {E_center:.6f} Ha")
            print(f"  dR = {self.dR}, f_tol = {f_tol}")

        # 自适应步长（基于近期 |F|）
        step_size = dt    # 初始步长（坐标单位）
        C_N = 0.0

        for iter_n in range(max_iter):
            iter_t0 = time.time()

            # 步骤 1: 旋转 dimer 找最低不稳定模方向
            C_N = self._rotate()

            # 步骤 2: 计算修饰后的平移力
            F_eff = self._translation_force(self._F_center, C_N)

            # 收敛检查（用真实力 F_center，不是 F_eff）
            max_force = float(np.max(np.abs(self._F_center)))
            history.append(max_force)

            if self.verbose:
                print(f"  iter {iter_n:3d}: max|F| = {max_force:.4e}, "
                      f"E = {self._E_center:.6f}, "
                      f"C_N = {C_N:+.4e}, "
                      f"step = {step_size:.4f}, "
                      f"用时 {time.time()-iter_t0:.1f}s")

            if max_force < f_tol and C_N < 0:
                if self.verbose:
                    print(f"  ✓ 收敛于鞍点（曲率 C_N = {C_N:.3e} < 0）")
                return DimerResult(
                    coords=self.R.copy(),
                    energy=self._E_center,
                    forces=self._F_center.copy(),
                    direction=self.N.copy(),
                    n_iter=iter_n + 1,
                    converged=True,
                    history=history,
                    elapsed_sec=time.time() - t0,
                    curvature=C_N,
                )

            # 步骤 3: backtracking line search
            # 试探方向：F_eff 的归一化方向
            F_eff_norm = np.linalg.norm(F_eff)
            if F_eff_norm < 1e-12:
                if self.verbose:
                    print(f"    [warn] |F_eff| ≈ 0，放弃")
                break

            direction = F_eff / F_eff_norm

            # 试探步长，从 step_size 开始，能量飙升就缩
            success = False
            for backtrack in range(8):
                trial_step = step_size * direction
                # 单分量截断
                comp_max = float(np.max(np.abs(trial_step)))
                if comp_max > dt_max:
                    trial_step *= dt_max / comp_max

                R_trial = self.R + trial_step
                try:
                    E_trial, F_trial = self.ef_fn(R_trial)
                except Exception:
                    E_trial = float('inf')
                    F_trial = None

                # Dimer "上山方向"的步长合理性判据：
                # F_eff 包含沿 N 反向的"上山力"，所以能量上升一定程度是正常的。
                # 但不应飞掉。判据：
                #   |E_trial - E_center| < some_max_increase
                # 与梯度大小相关。
                # 最合理的"线性近似"：
                #   ΔE_predicted ≈ -F_eff · trial_step（梯度·位移，有符号）
                #   实际 ΔE 不应超过 |ΔE_predicted| 的 5 倍（Wolfe 类似）
                if F_trial is not None and np.isfinite(E_trial):
                    delta_E = E_trial - self._E_center
                    predicted = -float(np.sum(F_eff * trial_step))
                    # 接受条件：|实际| < 5× |预测| + 容差
                    bound = 5.0 * abs(predicted) + 1e-3 * F_eff_norm
                    if abs(delta_E) < bound:
                        # 接受
                        self.R = R_trial
                        self._E_center = E_trial
                        self._F_center = np.asarray(F_trial).reshape(self.R.shape)
                        # 探索：步长适度增加
                        step_size = min(step_size * 1.2, dt_max * 2)
                        success = True
                        break
                # 否则缩小步长
                step_size *= 0.5

            if not success:
                if self.verbose:
                    print(f"    [warn] 8 次回溯仍未找到可接受步长，"
                          f"step_size = {step_size:.2e}，停止")
                break

            # 最低步长保护
            if step_size < 1e-8:
                if self.verbose:
                    print(f"    [warn] step_size 过小 {step_size}，停止")
                break

        return DimerResult(
            coords=self.R.copy(),
            energy=self._E_center,
            forces=self._F_center.copy(),
            direction=self.N.copy(),
            n_iter=iter_n + 1,
            converged=False,
            history=history,
            elapsed_sec=time.time() - t0,
            curvature=C_N,
        )
