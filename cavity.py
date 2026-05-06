# -*- coding: utf-8 -*-
"""
nqeddft.cavity  —  腔场参数封装
物理约定（原子单位 a.u.）：1 a.u. 频率 ≈ 27.211 eV
耦合强度 λ（无量纲）：弱 <0.01 | 中等 0.01-0.1 | 强 >0.1
"""
import numpy as np
from dataclasses import dataclass
from typing import List
import warnings


@dataclass
class CavityMode:
    omega: float            # 模式频率 (a.u.)
    lambda_scalar: float    # 无量纲耦合强度 λ
    polarization: np.ndarray  # 极化方向单位矢量
    name: str = ""

    def __post_init__(self):
        self.polarization = np.asarray(self.polarization, dtype=float)
        norm = np.linalg.norm(self.polarization)
        if norm < 1e-12:
            raise ValueError("极化矢量不能为零")
        self.polarization /= norm

    @property
    def lambda_vec(self) -> np.ndarray:
        return self.lambda_scalar * self.polarization

    @property
    def sqrt_omega_half(self) -> float:
        return float(np.sqrt(self.omega / 2.0))

    @property
    def coupling_prefactor(self) -> np.ndarray:
        return self.sqrt_omega_half * self.lambda_vec

    def omega_ev(self) -> float:
        return self.omega * 27.2114

    def regime(self) -> str:
        l = self.lambda_scalar
        if l < 0.01:  return "弱耦合"
        if l < 0.1:   return "中等耦合"
        if l < 0.3:   return "强耦合"
        return "超强耦合"

    def rabi_estimate(self, dip_au: float) -> float:
        """Ω_R ≈ 2·sqrt(ω/2)·λ·|d_eg| (a.u.)"""
        return 2.0 * self.sqrt_omega_half * self.lambda_scalar * abs(dip_au)


class Cavity:
    """
    多模光学腔参数容器，支持链式 add_mode() 构建。

    Examples
    --------
    >>> cav = (Cavity()
    ...        .add_mode(omega=0.1, lambda_scalar=0.05, polarization=[0,0,1])
    ...        .add_mode(omega=0.12, lambda_scalar=0.02, polarization=[1,0,0]))
    """
    def __init__(self):
        self.modes: List[CavityMode] = []

    def add_mode(self, omega: float, lambda_scalar: float,
                 polarization=(0., 0., 1.), name: str = "") -> "Cavity":
        if not name:
            name = f"mode_{len(self.modes)}"
        self.modes.append(CavityMode(float(omega), float(lambda_scalar),
                                     np.array(polarization, float), name))
        return self

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def validate(self):
        if self.n_modes == 0:
            raise ValueError("Cavity 为空，请先调用 add_mode()")
        for i, m in enumerate(self.modes):
            if m.omega <= 0:
                raise ValueError(f"模式 {i}: omega={m.omega} 必须 > 0")
            if m.lambda_scalar < 0:
                raise ValueError(f"模式 {i}: lambda 不能为负")
            if m.lambda_scalar > 2.0:
                warnings.warn(f"模式 {i}: λ={m.lambda_scalar:.3f} 超过 2.0，"
                               "偶极近似可能失效")

    def summary(self) -> str:
        lines = ["腔场参数", "=" * 40]
        for i, m in enumerate(self.modes):
            lines += [
                f"  [{m.name}]  ω={m.omega:.5f} a.u. ({m.omega_ev():.3f} eV)",
                f"          λ={m.lambda_scalar:.5f}  ({m.regime()})",
                f"          ε={m.polarization}",
            ]
        return "\n".join(lines)

    def __repr__(self):
        return f"Cavity(n_modes={self.n_modes})"
