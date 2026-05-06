# -*- coding: utf-8 -*-
"""
qeddft.analysis.polariton  —  极化子态分析（v2，修复版）

极化子识别逻辑：
  真正的极化子 = 光子权重接近 50% 的混合态
  混合度 = 1 - |photon_wt - 0.5| * 2   (1=完美混合, 0=纯态)

is_resonant() 方法判断是否存在真实 Rabi 劈裂。
"""
import numpy as np


class PolaritonAnalysis:
    """
    极化子态分析工具。

    Parameters
    ----------
    td : QEDTDA 或 QEDTDDFT（已调用 kernel()）
    """

    def __init__(self, td):
        self.td     = td
        self.cavity = td.cavity
        self._done  = False

    # ------------------------------------------------------------------
    # 主计算入口
    # ------------------------------------------------------------------

    def compute(self) -> "PolaritonAnalysis":
        """提取所有极化子特征量，返回 self 支持链式调用。"""
        td = self.td
        if not hasattr(td, "e"):
            raise RuntimeError("请先调用 td.kernel()")

        self.energies       = np.asarray(td.e) * 27.2114        # eV
        self.photon_weight  = np.array(td.photon_weight,  dtype=float)
        self.exciton_weight = np.array(td.exciton_weight, dtype=float)
        self.osc_strength   = td.oscillator_strength()

        # 混合度：1 = 完美 50/50，0 = 纯态
        self.mixing = 1.0 - np.abs(self.photon_weight - 0.5) * 2.0

        # 极化子对：混合度最高的两个态，按能量升序
        sorted_by_mix = np.argsort(self.mixing)[::-1]
        top2 = sorted(sorted_by_mix[:2].tolist())
        self.lp_idx = int(top2[0])
        self.up_idx = int(top2[1])

        self.lp_energy      = float(self.energies[self.lp_idx])
        self.up_energy      = float(self.energies[self.up_idx])
        self.rabi_splitting = abs(self.up_energy - self.lp_energy)
        self.max_mixing     = float(self.mixing[sorted_by_mix[0]])

        self._done = True
        return self

    # ------------------------------------------------------------------
    # 共振判断
    # ------------------------------------------------------------------

    def is_resonant(self, threshold: float = 0.3) -> bool:
        """
        判断腔是否真正与激子共振。

        混合度 > threshold 认为是真极化子。
        threshold=0.3 对应光子权重在 35%–65% 之间。
        """
        if not self._done:
            self.compute()
        return bool(self.max_mixing > threshold)

    def resonance_suggestion(self) -> str:
        """
        分析不共振的原因并给出调整建议。
        检查两种情况：
          A. 腔频率与有强耦合的激发态失谐
          B. 极化方向与激发态跃迁偶极正交（耦合为零）
        """
        if not self._done:
            self.compute()

        lines = []
        for mode in self.cavity.modes:
            omega_ev = mode.omega_ev()
            lam      = mode.lambda_vec   # shape (3,)

            # 所有激子态的能量
            ex_mask = self.exciton_weight > 0.5
            if not ex_mask.any():
                lines.append(f"  [{mode.name}]: 无法识别激子态")
                continue

            e_exc = self.energies[ex_mask]
            f_exc = self.osc_strength[ex_mask]

            # 找最近的激子态
            nearest_idx = int(np.argmin(np.abs(e_exc - omega_ev)))
            nearest_e   = float(e_exc[nearest_idx])
            nearest_f   = float(f_exc[nearest_idx])
            detuning    = abs(omega_ev - nearest_e) * 1000   # meV

            # 找振子强度最大的激子态（最强偶极矩）
            brightest_idx = int(np.argmax(f_exc))
            brightest_e   = float(e_exc[brightest_idx])
            brightest_f   = float(f_exc[brightest_idx])

            lines.append(f"  腔模 [{mode.name}]  ω={omega_ev:.4f} eV  λ={lam}")
            lines.append(f"    ├ 最近激子态:   {nearest_e:.4f} eV  (f={nearest_f:.4f})")
            lines.append(f"    │   失谐量:     {detuning:.1f} meV")
            lines.append(f"    ├ 最亮激子态:   {brightest_e:.4f} eV  (f={brightest_f:.4f})")

            if brightest_f < 0.001:
                lines.append(f"    │   警告: 所有激子态振子强度均极小")
                lines.append(f"    │   可能原因: 腔极化方向与跃迁偶极矩正交")
                lines.append(f"    │   建议: 改变极化方向 polarization")
            elif detuning > 100:
                omega_suggest = brightest_e / 27.2114
                lines.append(f"    └ 建议腔频率: {omega_suggest:.6f} a.u."
                              f" ({brightest_e:.4f} eV) 以匹配最亮激子态")
            else:
                lines.append(f"    └ 频率接近共振，但耦合弱——检查极化方向是否与跃迁偶极平行")

        return "\n".join(lines) if lines else "无法生成建议"

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def hopfield_coeffs(self) -> np.ndarray:
        """各态的光子 Hopfield 系数 |β|², shape (nstates,)"""
        if not self._done:
            self.compute()
        return self.photon_weight

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def print_report(self):
        """打印极化子分析完整报告。"""
        if not self._done:
            self.compute()

        print("=" * 60)
        print("极化子分析报告")
        print("=" * 60)

        if self.is_resonant():
            print(f"  共振状态:    ✓ 真极化子  (混合度 {self.max_mixing:.3f})")
            print(f"  Rabi 劈裂:   {self.rabi_splitting*1000:.2f} meV"
                  f"  ({self.rabi_splitting:.4f} eV)")
            print(f"  下极化子 LP: {self.lp_energy:.4f} eV"
                  f"  (光子={self.photon_weight[self.lp_idx]:.3f},"
                  f" 激子={self.exciton_weight[self.lp_idx]:.3f})")
            print(f"  上极化子 UP: {self.up_energy:.4f} eV"
                  f"  (光子={self.photon_weight[self.up_idx]:.3f},"
                  f" 激子={self.exciton_weight[self.up_idx]:.3f})")
        else:
            print(f"  共振状态:    ✗ 非共振  (最大混合度 {self.max_mixing:.3f} < 0.3)")
            print(f"  原因分析与建议：")
            print(self.resonance_suggestion())

        print("-" * 60)
        print(f"{'态':>4} {'E/eV':>8} {'振子强度':>12} "
              f"{'光子%':>7} {'激子%':>7} {'混合度':>7}")
        for i, E in enumerate(self.energies):
            mark = " ◀LP" if i == self.lp_idx and self.is_resonant() else \
                   " ◀UP" if i == self.up_idx and self.is_resonant() else ""
            print(f"{i+1:>4} {E:>8.4f} {self.osc_strength[i]:>12.6f} "
                  f"{self.photon_weight[i]*100:>6.2f}% "
                  f"{self.exciton_weight[i]*100:>6.2f}% "
                  f"{self.mixing[i]:>7.3f}{mark}")
        print("=" * 60)
