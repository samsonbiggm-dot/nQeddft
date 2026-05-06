# -*- coding: utf-8 -*-
"""
nqeddft.saddle.ts_validate
==========================
过渡态合法性验证。

NEB/dimer 给出"候选 TS"后，必须做下面三层检查：
  1. **Hessian 检查**: 恰好 1 个虚频（其他都是正频或 0）
  2. **虚频幅度合理**: 通常 |ν‡| ∈ [200, 3500] cm⁻¹
                       太小可能是数值噪声，太大可能是几何不真实
  3. **反应坐标一致**: 虚频对应的正则模与 NEB tangent 重叠 > 0.5
                       否则可能找到了"假鞍点"（高阶鞍点或物理无关方向）

通过验证后，可计算：
  - 反向势垒（用于 Eckart 隧穿）
  - IRC 验证（可选，时间贵）

参考
----
  Baker & Bergman, JCC 14, 1085 (1993) — TS 验证最佳实践
  Tachibana, JCP 81, 4538 (1984) — 反应坐标的几何意义
"""
from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class TSValidation:
    """TS 验证报告"""
    is_valid:        bool
    n_imag:          int                    # 虚频数（应为 1）
    imag_freq_cm:    Optional[float]        # 虚频幅度 (cm⁻¹)，正数
    real_freqs_cm:   Optional[np.ndarray]   # 实频列表 (cm⁻¹)
    reaction_mode_overlap: Optional[float]  # 与参考方向的重叠 ∈ [0, 1]
    warnings:        list                    # 文字警告
    summary:         str

    @property
    def passed(self) -> bool:
        return self.is_valid


# ══════════════════════════════════════════════════════════════════════
# 单点 Hessian 计算（包装 QEDPhonon）
# ══════════════════════════════════════════════════════════════════════

def compute_ts_hessian(mf, coords_bohr: np.ndarray,
                       stepsize: float = 0.005,
                       verbose: bool = False) -> dict:
    """
    在指定坐标处计算 QED-DFT Hessian 和振动频率。

    Parameters
    ----------
    mf          : QEDRKS / QEDUKS（已配置 cavity）
    coords_bohr : 待评估的几何 (natm, 3) Bohr
    stepsize    : 数值 Hessian 步长

    Returns
    -------
    dict:
        freqs_cm  : (3*natm,) 频率，虚频为负
        modes     : (3*natm, 3*natm) 正则模（笛卡尔）
        hess      : Hessian (Ha/Bohr²)
        e_at_ts   : 该几何处的能量
    """
    from nqeddft.phonon import QEDPhonon

    # 设置几何
    mol = mf.mol
    mol.set_geom_(coords_bohr, unit='Bohr')
    mol.build()

    # 重新跑 SCF
    mf.kernel()
    e_at_ts = float(mf.e_tot)

    # Hessian
    ph = QEDPhonon(mf)
    hess = ph.numerical_hessian_fast(stepsize=stepsize, verbose=verbose)
    freqs, modes = ph.harmonic_analysis(hess)

    return {
        'freqs_cm': np.asarray(freqs),
        'modes':    np.asarray(modes),
        'hess':     hess,
        'e_at_ts':  e_at_ts,
    }


# ══════════════════════════════════════════════════════════════════════
# 验证主函数
# ══════════════════════════════════════════════════════════════════════

def validate_ts(freqs_cm: np.ndarray,
                modes: Optional[np.ndarray] = None,
                reference_direction: Optional[np.ndarray] = None,
                imag_freq_min: float = 100.0,
                imag_freq_max: float = 4000.0,
                small_freq_threshold: float = 50.0,
                tolerance_extra_imag: int = 0,
                overlap_threshold: float = 0.5) -> TSValidation:
    """
    验证给定 TS 的振动谱是否合法。

    Parameters
    ----------
    freqs_cm : (3*natm,) 频率，虚频以负值表示
    modes    : (3*natm, 3*natm) 笛卡尔正则模（可选，用于反应坐标检查）
    reference_direction : (natm, 3) 参考反应坐标方向（如 NEB tangent 或 dimer N）
    imag_freq_min/max   : 虚频幅度的合理范围 (cm⁻¹)
    small_freq_threshold: 低于此值视为平动转动（不计入"实频"）
    tolerance_extra_imag: 容忍的"额外"虚频数。0 = 严格 1 个；
                          1 = 允许 2 个虚频（例如 H 转移 TS 中常见的二级鞍点干扰）
    overlap_threshold   : 反应坐标重叠阈值

    Returns
    -------
    TSValidation
    """
    freqs = np.asarray(freqs_cm)
    warnings = []

    # 1. 计虚/实频
    mask_imag = freqs < -small_freq_threshold
    mask_real = freqs > small_freq_threshold

    n_imag = int(mask_imag.sum())
    real_freqs = np.sort(freqs[mask_real])

    # 2. 检查虚频数
    if n_imag == 0:
        warnings.append(f"无虚频（应有 1 个）—— 这是极小点而非 TS")
        return TSValidation(
            is_valid=False, n_imag=0,
            imag_freq_cm=None, real_freqs_cm=real_freqs,
            reaction_mode_overlap=None,
            warnings=warnings,
            summary="❌ 不是过渡态（无虚频）",
        )

    if n_imag > 1 + tolerance_extra_imag:
        warnings.append(f"{n_imag} 个虚频 > 容忍值 {1+tolerance_extra_imag} — 高阶鞍点")

    # 取最大幅度的虚频作为反应坐标
    imag_arr = -freqs[mask_imag]   # 取绝对值
    idx_max_in_imag = int(np.argmax(imag_arr))
    nu_imag = float(imag_arr[idx_max_in_imag])

    # 3. 虚频幅度检查
    if nu_imag < imag_freq_min:
        warnings.append(f"虚频 {nu_imag:.1f} cm⁻¹ < {imag_freq_min} - 可能数值噪声")
    if nu_imag > imag_freq_max:
        warnings.append(f"虚频 {nu_imag:.1f} cm⁻¹ > {imag_freq_max} - 量级异常")

    # 4. 反应坐标重叠检查（如果提供 modes 和 reference）
    overlap = None
    if modes is not None and reference_direction is not None:
        # 在所有 freqs 中找虚频对应列
        idx_in_full = int(np.where(mask_imag)[0][idx_max_in_imag])
        rxn_mode = modes[:, idx_in_full]   # (3*natm,)

        # reference 展平
        ref = np.asarray(reference_direction).ravel()
        if ref.size != rxn_mode.size:
            warnings.append(f"参考方向维度 {ref.size} 与模式维度 {rxn_mode.size} 不匹配")
        else:
            ref = ref / max(np.linalg.norm(ref), 1e-12)
            mode_norm = rxn_mode / max(np.linalg.norm(rxn_mode), 1e-12)
            overlap = float(abs(np.dot(ref, mode_norm)))   # 取绝对值

            if overlap < overlap_threshold:
                warnings.append(
                    f"反应坐标与参考方向重叠 {overlap:.3f} < {overlap_threshold}"
                    " - 可能找错鞍点"
                )

    # 综合判定
    is_valid = (
        n_imag >= 1 and
        n_imag <= 1 + tolerance_extra_imag and
        imag_freq_min <= nu_imag <= imag_freq_max
    )
    if overlap is not None:
        is_valid = is_valid and (overlap >= overlap_threshold)

    # 摘要
    if is_valid:
        summary = (f"✓ 有效 TS: 1 虚频 ν‡={nu_imag:.1f} cm⁻¹, "
                   f"{len(real_freqs)} 个实模"
                   + (f", 重叠={overlap:.3f}" if overlap is not None else ""))
    else:
        summary = f"❌ TS 验证失败 ({len(warnings)} 警告)"

    return TSValidation(
        is_valid=is_valid,
        n_imag=n_imag,
        imag_freq_cm=nu_imag,
        real_freqs_cm=real_freqs,
        reaction_mode_overlap=overlap,
        warnings=warnings,
        summary=summary,
    )


# ══════════════════════════════════════════════════════════════════════
# 一站式验证（含计算）
# ══════════════════════════════════════════════════════════════════════

def validate_ts_from_mf(mf, coords_bohr: np.ndarray,
                        reference_direction: Optional[np.ndarray] = None,
                        stepsize: float = 0.005,
                        **kwargs) -> tuple:
    """
    一站式：计算 Hessian + 验证。

    Returns
    -------
    (TSValidation, hessian_dict)
    """
    h = compute_ts_hessian(mf, coords_bohr, stepsize=stepsize, verbose=False)
    val = validate_ts(h['freqs_cm'], modes=h['modes'],
                      reference_direction=reference_direction, **kwargs)
    return val, h


# ══════════════════════════════════════════════════════════════════════
# 打印报告
# ══════════════════════════════════════════════════════════════════════

def print_ts_validation(val: TSValidation):
    """格式化打印"""
    print("=" * 60)
    print("  TS 验证报告")
    print("=" * 60)
    print(f"  整体判定: {val.summary}")
    print(f"  虚频数:    {val.n_imag}")
    if val.imag_freq_cm is not None:
        print(f"  ν‡:        {val.imag_freq_cm:.1f} cm⁻¹")
    if val.real_freqs_cm is not None and len(val.real_freqs_cm) > 0:
        rf = val.real_freqs_cm
        print(f"  实频范围:  {rf[0]:.1f} - {rf[-1]:.1f} cm⁻¹ ({len(rf)} 个)")
    if val.reaction_mode_overlap is not None:
        print(f"  反应坐标重叠: {val.reaction_mode_overlap:.3f}")
    if val.warnings:
        print(f"  警告 ({len(val.warnings)}):")
        for w in val.warnings:
            print(f"    ⚠️  {w}")
    print("=" * 60)
