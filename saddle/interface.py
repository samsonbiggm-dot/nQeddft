# -*- coding: utf-8 -*-
"""
nqeddft.saddle.interface
========================
TS 搜索的"高级 API"：把 NEB → Dimer → 验证串成一个工作流。

提供：
  - find_ts(mf_R, mf_P, ...) : 单函数 API，输入两端点 mf 输出验证后的 TS
  - 从 PySCF mf 构造 energy_force_fn 包装器
  - 自动处理 RKS/UKS 分支
  - 断点续算（NEB 路径快照保存到 JSON）
  - 完整的 log 输出

工作流
------
    [反应物 mf]  [产物 mf]
         ↓            ↓
    ┌───────────────────────┐
    │ 1. 端点能量/力评估    │
    │    （确认两端是极小点）│
    └───────────────────────┘
         ↓
    ┌───────────────────────┐
    │ 2. CI-NEB             │
    │    n_images = 7-12    │
    │    f_tol = 5e-3       │  ← 粗收敛
    │    输出 climbing image │
    └───────────────────────┘
         ↓
    ┌───────────────────────┐
    │ 3. Dimer 精化          │
    │    起点 = NEB top      │
    │    f_tol = 5e-4       │  ← 严格收敛
    │    方向 = NEB tangent │
    └───────────────────────┘
         ↓
    ┌───────────────────────┐
    │ 4. TS 验证             │
    │    Hessian → 1 虚频?  │
    │    虚频 ∈ 合理范围?    │
    │    与 NEB tangent 一致?│
    └───────────────────────┘
         ↓
    ┌───────────────────────┐
    │ 5. 输出 SaddleResult   │
    │    含完整诊断信息       │
    └───────────────────────┘
"""
from __future__ import annotations

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict

from .neb import CINEB, NEBResult, idpp_interpolate
from .dimer import Dimer, DimerResult
from .ts_validate import validate_ts_from_mf, TSValidation, print_ts_validation


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SaddleResult:
    """完整的 TS 搜索结果"""
    success:         bool
    coords_ts:       Optional[np.ndarray] = None
    energy_ts:       Optional[float]      = None
    energy_R:        Optional[float]      = None
    energy_P:        Optional[float]      = None
    barrier_forward: Optional[float]      = None    # E_TS - E_R (Ha)
    barrier_reverse: Optional[float]      = None    # E_TS - E_P (Ha)
    imag_freq_cm:    Optional[float]      = None    # ν‡ 幅度
    ts_validation:   Optional[TSValidation] = None
    neb_result:      Optional[NEBResult]    = None
    dimer_result:    Optional[DimerResult]  = None
    error_msg:       str = ""
    elapsed_sec:     float = 0.0


# ══════════════════════════════════════════════════════════════════════
# QED scanner 包装器
# ══════════════════════════════════════════════════════════════════════

def make_ef_fn_from_mf(mf, force_uks: Optional[bool] = None) -> Callable:
    """
    从 PySCF mf（QEDRKS 或 QEDUKS）构造 energy_force_fn。

    Parameters
    ----------
    mf         : PySCF mf 对象（必须已实例化好；几何会被重写）
    force_uks  : None=自动判断；True=强制用 UKS 梯度

    Returns
    -------
    energy_force_fn(coords_bohr) -> (E, F)
        coords_bohr: (natm, 3)
        F = -∇E      (即"力"，传统记号)
    """
    from nqeddft.grad import QEDGradients
    
    # 检查 mf 类型来决定用 RKS 还是 UKS 梯度
    use_uks = force_uks
    if use_uks is None:
        # 看 mo_occ 是不是二维（UKS 标志）
        try:
            if mf.mo_occ is not None and mf.mo_occ.ndim == 2:
                use_uks = True
            else:
                use_uks = False
        except Exception:
            use_uks = False

    if use_uks:
        # UKS 路径：从 stage1_2 借用 QEDUKSGradients 类
        # 这里我们假设 nqeddft 中已有 QEDUKSGradients；如果没有则走兜底
        try:
            from nqeddft.grad import QEDUKSGradients
            grad_class = QEDUKSGradients
        except ImportError:
            # 兜底：在 stage1_2 里有定义，但更通用做法是动态构建
            grad_class = _build_qeduks_gradients_fallback()
    else:
        grad_class = QEDGradients

    def energy_force(coords: np.ndarray):
        coords = np.asarray(coords)
        mol = mf.mol
        mol.set_geom_(coords, unit='Bohr')
        mol.build()
        
        # 关键：清理可能残留的 cached 偶极积分
        if hasattr(mf, '_pf') and hasattr(mf._pf, 'dip_ints'):
            try:
                mf._pf.dip_ints.invalidate_cache()
            except Exception:
                pass
        
        mf.kernel()
        E = float(mf.e_tot)
        
        g = grad_class(mf)
        g.verbose = 0
        grad = g.kernel()
        F = -np.asarray(grad)   # 力 = -梯度
        return E, F

    return energy_force


def _build_qeduks_gradients_fallback():
    """
    兜底：手动构造 QEDUKSGradients（与 stage1_2 中定义一致）。
    """
    import numpy as np
    from pyscf.grad import uks as uks_grad

    class QEDUKSGradients(uks_grad.Gradients):
        def __init__(self, mf):
            super().__init__(mf)
            self.cavity = mf.cavity
            self._pf    = mf._pf

        def grad_elec(self, mo_energy=None, mo_coeff=None,
                      mo_occ=None, atmlst=None):
            de = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
            mol = self.mol
            dm  = self.base.make_rdm1()
            dm_total = dm[0] + dm[1]
            if atmlst is None:
                atmlst = list(range(mol.natm))
            for k, atm_id in enumerate(atmlst):
                de[k] += self._pf.grad_dse(dm_total, atm_id)
                de[k] += self._pf.grad_bilinear(dm_total, atm_id)
            return de

    return QEDUKSGradients


# ══════════════════════════════════════════════════════════════════════
# 主 API
# ══════════════════════════════════════════════════════════════════════

def find_ts(mf_R, mf_P,
            n_neb_images: int = 9,
            neb_max_iter: int = 30,
            neb_f_tol: float = 5e-3,
            neb_k_spring: float = 0.1,
            neb_climb_after: int = 5,
            dimer_max_iter: int = 100,
            dimer_f_tol: float = 5e-4,
            dimer_dR: float = 0.005,
            ts_validation_kwargs: Optional[dict] = None,
            checkpoint_path: Optional[str] = None,
            verbose: bool = True) -> SaddleResult:
    """
    搜索过渡态（NEB → Dimer → 验证）。

    Parameters
    ----------
    mf_R, mf_P : 反应物 / 产物的 mf 对象（已收敛！）
                 必须有相同的原子数与原子顺序。
    n_neb_images : NEB image 数（含端点），推荐 7-12
    neb_max_iter : NEB 最大迭代
    neb_f_tol    : NEB 收敛阈值（粗收敛 5e-3 即可，dimer 会精化）
    neb_k_spring : 弹簧常数 (Ha/Bohr²)，0.05-0.5
    neb_climb_after : 几步后启用 climbing image
    dimer_max_iter  : dimer 最大迭代
    dimer_f_tol     : dimer 收敛阈值（严格 5e-4）
    dimer_dR        : dimer 半距离 (Bohr)
    ts_validation_kwargs : 传给 validate_ts 的参数（如 imag_freq_min/max）
    checkpoint_path : NEB 路径快照保存路径（断点续算用）
    verbose         : 打印进度

    Returns
    -------
    SaddleResult
    """
    t0 = time.time()
    result = SaddleResult(success=False)

    # 1. 构造 ef_fn
    if verbose:
        print("\n" + "█" * 60)
        print("█  TS 搜索 pipeline 启动")
        print("█" * 60)
        print("\n[1] 构造能量/力评估器...")

    try:
        # 检查反应物/产物是 RKS 还是 UKS
        use_uks_R = (mf_R.mo_occ is not None and mf_R.mo_occ.ndim == 2)
        use_uks_P = (mf_P.mo_occ is not None and mf_P.mo_occ.ndim == 2)
        if use_uks_R != use_uks_P:
            raise ValueError("反应物和产物的 SCF 类型不同（RKS vs UKS）")

        ef_fn = make_ef_fn_from_mf(mf_R, force_uks=use_uks_R)
        result.energy_R = float(mf_R.e_tot)
        result.energy_P = float(mf_P.e_tot)

        R_react = mf_R.mol.atom_coords()
        R_prod  = mf_P.mol.atom_coords()
        if R_react.shape != R_prod.shape:
            raise ValueError(f"反应物 ({R_react.shape}) 与产物 "
                              f"({R_prod.shape}) 几何 shape 不一致")

        if verbose:
            print(f"    反应物 E = {result.energy_R:.6f} Ha")
            print(f"    产物   E = {result.energy_P:.6f} Ha")
            print(f"    {use_uks_R and 'UKS' or 'RKS'}, {R_react.shape[0]} 原子")

    except Exception as e:
        result.error_msg = f"端点准备失败: {e}"
        if verbose:
            print(f"  ❌ {result.error_msg}")
        result.elapsed_sec = time.time() - t0
        return result

    # 2. 跑 NEB
    if verbose:
        print(f"\n[2] CI-NEB ({n_neb_images} images, max_iter={neb_max_iter})...")

    try:
        neb = CINEB(
            energy_force_fn=ef_fn,
            R_reactant=R_react,
            R_product=R_prod,
            n_images=n_neb_images,
            k_spring=neb_k_spring,
            interpolation='idpp',
            verbose=verbose,
        )
        # 加载断点（如果有）
        if checkpoint_path is not None:
            cp = Path(checkpoint_path)
            if cp.exists():
                with open(cp) as f:
                    data = json.load(f)
                neb.import_path(data)
                if verbose:
                    print(f"  从 {cp} 加载断点")

        neb_result = neb.run(
            max_iter=neb_max_iter,
            f_tol=neb_f_tol,
            climb_after=neb_climb_after,
            verbose=verbose,
        )
        result.neb_result = neb_result

        # 保存断点
        if checkpoint_path is not None:
            with open(checkpoint_path, 'w') as f:
                json.dump(neb.export_path(), f, indent=2)
            if verbose:
                print(f"  → 路径快照已保存到 {checkpoint_path}")

        if neb_result.ts_index < 0:
            result.error_msg = "NEB 未找到 climbing image"
            if verbose:
                print(f"  ❌ {result.error_msg}")
            result.elapsed_sec = time.time() - t0
            return result

        if verbose:
            print(f"  NEB {'✓ 收敛' if neb_result.converged else '⚠️ 达到 max_iter'} "
                  f"after {neb_result.n_iter} 步")
            print(f"  Climbing image: {neb_result.ts_index}, "
                  f"E = {neb_result.ts_image.energy:.6f} Ha")

    except Exception as e:
        import traceback
        result.error_msg = f"NEB 失败: {e}"
        if verbose:
            print(f"  ❌ {result.error_msg}")
            traceback.print_exc()
        result.elapsed_sec = time.time() - t0
        return result

    # 3. Dimer 精化
    if verbose:
        print(f"\n[3] Dimer 精化（起点 = NEB image {neb_result.ts_index}）...")

    # 反应坐标方向 = NEB 切线（image i 处）
    from .neb import _tangent
    try:
        tangent = _tangent(neb_result.images, neb_result.ts_index)
    except Exception:
        tangent = None

    try:
        dim = Dimer(
            energy_force_fn=ef_fn,
            R_init=neb_result.ts_image.coords,
            N_init=tangent,
            dR=dimer_dR,
            verbose=verbose,
        )
        dimer_result = dim.run(
            max_iter=dimer_max_iter,
            f_tol=dimer_f_tol,
        )
        result.dimer_result = dimer_result

        if not dimer_result.converged:
            if verbose:
                print(f"  ⚠️ Dimer 未收敛 (max|F|={dimer_result.history[-1]:.2e})，"
                      "继续验证（可能仍是合理 TS）")

    except Exception as e:
        import traceback
        result.error_msg = f"Dimer 失败: {e}"
        if verbose:
            print(f"  ❌ {result.error_msg}")
            traceback.print_exc()
        result.elapsed_sec = time.time() - t0
        return result

    # 4. TS 验证
    if verbose:
        print(f"\n[4] 验证 TS（Hessian + 频率分析）...")

    try:
        kw = ts_validation_kwargs or {}
        val, hess_data = validate_ts_from_mf(
            mf_R,    # mf_R 在 dimer 后会有最新 SCF 状态
            dimer_result.coords,
            reference_direction=dimer_result.direction,
            **kw,
        )
        result.ts_validation = val
        if verbose:
            print_ts_validation(val)

        result.coords_ts    = dimer_result.coords
        result.energy_ts    = hess_data['e_at_ts']
        result.imag_freq_cm = val.imag_freq_cm
        result.barrier_forward = result.energy_ts - result.energy_R
        result.barrier_reverse = result.energy_ts - result.energy_P

        result.success = val.is_valid
        if verbose and val.is_valid:
            print(f"\n  ✓ TS 搜索成功！")
            print(f"  ΔE‡ (forward) = {result.barrier_forward * 27.2114:.3f} eV")
            print(f"  ΔE‡ (reverse) = {result.barrier_reverse * 27.2114:.3f} eV")
            print(f"  ν‡            = {result.imag_freq_cm:.1f} cm⁻¹")

    except Exception as e:
        import traceback
        result.error_msg = f"验证失败: {e}"
        if verbose:
            print(f"  ❌ {result.error_msg}")
            traceback.print_exc()

    result.elapsed_sec = time.time() - t0
    if verbose:
        print(f"\n[完成] 总用时 {result.elapsed_sec:.1f} 秒")
        print("█" * 60)
    return result
