# -*- coding: utf-8 -*-
"""
nqeddft.saddle — 过渡态搜索

提供腔感知的鞍点搜索：
  CINEB        : Climbing-Image NEB
  Dimer        : Dimer 方法（最低不稳定模上山）
  validate_ts  : 验证 TS 合法性（1 个虚频 + 反应坐标一致）
  find_ts      : 一站式 API: 反应物 mf + 产物 mf → TS

主要使用方式
------------

    >>> from nqeddft.saddle import find_ts
    >>> result = find_ts(mf_R, mf_P, n_neb_images=9, neb_max_iter=30)
    >>> if result.success:
    ...     print(f"TS @ E = {result.energy_ts}")
    ...     print(f"ν‡ = {result.imag_freq_cm} cm⁻¹")

或者分步用：

    >>> from nqeddft.saddle import CINEB, Dimer, validate_ts_from_mf
    >>> neb = CINEB(ef_fn, R_init, R_final, n_images=9)
    >>> nr = neb.run(max_iter=30)
    >>> dim = Dimer(ef_fn, nr.ts_image.coords, N_init=tangent)
    >>> dr = dim.run(max_iter=100)
    >>> val, h = validate_ts_from_mf(mf_R, dr.coords, reference_direction=dr.direction)
"""
from .neb        import CINEB, NEBImage, NEBResult, linear_interpolate, idpp_interpolate
from .dimer      import Dimer, DimerResult
from .ts_validate import (
    validate_ts, validate_ts_from_mf, compute_ts_hessian,
    TSValidation, print_ts_validation,
)
from .interface  import find_ts, SaddleResult, make_ef_fn_from_mf

__version__ = '0.1.0'

__all__ = [
    # NEB
    'CINEB', 'NEBImage', 'NEBResult', 'linear_interpolate', 'idpp_interpolate',
    # Dimer
    'Dimer', 'DimerResult',
    # 验证
    'validate_ts', 'validate_ts_from_mf', 'compute_ts_hessian',
    'TSValidation', 'print_ts_validation',
    # 顶层 API
    'find_ts', 'SaddleResult', 'make_ef_fn_from_mf',
]
