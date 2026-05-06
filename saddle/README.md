# nqeddft.saddle — 腔感知的过渡态搜索

## 文件清单

```
saddle/
├── __init__.py        公开接口 (CINEB, Dimer, find_ts, validate_ts)
├── neb.py             Climbing-Image NEB (~430行)
├── dimer.py           Dimer 方法 + backtracking line search (~360行)
├── ts_validate.py     TS 验证 (Hessian + 反应坐标重叠)
├── interface.py       一站式 find_ts() API
└── test_saddle_core.py 12/12 单元测试通过
```

## 集成步骤

### 1. 复制目录

```bash
cp -r saddle/ ~/项目目录/nqeddft/
```

### 2. 在顶层 `__init__.py` 加导出（可选）

```python
# nqeddft/__init__.py
from .saddle import find_ts, CINEB, Dimer, validate_ts
```

### 3. 验证

```bash
cd ~/项目目录
python -c "from nqeddft.saddle import find_ts; print('OK')"
```

## 使用方式

### A. 一站式 API（推荐）

```python
from nqeddft import Cavity, QEDRKS, QEDUKS
from nqeddft.saddle import find_ts

# 准备反应物 mf 和产物 mf（已 SCF 收敛）
mf_R = ...  # CO2*+H* 反应物
mf_P = ...  # COOH* 产物

result = find_ts(
    mf_R, mf_P,
    n_neb_images=9,            # NEB images 数（含端点）
    neb_max_iter=30,
    neb_f_tol=5e-3,            # NEB 粗收敛
    dimer_max_iter=100,
    dimer_f_tol=5e-4,          # Dimer 严格收敛
    checkpoint_path='neb_path.json',  # 断点续算
)

if result.success:
    print(f"TS @ E = {result.energy_ts:.6f} Ha")
    print(f"ΔE‡ = {result.barrier_forward * 27.2114:.3f} eV")
    print(f"ν‡  = {result.imag_freq_cm:.1f} cm⁻¹")
    coords_ts = result.coords_ts   # (natm, 3) Bohr，TS 几何
```

### B. 分步用（更灵活）

```python
from nqeddft.saddle import CINEB, Dimer, validate_ts_from_mf
from nqeddft.saddle.interface import make_ef_fn_from_mf

# 1. 构造 ef_fn 包装器
ef_fn = make_ef_fn_from_mf(mf_R)   # 自动判断 RKS/UKS

# 2. NEB
neb = CINEB(ef_fn, mf_R.mol.atom_coords(), mf_P.mol.atom_coords(),
             n_images=11, k_spring=0.1, interpolation='idpp')
nr = neb.run(max_iter=30, f_tol=5e-3, climb_after=5)

# 保存路径（每个 image 的 coords/energy）
import json
with open('path.json', 'w') as f:
    json.dump(neb.export_path(), f)

# 3. Dimer 精化
from nqeddft.saddle.neb import _tangent
tangent = _tangent(nr.images, nr.ts_index)
dim = Dimer(ef_fn, nr.ts_image.coords, N_init=tangent, dR=0.005)
dr = dim.run(max_iter=100, f_tol=5e-4)

# 4. 验证 TS
val, hess_data = validate_ts_from_mf(
    mf_R, dr.coords,
    reference_direction=dr.direction,
    imag_freq_min=200, imag_freq_max=4000,
)
print(val.summary)   # ✓ 有效 TS: 1 虚频 ν‡=1245 cm⁻¹...
```

### C. 与 stage3 v3 集成的工作流

替换您现在 stage3 中"扫描+取最大点"的做法：

```python
# 旧 stage3：扫描势垒
# for R in R_arr:
#     E_arr.append(single_point(R))
# barrier = max(E_arr)

# 新做法：真实 NEB+Dimer
from nqeddft.saddle import find_ts

result = find_ts(mf_CO2H_reactant, mf_COOH_product, ...)
barrier = result.barrier_forward
ts_coords = result.coords_ts
ts_freqs = result.ts_validation.imag_freq_cm   # 真正的 ν‡

# 然后送入 QED-TST
from nqeddft.tst import StationaryPoint, QEDTST
TS = StationaryPoint(
    name='TS', e_elec=result.energy_ts,
    freqs_cm=hess_data['freqs_cm'],   # 完整振动谱
    is_ts=True, phase='cluster',
)
```

## 核心算法

### NEB (Climbing-Image)
- **改进型切线** (Henkelman 2000)：避免 corner-cutting
- **IDPP 插值** (Smidstrup 2014)：避开非物理化学键穿插
- **FIRE 优化器**：稳健且无需 Hessian
- **CI-NEB** 在第 N 步后启用：让最高能量 image"爬"到鞍点

### Dimer
- **两点 dimer** (Heyden 2005)：节省一半 SCF
- **二次插值旋转**：找最低不稳定模
- **Backtracking line search**：能量飙升时自动缩步
- **C_N 监控**：曲率为负时确认是鞍点区

### TS 验证
- 频率分类（实/虚/平动转动）
- 虚频量级合理性 (200-4000 cm⁻¹)
- 反应坐标重叠 (与参考方向 > 0.5)
- 多虚频警告

## 测试结果（解析势）

12/12 通过：

| 测试 | 结果 |
|---|---|
| IDPP 插值 | ✓ |
| 线性插值 | ✓ |
| NEB 双井 | TS @ x=0, E=1.0 |
| NEB Müller-Brown | TS_AB @ E=-40.66 |
| Dimer 双井 | C_N<0, 曲率正确 |
| Dimer Müller-Brown | 不发散，距 TS<0.5 |
| TS 验证 — 无虚频 | ✓ 拒绝 |
| TS 验证 — 1 虚频 | ✓ 通过 |
| TS 验证 — 2 虚频 | ✓ 拒绝(严格)/通过(宽松) |
| TS 验证 — 量级异常 | ✓ 拒绝 |
| TS 验证 — 重叠检查 | ✓ |
| **完整 pipeline (Müller-Brown)** | **TS_AB 距离 0.0032** ✓ |

## 参考文献

- Henkelman, Jónsson. JCP 113, 9978 (2000) — NEB 改进切线
- Henkelman, Uberuaga, Jónsson. JCP 113, 9901 (2000) — CI-NEB
- Smidstrup et al. JCP 140, 214106 (2014) — IDPP
- Henkelman, Jónsson. JCP 111, 7010 (1999) — Dimer 原版
- Heyden, Bell, Keil. JCP 123, 224101 (2005) — Dimer 改进
- Sheppard, Terrell, Henkelman. JCP 128, 134106 (2008) — improved tangent

## 时间预估（PBE0/def2-SVP）

对您 Cu_n+CO2*+H* → COOH* 的 NEB+Dimer 完整搜索：

| 体系 | NEB (9 image × 30 iter) | Dimer (50 iter) | 验证 (Hessian) | 总计 |
|---|---|---|---|---|
| Cu₁ (5原子) | ~30 min | ~10 min | ~5 min | **~45 min** |
| Cu₂ (6原子) | ~50 min | ~15 min | ~10 min | **~75 min** |
| Cu₄ (8原子) | ~2 h | ~30 min | ~20 min | **~3 h** |
| Cu₈ (12原子) | ~6 h | ~1.5 h | ~1 h | **~9 h** |

每个 cluster 单独跑一次（含 5 个 λ 时所有 NEB 共享 stage0 的端点几何）。
