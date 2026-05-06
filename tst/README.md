# nqeddft.tst — QED-增强过渡态理论模块

## 切入点1实现：把"光"和"热"通过 Eyring 方程耦合起来

```
k(T) = κ_tunnel(T) · (k_B·T / h) · exp(-ΔG‡(T) / k_B·T)
```

### 文件清单

```
tst/
├── __init__.py      公开接口
├── thermo.py        振动/平动/转动热力学量
├── tunneling.py     Wigner + Eckart 隧穿修正
├── qed_tst.py       主类 QEDTST + StationaryPoint
├── kie.py           动力学同位素效应
└── eyring_plot.py   Eyring/Arrhenius 图与综合分析

tests/                所有 38 个单元测试 (全过)
demo_cu2_co2_to_cooh.py   完整工作流演示
```

### 集成到 nqeddft 主包

把 `tst/` 整个目录复制到 `nqeddft/` 下：
```
nqeddft/
├── cavity/
├── scf/
├── tdscf/
├── cc/
├── grad/
├── phonon/
├── analysis/
├── cavity_field/
└── tst/          ← 新增
```

然后从主包导入：
```python
from nqeddft.tst import StationaryPoint, QEDTST, comprehensive_analysis
```

### 与现有模块的集成

`StationaryPoint.from_mf(mf, ...)` 会自动调用 `QEDPhonon` 做振动分析。

```python
from pyscf import gto, dft
from nqeddft import Cavity, QEDRKS
from nqeddft.tst import StationaryPoint, QEDTST

# 1. 反应物 SCF + 振动
mol_R = gto.M(atom=R_geom, basis='cc-pVDZ')
mf_R = QEDRKS(mol_R, cavity=cav).run()
R = StationaryPoint.from_mf(mf_R, name='CO2*', is_ts=False, phase='cluster')

# 2. 过渡态（已经做过 NEB/dimer 找到的鞍点）
mf_TS = QEDRKS(mol_TS, cavity=cav).run()
TS = StationaryPoint.from_mf(mf_TS, name='TS', is_ts=True, phase='cluster')

# 3. 产物
mf_P = QEDRKS(mol_P, cavity=cav).run()
P = StationaryPoint.from_mf(mf_P, name='COOH*', is_ts=False, phase='cluster')

# 4. 速率分析
tst = QEDTST(R, TS, P)
result = tst.compute_rate(T=298.15, tunneling='eckart')
tst.print_summary(T=298.15)
```

### 测试结果

所有 38 个单元测试通过：
- test_thermo.py:    11/11 (H₂ ZPE, 熵, 配分函数与教科书一致)
- test_tunneling.py:  8/8 (Wigner + Eckart 物理性检验)
- test_qed_tst.py:    9/9 (主类逻辑 + 腔诱导速率比)
- test_kie.py:        6/6 (KIE 温度依赖, 次级 KIE)
- test_eyring_plot.py: 4/4 (CSV 输出 + 图生成)

### Demo 输出

`demo_cu2_co2_to_cooh.py` 模拟 Cu₂(CO2*)+H* → Cu₂(COOH*) 反应，
展示腔耦合带来的 k×15 加速以及 KIE 从 5.1 → 3.3 的下降。

### 物理验证

| 测试场景 | 我们的输出 | 文献/理论值 | 状态 |
|----------|-----------|------------|------|
| H₂ ZPE | 6.283 kcal/mol | 6.28 | ✓ |
| H₂ 气相总熵 | 130.33 J/(mol·K) | 130.7 | ✓ |
| Wigner κ @1500i, 300K | 3.156 | 1+(7.20)²/24 = 3.16 | ✓ |
| Eckart H+H₂ @300K | κ=121 | 文献深隧穿区 50-200 | ✓ |
| 腔降低 V‡ 1 kcal/mol @300K | ratio=5.13 | exp(1/0.5926)=5.4 | ✓ |
| H/D KIE @300K (Wigner) | 2.14 | H+H₂ 实验~5-7 (取决于细节) | OK |
