# nqeddft — QED-DFT Software Package

基于 PySCF 框架的量子电动力学密度泛函理论（QED-DFT）软件包，
实现光子-电子耦合、光子-声子耦合的从头算计算。

## 安装

```bash
pip install pyscf numpy scipy
pip install -e .
```

## 快速开始

```python
from pyscf import gto
from nqeddft import Cavity, QEDRKS, QEDTDA

# 定义分子
mol = gto.M(atom="H 0 0 0; F 0 0 1.733", basis="cc-pVDZ",
            unit="Bohr", verbose=3)

# 定义腔场（单模，近红外，中等耦合）
cav = Cavity().add_mode(omega=0.1, lambda_scalar=0.05,
                         polarization=[0, 0, 1])

# QED-DFT 基态
mf = QEDRKS(mol, cav)
mf.xc = "b3lyp"
e = mf.kernel()
mf.print_qed_summary()

# QED-TDDFT 极化子谱
td = QEDTDA(mf, cav)
td.nstates = 8
td.kernel()
td.print_spectrum()
```

## 模块结构

```
nqeddft/
├── cavity.py              腔场参数（Cavity, CavityMode）
├── integrals/
│   └── dipole.py          偶极矩积分封装（DipoleIntegrals）
├── scf/
│   ├── pauli_fierz.py     Pauli-Fierz Fock 修正
│   ├── qed_rks.py         QED-RKS（继承 pyscf.dft.RKS）
│   ├── qed_rhf.py         QED-RHF（继承 pyscf.scf.RHF）
│   └── qed_uks.py         QED-UKS（继承 pyscf.dft.UKS）
├── tdscf/
│   └── qed_tddft.py       QED-TDA/TDDFT（继承 pyscf.tdscf）
├── cc/
│   └── qed_ccsd.py        QED-CCSD（继承 pyscf.cc.CCSD）
├── grad/
│   └── qed_grad.py        QED 核梯度（继承 pyscf.grad.rks）
├── phonon/
│   └── qed_phonon.py      光子-声子耦合（振动分析）
├── analysis/
│   ├── polariton.py       极化子分析（Hopfield 系数、Rabi 劈裂）
│   └── spectrum.py        吸收光谱生成
├── validation/
│   └── checks.py          物理正确性验证（三个黄金测试）
├── tests/                 pytest 测试套件
└── examples/              示例脚本
```

## 物理实现

### Pauli-Fierz Hamiltonian（偶极规范）
```
H_PF = H_el + Σ_α[ ω_α(a†a+½) - √(ω/2)(λ⃗·d̂)(a†+a) + ½(λ⃗·d̂)² ]
```

### Fock 矩阵修正（get_veff Hook）
- `V_dse_J  = <λ·d>·(λ·d)_μν`           DSE Coulomb-like
- `V_dse_K  = (λ·d·P·λ·d) exchange`      DSE Exchange-like
- `V_bilinear = -√(ω/2)·<a†+a>·(λ·d)`   双线性耦合均场

### QED-TDDFT（极化子谱）
扩展 Casida 矩阵加入光子-激子耦合块，
`G_ia = √(ω/2)·<i|λ⃗·d̂|a>`，求解上/下极化子本征态。

## 验证

```bash
# 运行全部测试（不含慢速数值梯度测试）
pytest tests/ -v

# 运行含数值梯度验证的完整测试
pytest tests/ -v -m "slow or not slow"
```

## 开发路线

| 阶段 | 状态 | 内容 |
|------|------|------|
| Phase 1 | ✓ 已实现 | QED-RKS/RHF/UKS、TDA、CCSD(pt2)、核梯度、振动分析 |
| Phase 2 | 计划中 | 完整 velocity-gauge 验证、OEP 光子 XC 泛函 |
| Phase 3 | 计划中 | 完整 QED-CCSD 振幅方程（Haugland 2020） |
| Phase 4 | 计划中 | Holstein 光子-声子全量子耦合（DMRG 接口） |
| Phase 5 | 计划中 | GPU 加速（gpu4pyscf 后端） |

## 参考文献

1. Ruggenthaler et al., *Phys. Rev. A* **90**, 012508 (2014)
2. Flick et al., *PNAS* **114**, 3026 (2017)
3. Haugland et al., *Phys. Rev. Lett.* **125**, 233101 (2020)
4. Lindoy, Mandal, Reichman, *J. Phys. Chem. Lett.* **14**, 2451 (2023)
