# -*- coding: utf-8 -*-
"""
示例 04（修复版）：腔调制 IR 光谱

修复：CO2 必须先几何优化才能做振动分析
步骤：
  1. 几何优化（无腔，得到平衡结构）
  2. 在平衡结构上加腔，做 QED-DFT
  3. 振动分析（数值 Hessian）
  4. IR 强度和腔诱导频率移动
"""
import os
import numpy as np
from pyscf import gto
from pyscf.dft import rks as pyscf_rks
from nqeddft import Cavity, QEDRKS
from nqeddft.phonon import QEDPhonon

def safe_path(f):
    return f if os.access('.', os.W_OK) else os.path.join(os.environ.get('HOME','/tmp'), f)

# ── 步骤 1：先做无腔几何优化，找 CO2 平衡结构 ────────────────────────────
print("=" * 55)
print("步骤 1：CO2 几何优化（无腔）")
print("=" * 55)

# 初始几何（实验值，单位 Bohr）
mol_init = gto.M(
    atom="C 0 0 0; O 0 0 2.197; O 0 0 -2.197",
    basis="cc-pVDZ", unit="Bohr", verbose=0
)

try:
    from pyscf.geomopt import geometric_solver
    mf_opt = pyscf_rks.RKS(mol_init)
    mf_opt.xc = 'b3lyp'
    mf_opt.verbose = 0
    mf_opt.kernel()
    mol_eq = geometric_solver.optimize(mf_opt.Gradients())
    coords_eq = mol_eq.atom_coords()
    bond_CO = np.linalg.norm(coords_eq[1] - coords_eq[0])
    print(f"优化后 C-O 键长: {bond_CO:.6f} Bohr = {bond_CO*0.529177:.6f} Ang")
    mol = mol_eq
except ImportError:
    print("geometric_solver 不可用，使用初始几何（注意：可能有虚频！）")
    mol = mol_init

# ── 步骤 2：在平衡结构上设置腔 ──────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤 2：设置腔场")
print("=" * 55)

# CO2 反对称伸缩振动约 2349 cm-1 = 0.01070 a.u.
# 使用 B3LYP/cc-pVDZ 的计算值约 2380 cm-1 = 0.01084 a.u.
omega_CO2 = 2380.0 / 219474.6   # cm-1 → a.u.
cav = Cavity().add_mode(
    omega=omega_CO2,
    lambda_scalar=0.05,
    polarization=[0, 0, 1],
    name="CO2_asym_stretch"
)
print(cav.summary())

# ── 步骤 3：QED-DFT 基态 ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤 3：QED-DFT 基态")
print("=" * 55)
mf = QEDRKS(mol, cav)
mf.xc = 'b3lyp'
mf.verbose = 0
mf.kernel()
mf.print_qed_summary()

# ── 步骤 4：振动分析 ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤 4：数值 Hessian + 振动频率")
print("=" * 55)
ph = QEDPhonon(mf)

# 快速梯度差分 Hessian
hess = ph.numerical_hessian_fast(stepsize=0.001, verbose=True)
freqs, modes = ph.harmonic_analysis(hess)

# 检查虚频
n_imag = sum(1 for f in freqs if f < -50)
if n_imag > 0:
    print(f"\n警告：存在 {n_imag} 个虚频（< -50 cm-1）")
    print("CO2 是线形分子，应有 2 个零频（转动）和 4 个简并模")
    print("大虚频表明几何未收敛，请检查优化步骤")
else:
    print("\n无显著虚频，几何正常")

ph.print_frequencies(min_freq=50.0)

# ── 步骤 5：IR 强度 ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤 5：腔调制 IR 强度")
print("=" * 55)
ir = ph.ir_intensities(stepsize=0.001)
ph.print_ir_spectrum(min_freq=50.0)

# ── 步骤 6：腔诱导频率移动 ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤 6：腔诱导频率移动")
print("=" * 55)
shift_result = ph.cavity_shift(verbose=True)

# ── 步骤 7：振动-腔耦合强度 ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤 7：振动-腔耦合强度（振动 Rabi 劈裂）")
print("=" * 55)
ph.print_coupling_strength(min_freq=50.0)

# ── 保存结果 ─────────────────────────────────────────────────────────────
freqs_real = freqs[freqs > 50]
ir_real    = ir['ir_intensity'][freqs > 50]
out_file   = safe_path("co2_ir_cavity.csv")
np.savetxt(out_file,
           np.column_stack([freqs_real, ir_real]),
           delimiter=',', header='freq_cm,ir_intensity', comments='')
print(f"\nIR 数据已保存至 {out_file}")
