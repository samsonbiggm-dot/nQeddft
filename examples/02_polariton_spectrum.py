# -*- coding: utf-8 -*-
"""
示例 02（v3，修复版）：极化子谱计算

关键物理要点：
  极化子混合需要两个条件同时满足：
  1. 腔频率 ≈ 激子激发能（频率共振）
  2. 腔极化方向 ∥ 跃迁偶极矩方向（空间匹配）

HF 分子沿 z 轴：
  第一激发态 (9.86 eV)：Σ→Π 跃迁，偶极矩沿 x/y 轴 → 需 x/y 极化腔
  第四激发态 (14.76 eV)：Σ→Σ 跃迁，偶极矩沿 z 轴 → 需 z 极化腔
"""
import os
import numpy as np
from pyscf import gto
from pyscf.dft import rks as pyscf_rks
from pyscf.tdscf import rhf as td_pyscf
from nqeddft import Cavity, QEDRKS, QEDTDA
from nqeddft.analysis import PolaritonAnalysis, AbsorptionSpectrum

def safe_path(f):
    return f if os.access('.', os.W_OK) else \
           os.path.join(os.environ.get('HOME', '/tmp'), f)

# ── 分子定义 ─────────────────────────────────────────────────────────────
mol = gto.M(atom="H 0 0 0; F 0 0 1.733", basis="cc-pVDZ",
            unit="Bohr", verbose=0)

# ── 步骤 1：分析所有激发态，找出哪个与 z 极化腔耦合 ─────────────────────
print("=" * 60)
print("步骤 1：分析激发态特性（能量 + 振子强度）")
print("=" * 60)
mf_ref = pyscf_rks.RKS(mol)
mf_ref.xc = 'b3lyp'; mf_ref.verbose = 0; mf_ref.kernel()

td_ref = td_pyscf.TDA(mf_ref)
td_ref.nstates = 5; td_ref.kernel()

print(f"{'态':>4}  {'E/eV':>9}  {'E/a.u.':>10}  {'振子强度':>10}  说明")
print("-" * 60)
for i, (e, f) in enumerate(zip(td_ref.e, td_ref.oscillator_strength())):
    e_ev = e * 27.2114
    note = "强 z 偶极（Σ→Σ）" if f > 0.05 else "弱/x-y 偶极（Σ→Π）"
    print(f"{i+1:>4}  {e_ev:>9.4f}  {e:>10.6f}  {f:>10.6f}  {note}")

# 选出振子强度最大的态（最强偶极矩，最容易与 z 腔耦合）
f_arr    = np.array(td_ref.oscillator_strength())
best_idx = int(np.argmax(f_arr))
e_best   = float(td_ref.e[best_idx])
f_best   = float(f_arr[best_idx])
print(f"\n→ 选择态 {best_idx+1}（f={f_best:.4f}，最强 z 方向偶极矩）作为共振目标")
print(f"  目标激发能：{e_best:.6f} a.u. = {e_best*27.2114:.4f} eV")

# ── 步骤 2：方案 A — z 极化腔 + 与最强 z 偶极态共振 ─────────────────────
print("\n" + "=" * 60)
print("步骤 2A：z 极化腔，与最强激发态共振")
print("=" * 60)

cav_z = Cavity().add_mode(
    omega=e_best, lambda_scalar=0.05,
    polarization=[0, 0, 1], name='z_resonant'
)
print(cav_z.summary())

mf_z = QEDRKS(mol, cav_z); mf_z.xc='b3lyp'; mf_z.verbose=0; mf_z.kernel()
td_z = QEDTDA(mf_z, cav_z); td_z.nstates=8; td_z.kernel()
td_z.print_spectrum()

pol_z = PolaritonAnalysis(td_z).compute()
pol_z.print_report()

# ── 步骤 3：方案 B — x 极化腔 + 与第一激发态（Π 态）共振 ──────────────
print("\n" + "=" * 60)
print("步骤 2B：x 极化腔，与第一激发态（Σ→Π）共振")
print("=" * 60)

e_first = float(td_ref.e[0])
cav_x = Cavity().add_mode(
    omega=e_first, lambda_scalar=0.05,
    polarization=[1, 0, 0], name='x_resonant'   # x 极化！
)
print(cav_x.summary())

mf_x = QEDRKS(mol, cav_x); mf_x.xc='b3lyp'; mf_x.verbose=0; mf_x.kernel()
td_x = QEDTDA(mf_x, cav_x); td_x.nstates=8; td_x.kernel()
td_x.print_spectrum()

pol_x = PolaritonAnalysis(td_x).compute()
pol_x.print_report()

# ── 步骤 4：保存共振方案的光谱 ──────────────────────────────────────────
# 选混合度更高的方案
best_pol = pol_z if pol_z.max_mixing >= pol_x.max_mixing else pol_x
best_td  = td_z  if pol_z.max_mixing >= pol_x.max_mixing else td_x
label    = "z" if pol_z.max_mixing >= pol_x.max_mixing else "x"
print(f"\n→ 方案 {label.upper()} 混合度更高（{best_pol.max_mixing:.3f}），保存该光谱")
AbsorptionSpectrum(best_td).save_csv(
    safe_path(f"hf_polariton_{label}.csv"), fwhm=0.02)

# ── 步骤 5：共振条件下 Rabi 劈裂 vs 耦合强度 ─────────────────────────────
# 使用混合度最高的方案
best_omega = e_best if pol_z.max_mixing >= pol_x.max_mixing else e_first
best_pol_vec = [0,0,1] if pol_z.max_mixing >= pol_x.max_mixing else [1,0,0]

print("\n" + "=" * 60)
print(f"步骤 3：Rabi 劈裂 vs 耦合强度（极化方向={best_pol_vec}）")
print("=" * 60)
print(f"{'lambda':>8}  {'混合度':>7}  {'Rabi/meV':>10}  {'LP/eV':>8}  {'UP/eV':>8}  {'状态':>6}")
print("-" * 58)

for lam in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    ci  = Cavity().add_mode(best_omega, lam, best_pol_vec)
    mfi = QEDRKS(mol, ci); mfi.xc='b3lyp'; mfi.verbose=0; mfi.kernel()
    tdi = QEDTDA(mfi, ci); tdi.nstates=6; tdi.kernel()
    pi  = PolaritonAnalysis(tdi).compute()
    status = "✓共振" if pi.is_resonant() else "✗失谐"
    print(f"{lam:>8.3f}  {pi.max_mixing:>7.3f}  "
          f"{pi.rabi_splitting*1000:>10.2f}  "
          f"{pi.lp_energy:>8.4f}  {pi.up_energy:>8.4f}  {status:>6}")

print("\n完成！物理规律：真共振时 Rabi 劈裂应正比于 lambda")
