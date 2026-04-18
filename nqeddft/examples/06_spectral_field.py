# -*- coding: utf-8 -*-
"""
示例 06：具有频率分布的光场
============================

展示如何将不同谱型的连续光场离散化为多模腔，
并传入 QED-DFT 计算。

物理场景
--------
  A. 单模腔 + 洛伦兹线宽（有限 Q 值腔）
  B. 高斯非均匀展宽（固体中的腔模）
  C. 平顶宽带激光（泵浦宽带腔）
  D. 多峰谱（法布里-珀罗腔多个纵模）
  E. 自定义实验谱（从 CSV 导入）
  F. 收敛性分析（确定最优模式数 N）
"""
import numpy as np
from pyscf import gto
from nqeddft import (Cavity, QEDRKS, QEDTDA,
                      LorentzianField, GaussianField, FlatbandField,
                      MultiPeakField, CustomField,
                      cm1_to_au, au_to_cm1, au_to_ev)
from nqeddft.analysis import PolaritonAnalysis

# ── 分子：HF，cc-pVDZ ──────────────────────────────────────────────
mol = gto.M(atom="H 0 0 0; F 0 0 1.733", basis="cc-pVDZ",
            unit="Bohr", verbose=0)

# 先做无腔参考计算，确定激发能
from pyscf.tdscf import rks as _td_ref
from pyscf.dft   import rks as _rks
_mf = _rks.RKS(mol); _mf.xc='b3lyp'; _mf.verbose=0; _mf.kernel()
_td = _td_ref.TDA(_mf); _td.nstates=5; _td.kernel()
omega_target = float(_td.e[np.argmax(_td.oscillator_strength())])
print(f"目标激发能：{omega_target:.5f} a.u. = {au_to_ev(omega_target):.4f} eV\n")


# ════════════════════════════════════════════════════════════════
# 场景 A：洛伦兹线宽腔
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("A. 洛伦兹线宽腔（有限 Q 值，Q ≈ 20）")
print("=" * 60)

field_lor = LorentzianField(
    omega0       = omega_target,   # 共振中心
    gamma        = omega_target / 20.0,  # Q = ω₀/Γ = 20
    lambda_total = 0.05,
)
print(field_lor.summary())

# 离散化为 20 个模式
cav_lor = field_lor.to_cavity(N=20, polarization=[0, 0, 1])
print(f"\n离散化后：{cav_lor.n_modes} 个模式")
print(cav_lor.summary())

mf_lor = QEDRKS(mol, cav_lor); mf_lor.xc='b3lyp'; mf_lor.verbose=0
mf_lor.kernel()
td_lor = QEDTDA(mf_lor, cav_lor); td_lor.nstates=8; td_lor.kernel()
td_lor.print_spectrum()
PolaritonAnalysis(td_lor).compute().print_report()


# ════════════════════════════════════════════════════════════════
# 场景 B：高斯展宽
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("B. 高斯非均匀展宽（σ = Γ/2.355）")
print("=" * 60)

sigma_b = (omega_target / 20.0) / 2.355   # 与场景 A 等 FWHM
field_gau = GaussianField(
    omega0       = omega_target,
    sigma        = sigma_b,
    lambda_total = 0.05,
)
print(field_gau.summary())

cav_gau = field_gau.to_cavity(N=20, polarization=[0, 0, 1])
mf_gau  = QEDRKS(mol, cav_gau); mf_gau.xc='b3lyp'; mf_gau.verbose=0
mf_gau.kernel()
td_gau  = QEDTDA(mf_gau, cav_gau); td_gau.nstates=8; td_gau.kernel()
pol_gau = PolaritonAnalysis(td_gau).compute()
print(f"Rabi 劈裂（高斯）：{pol_gau.rabi_splitting*1000:.2f} meV")


# ════════════════════════════════════════════════════════════════
# 场景 C：平顶宽带激光（±20% 带宽）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("C. 平顶宽带激光（±20% 带宽）")
print("=" * 60)

bw = 0.2
field_flat = FlatbandField(
    omega_min    = omega_target * (1 - bw),
    omega_max    = omega_target * (1 + bw),
    lambda_total = 0.05,
)
print(field_flat.summary())

# 使用工厂方法简写
cav_flat = Cavity.from_flatband(
    omega_min    = omega_target * (1 - bw),
    omega_max    = omega_target * (1 + bw),
    lambda_total = 0.05,
    N            = 15,
    polarization = [0, 0, 1],
)
mf_flat = QEDRKS(mol, cav_flat); mf_flat.xc='b3lyp'; mf_flat.verbose=0
mf_flat.kernel()
td_flat = QEDTDA(mf_flat, cav_flat); td_flat.nstates=8; td_flat.kernel()
pol_flat = PolaritonAnalysis(td_flat).compute()
print(f"Rabi 劈裂（平顶）：{pol_flat.rabi_splitting*1000:.2f} meV")


# ════════════════════════════════════════════════════════════════
# 场景 D：多峰谱（两个纵模）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("D. 双纵模法布里-珀罗腔")
print("=" * 60)

FSR = omega_target * 0.08   # 自由光谱范围 = 8%

field_mp = MultiPeakField(
    peaks=[
        {'omega0': omega_target - FSR/2, 'width': omega_target/50,
         'weight': 1.0, 'shape': 'lorentzian'},
        {'omega0': omega_target + FSR/2, 'width': omega_target/50,
         'weight': 1.0, 'shape': 'lorentzian'},
    ],
    lambda_total = 0.05,
)
print(field_mp.summary())

cav_mp = field_mp.to_cavity(N=30, polarization=[0, 0, 1])
print(f"离散化后：{cav_mp.n_modes} 个模式")
mf_mp  = QEDRKS(mol, cav_mp); mf_mp.xc='b3lyp'; mf_mp.verbose=0
mf_mp.kernel()
td_mp  = QEDTDA(mf_mp, cav_mp); td_mp.nstates=10; td_mp.kernel()
td_mp.print_spectrum()


# ════════════════════════════════════════════════════════════════
# 场景 E：自定义谱（模拟实验数据）
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("E. 自定义实验谱（模拟非对称谱线）")
print("=" * 60)

# 构造一个非对称谱（如 Fano 线型）
nu_data  = np.linspace(au_to_cm1(omega_target) - 3000,
                        au_to_cm1(omega_target) + 3000, 500)
nu0      = au_to_cm1(omega_target)
gamma_cm = au_to_cm1(omega_target) / 20.0
q_fano   = 2.0   # Fano 不对称参数
eps      = (nu_data - nu0) / (gamma_cm / 2.0)
J_fano   = (eps + q_fano)**2 / (eps**2 + 1.0)   # Fano 线型
J_fano  /= J_fano.max()   # 归一化到最大值为 1

field_cust = CustomField(nu_data, J_fano, lambda_total=0.04, unit='cm1')
print(field_cust.summary())

cav_cust = field_cust.to_cavity(N=25, polarization=[0, 0, 1])
mf_cust  = QEDRKS(mol, cav_cust); mf_cust.xc='b3lyp'; mf_cust.verbose=0
mf_cust.kernel()
td_cust  = QEDTDA(mf_cust, cav_cust); td_cust.nstates=8; td_cust.kernel()
pol_cust = PolaritonAnalysis(td_cust).compute()
print(f"Rabi 劈裂（Fano 谱）：{pol_cust.rabi_splitting*1000:.2f} meV")


# ════════════════════════════════════════════════════════════════
# 场景 F：收敛性分析
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("F. 离散化收敛性分析（洛伦兹谱，宽腔 Q=5）")
print("=" * 60)

field_conv = LorentzianField(
    omega0       = omega_target,
    gamma        = omega_target / 5.0,   # 宽谱，需要更多模式
    lambda_total = 0.05,
)
N_opt, conv_info = field_conv.convergence_check(
    N_list       = [5, 10, 20, 40, 80],
    polarization = [0, 0, 1],
    tol          = 0.005,
)
print(f"\n推荐模式数：N = {N_opt}")


# ════════════════════════════════════════════════════════════════
# 汇总比较
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("汇总：不同谱型的 Rabi 劈裂比较（λ_total=0.05，ε=z）")
print("=" * 60)
print(f"{'谱型':>16}  {'Rabi/meV':>10}  {'混合度':>8}")
print("-" * 38)
for label, pol in [
    ("洛伦兹 Q=20",  pol_gau),   # 与高斯等 FWHM
    ("高斯 (σ≈Γ/2.355)", pol_gau),
    ("平顶 ±20%",   pol_flat),
    ("Fano 谱",     pol_cust),
]:
    print(f"{label:>16}  {pol.rabi_splitting*1000:>10.2f}  "
          f"{pol.max_mixing:>8.3f}")

print("\n完成！")
