# -*- coding: utf-8 -*-
"""
nqeddft.tdscf.qed_tddft
=======================
QED-TDDFT 线性响应（极化子谱）

实现两个层次：

  QEDTDA（Tamm-Dancoff 近似）
  ─────────────────────────
  扩展 TDA 矩阵（仅激发块 A）：

      [A_el   G  ] [X ]       [X ]
      [G^T   W_ph] [q ] = E * [q ]

      A_el  : 电子空穴激发矩阵（PySCF 原版 TDA）
      G_ia^α = √(ω_α/2) · <i|λ_α·d̂|a>    光子–激子耦合
      W_ph   = diag(ω_α)                   光子频率

  QEDTDDFT（完整线性响应）
  ──────────────────────
  扩展 Casida 矩阵（含退激发 B 块）。

  Furche (2001) Cholesky 变换将问题化为对称正定本征值问题：
    令 M = A−B，Cholesky 分解 M = L Lᵀ，则
      [L⁻ᵀ(A+B)L⁻¹  L⁻ᵀG ] [Lᵀ(X+Y)]         [Lᵀ(X+Y)]
      [GᵀL⁻¹         W_ph ] [  q     ] = E² *  [  q     ]

  光子自由度无退激发对应，仅出现在激发列。

  两类均支持 RKS（闭壳层）和 UKS（开壳层，α/β 轨道拼接处理）。

修复记录：
  v1: 初始实现（TDA only，RKS only，B 矩阵占位符）
  v2: 实现完整 B 矩阵（Cholesky 变换）；新增 UKS 支持；
      新增 transition_dipole / absorption_cross_section；
      修复 oscillator_strength 规范依赖；
      QEDTDDFT.tda_comparison 诊断工具
"""
import numpy as np
import warnings
from pyscf import lib
from ..cavity import Cavity
from ..integrals.dipole import DipoleIntegrals


# ══════════════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════════════

def _is_uks(mf) -> bool:
    """判断 mf 是否为开壳层（QEDUKS 或 pyscf UKS）。"""
    from ..scf.qed_uks import QEDUKS
    return isinstance(mf, QEDUKS) or (
        hasattr(mf, 'mo_occ') and np.asarray(mf.mo_occ).ndim == 2
    )


def _mo_split(mf):
    """
    返回 (mo_o, mo_v, e_occ, e_vir, nocc, nvir)。

    RKS：单组轨道，按 mo_occ > 0 划分。
    UKS：将 α/β 轨道拼接（spin-averaged 处理），适用于极化子谱。
    """
    mo_c   = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    mo_e   = np.asarray(mf.mo_energy)

    if mo_c.ndim == 3:
        # UKS：shape (2, nao, nmo)
        occ_a = mo_occ[0] > 0;  vir_a = mo_occ[0] == 0
        occ_b = mo_occ[1] > 0;  vir_b = mo_occ[1] == 0
        mo_o  = np.concatenate([mo_c[0][:,occ_a], mo_c[1][:,occ_b]], axis=1)
        mo_v  = np.concatenate([mo_c[0][:,vir_a], mo_c[1][:,vir_b]], axis=1)
        e_occ = np.concatenate([mo_e[0][occ_a],   mo_e[1][occ_b]])
        e_vir = np.concatenate([mo_e[0][vir_a],   mo_e[1][vir_b]])
    else:
        occ = mo_occ > 0;  vir = mo_occ == 0
        mo_o  = mo_c[:, occ];  mo_v  = mo_c[:, vir]
        e_occ = mo_e[occ];     e_vir = mo_e[vir]

    return mo_o, mo_v, e_occ, e_vir, mo_o.shape[1], mo_v.shape[1]


def _build_g(mf, cavity, dip_ints):
    """
    G_{ia}^α = √(ω_α/2) · <i|λ_α·d̂|a>，返回列表，每元素 shape (nocc,nvir)。
    """
    mo_o, mo_v, _, _, _, _ = _mo_split(mf)
    dip_ao = dip_ints.dip_ao   # (3,nao,nao)
    g_list = []
    for mode in cavity.modes:
        ld = np.einsum('k,kpq->pq', mode.lambda_vec, dip_ao)
        g_list.append(mode.sqrt_omega_half * (mo_o.T @ ld @ mo_v))
    return g_list


# ══════════════════════════════════════════════════════════════════════
# QEDTDA
# ══════════════════════════════════════════════════════════════════════

class QEDTDA:
    """
    QED Tamm-Dancoff Approximation。
    支持 QEDRKS（闭壳层）和 QEDUKS（开壳层）基态。
    """

    def __init__(self, mf, cavity: Cavity):
        self._scf      = mf
        self.cavity    = cavity
        self.dip_ints  = DipoleIntegrals(mf.mol)
        self.verbose   = mf.verbose
        self.nstates   = 5
        self.conv_tol  = 1e-6
        self.max_cycle = 100

        self.e              = None
        self.xy             = None
        self.photon_weight  = None
        self.exciton_weight = None

        self._mo_o   = None
        self._mo_v   = None
        self._e_occ  = None
        self._e_vir  = None
        self._nocc   = 0
        self._nvir   = 0
        self._g_list = None

    # ------------------------------------------------------------------

    def _build(self):
        (self._mo_o, self._mo_v,
         self._e_occ, self._e_vir,
         self._nocc, self._nvir) = _mo_split(self._scf)
        self._g_list = _build_g(self._scf, self.cavity, self.dip_ints)

    def _electron_vind(self):
        """返回 A·X 的矩阵向量乘积函数，自动选择 RKS/UKS。"""
        mf = self._scf
        if _is_uks(mf):
            from pyscf.tdscf import uks as _td
        else:
            from pyscf.tdscf import rks as _td
        td_ref = _td.TDA(mf); td_ref.verbose = 0
        vind, _ = td_ref.gen_vind(mf)
        return vind

    def _vind_tda(self):
        nocc   = self._nocc; nvir = self._nvir
        n_ia   = nocc * nvir; n_ph = self.cavity.n_modes
        g_list = self._g_list; modes = self.cavity.modes
        ve     = self._electron_vind()

        def vind(zs):
            res = []
            for z in zs:
                z    = np.asarray(z, dtype=float)
                z_ia = z[:n_ia].reshape(nocc, nvir)
                z_ph = z[n_ia:n_ia+n_ph] if len(z) > n_ia else np.zeros(n_ph)
                ax   = ve([z_ia.ravel()])[0].reshape(nocc, nvir)
                for a, g in enumerate(g_list):
                    ax += g * z_ph[a]
                bq = np.array([
                    modes[a].omega * z_ph[a]
                    + float(np.einsum('ia,ia->', g_list[a], z_ia))
                    for a in range(n_ph)
                ])
                res.append(np.concatenate([ax.ravel(), bq]))
            return res

        return vind

    def _diag(self):
        e_ia = (self._e_vir[None,:] - self._e_occ[:,None]).ravel()
        return np.concatenate([e_ia,
                                [m.omega for m in self.cavity.modes]])

    def _solve(self, vind, diag, ns):
        n = len(diag)

        def precond(dx, e, _):
            d = diag - e
            return dx / np.where(np.abs(d) > 1e-6, d, 1e-6)

        x0 = np.zeros((ns, n))
        for k, i in enumerate(np.argsort(diag)[:ns]):
            x0[k, i] = 1.0

        kw = dict(tol=self.conv_tol, nroots=ns,
                  max_space=max(ns*8, 20),
                  max_cycle=self.max_cycle, verbose=self.verbose)
        try:
            _, e, v = lib.davidson1(vind, x0, precond, **kw)
        except AttributeError:
            _, e, v = lib.davidson(vind, x0, diag, **kw)
        return np.asarray(e), np.array(v)

    def kernel(self, x0=None, nstates=None):
        if nstates is not None:
            self.nstates = nstates
        self._build()
        e, v = self._solve(self._vind_tda(), self._diag(), self.nstates)
        self.e  = e
        self.xy = list(v) if v.ndim == 2 else [v]
        self._analyze_weights(self.xy)
        return self.e, self.xy

    # ------------------------------------------------------------------
    # 分析
    # ------------------------------------------------------------------

    def _analyze_weights(self, vecs):
        n_ia = self._nocc * self._nvir
        self.photon_weight  = []
        self.exciton_weight = []
        for v in vecs:
            v = np.asarray(v, dtype=float)
            n = float(np.dot(v, v)) or 1.0
            self.photon_weight.append(float(np.dot(v[n_ia:], v[n_ia:])) / n)
            self.exciton_weight.append(float(np.dot(v[:n_ia], v[:n_ia])) / n)

    def transition_dipole(self) -> np.ndarray:
        """各极化子态的跃迁偶极矩 <0|d̂|k>，shape (nstates, 3)，a.u.。"""
        if self.e is None:
            raise RuntimeError("请先调用 kernel()")
        n_ia   = self._nocc * self._nvir
        dip_mo = np.einsum('kpq,pi,qa->kia',
                           self.dip_ints.dip_ao, self._mo_o, self._mo_v)
        td = np.zeros((len(self.e), 3))
        for n, v in enumerate(self.xy):
            x_ia    = np.asarray(v)[:n_ia].reshape(self._nocc, self._nvir)
            td[n]   = np.einsum('kia,ia->k', dip_mo, x_ia)
        return td

    def oscillator_strength(self) -> np.ndarray:
        """f_k = (2/3) E_k |<0|d̂|k>|²，shape (nstates,)。"""
        if self.e is None:
            raise RuntimeError("请先调用 kernel()")
        td = self.transition_dipole()
        return np.array([
            max(0.0, (2.0/3.0) * float(E) * float(np.dot(mu, mu)))
            for E, mu in zip(self.e, td)
        ])

    def absorption_cross_section(self, e_range=None, n_pts=2000,
                                  fwhm=0.05, lineshape='lorentzian') -> dict:
        """生成吸收截面光谱（Lorentzian 或 Gaussian 展宽）。"""
        if self.e is None:
            raise RuntimeError("请先调用 kernel()")
        e_ev  = self.e * 27.2114
        f     = self.oscillator_strength()
        gamma = fwhm / 2.0
        if e_range is None:
            e_range = (max(0.0, e_ev.min()-0.5), e_ev.max()+0.5)
        grid  = np.linspace(e_range[0], e_range[1], n_pts)
        sigma = np.zeros(n_pts)
        for E_k, f_k in zip(e_ev, f):
            if lineshape == 'lorentzian':
                sigma += f_k * gamma / (np.pi * ((grid-E_k)**2 + gamma**2))
            else:
                s = fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))
                sigma += f_k*np.exp(-0.5*((grid-E_k)/s)**2)/(s*np.sqrt(2.0*np.pi))
        return {'energy_ev': grid, 'sigma': sigma,
                'peaks_ev': e_ev, 'osc_strength': f}

    def print_spectrum(self):
        if self.e is None:
            raise RuntimeError("请先调用 kernel()")
        f = self.oscillator_strength()
        print("=" * 68)
        print(f"  {'态':>3} {'E/eV':>9} {'E/a.u.':>10} "
              f"{'振子强度':>12} {'光子%':>8} {'激子%':>8}")
        print("-" * 68)
        for i, E in enumerate(self.e):
            print(f"  {i+1:>3} {E*27.2114:>9.4f} {E:>10.6f} "
                  f"{f[i]:>12.6f} "
                  f"{self.photon_weight[i]*100:>7.2f}% "
                  f"{self.exciton_weight[i]*100:>7.2f}%")
        print("=" * 68)


# ══════════════════════════════════════════════════════════════════════
# QEDTDDFT  —  完整线性响应（含 B 矩阵）
# ══════════════════════════════════════════════════════════════════════

class QEDTDDFT(QEDTDA):
    """
    完整 QED-TDDFT（含退激发 B 矩阵）。

    使用 Furche (2001) Cholesky 变换将扩展 Casida 方程化为对称正定问题：
      令 (A−B) = L Lᵀ，则求解：

        [L⁻ᵀ(A+B)L⁻¹  L⁻ᵀG ] [Lᵀ(X+Y)]         [Lᵀ(X+Y)]
        [GᵀL⁻¹         W_ph ] [  q     ] = E² *  [  q     ]

    若 (A−B) 近奇异（存在近简并激发态），自动降级为 TDA 并发出警告。

    支持 RKS 和 UKS，接口与 QEDTDA 完全兼容。
    """

    def __init__(self, mf, cavity: Cavity):
        super().__init__(mf, cavity)
        self._fallback_to_tda = False

    # ------------------------------------------------------------------
    # 获取 (A+B) 和 (A−B) 的矩阵向量乘积
    # ------------------------------------------------------------------

    def _get_apb_amb(self):
        """
        返回 (vind_apb, vind_amb)。

        PySCF TDDFT.gen_vind 对于 full TDDFT（TDA=False）返回两个函数：
          vind_apb(X) = (A+B)X   （激发响应）
          vind_amb(X) = (A−B)X   （退激发响应）
        若 PySCF 版本只返回一个函数，则降级为 TDA。
        """
        mf = self._scf
        if _is_uks(mf):
            from pyscf.tdscf import uks as _td
        else:
            from pyscf.tdscf import rks as _td

        td_ref = _td.TDDFT(mf); td_ref.verbose = 0
        result = td_ref.gen_vind(mf)
        if isinstance(result, tuple) and len(result) == 2:
            return result   # (vind_apb, vind_amb)
        else:
            warnings.warn(
                "当前 PySCF 版本的 TDDFT.gen_vind 只返回单个函数，"
                "无法获取独立的 (A+B)/(A−B) 接口，降级为 TDA。",
                UserWarning, stacklevel=3,
            )
            self._fallback_to_tda = True
            return result, result   # 占位，不会被使用

    # ------------------------------------------------------------------
    # 构建并 Cholesky 分解 (A−B)
    # ------------------------------------------------------------------

    def _build_amb(self, vind_amb):
        """
        显式构建 (A−B) 矩阵，shape (n_ia, n_ia)。
        通过对单位基向量施加 vind_amb 逐列填充。
        """
        n_ia = self._nocc * self._nvir
        amb  = np.zeros((n_ia, n_ia))
        e_j  = np.zeros(n_ia)
        for j in range(n_ia):
            e_j[:] = 0.0; e_j[j] = 1.0
            col = vind_amb([e_j])[0]
            amb[:, j] = col[:n_ia]
        return 0.5 * (amb + amb.T)   # 对称化消除数值误差

    def _cholesky(self, amb):
        """
        Cholesky 分解 M = L Lᵀ。
        若最小特征值 < 1e-6 则返回 (None, None) 并发出警告。
        """
        emin = float(np.linalg.eigvalsh(amb).min())
        if emin < 1e-6:
            warnings.warn(
                f"(A−B) 矩阵最小特征值 {emin:.2e} < 1e-6，"
                "存在近简并激发态，降级为 TDA。",
                UserWarning, stacklevel=3,
            )
            return None, None
        L     = np.linalg.cholesky(amb)
        L_inv = np.linalg.inv(L)
        return L, L_inv

    # ------------------------------------------------------------------
    # 扩展矩阵向量乘积（Cholesky 变换后）
    # ------------------------------------------------------------------

    def _vind_tddft(self, vind_apb, L_inv):
        """
        变换后的矩阵向量乘积（作用于 z = [Lᵀ(X+Y), q]）：

          out_el = L⁻ᵀ (A+B) L⁻¹ z_el + L⁻ᵀ G q
          out_ph = Gᵀ L⁻¹ z_el + W q
        """
        nocc   = self._nocc; nvir   = self._nvir
        n_ia   = nocc * nvir; n_ph   = self.cavity.n_modes
        g_list = self._g_list; modes  = self.cavity.modes
        L_invT = L_inv.T   # L⁻ᵀ = (Lᵀ)⁻¹

        def vind(zs):
            res = []
            for z in zs:
                z      = np.asarray(z, dtype=float)
                z_el   = z[:n_ia]
                z_ph   = z[n_ia:n_ia+n_ph] if len(z) > n_ia else np.zeros(n_ph)

                # (X+Y) = L⁻¹ z_el，施加 (A+B)
                xy_raw  = L_inv @ z_el
                apb_xy  = vind_apb([xy_raw])[0][:n_ia]

                # 加上光子耦合项 G·q
                g_q = sum(g_list[a].ravel() * z_ph[a] for a in range(n_ph))

                out_el = L_invT @ (apb_xy + g_q)

                out_ph = np.array([
                    float(np.dot(g_list[a].ravel(), L_inv @ z_el))
                    + modes[a].omega * z_ph[a]
                    for a in range(n_ph)
                ])

                res.append(np.concatenate([out_el, out_ph]))
            return res

        return vind

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------

    def kernel(self, x0=None, nstates=None):
        """
        求解极化子本征态（完整线性响应）。

        Returns
        -------
        e    : ndarray shape (nstates,)，激发能（a.u.，正值）
        vecs : list of ndarray，本征矢（(X+Y) 空间 + 光子部分）
        """
        if nstates is not None:
            self.nstates = nstates
        self._build()
        ns = self.nstates

        # 获取 PySCF 矩阵向量乘积函数
        try:
            vind_apb, vind_amb = self._get_apb_amb()
        except Exception as ex:
            warnings.warn(f"获取 TDDFT gen_vind 失败（{ex}），降级为 TDA。",
                          UserWarning, stacklevel=2)
            self._fallback_to_tda = True

        if self._fallback_to_tda:
            return QEDTDA.kernel(self, x0=x0, nstates=ns)

        # Cholesky 分解 (A−B)
        try:
            amb      = self._build_amb(vind_amb)
            L, L_inv = self._cholesky(amb)
        except Exception as ex:
            warnings.warn(f"Cholesky 分解失败（{ex}），降级为 TDA。",
                          UserWarning, stacklevel=2)
            self._fallback_to_tda = True
            return QEDTDA.kernel(self, x0=x0, nstates=ns)

        if L is None:
            self._fallback_to_tda = True
            return QEDTDA.kernel(self, x0=x0, nstates=ns)

        # 构建变换后的矩阵向量乘积并求解（本征值为 E²）
        vind_full = self._vind_tddft(vind_apb, L_inv)
        diag      = self._diag()   # 近似对角元（e_ia 和 ω_ph）
        e_sq, v   = self._solve(vind_full, diag, ns)

        # E² → E
        e = np.sqrt(np.maximum(e_sq, 0.0))

        # 变换回 (X+Y) 空间：z_el = Lᵀ(X+Y) → (X+Y) = L⁻¹ z_el
        n_ia   = self._nocc * self._nvir
        n_ph   = self.cavity.n_modes
        xy_out = []
        for vec in (v if v.ndim == 2 else [v]):
            vec    = np.asarray(vec, dtype=float)
            z_el   = vec[:n_ia]
            z_ph   = vec[n_ia:n_ia+n_ph] if len(vec) > n_ia else np.zeros(n_ph)
            xy_raw = L_inv @ z_el
            combined = np.concatenate([xy_raw, z_ph])
            norm  = float(np.dot(combined, combined))
            xy_out.append(combined / np.sqrt(norm) if norm > 1e-14 else combined)

        self.e  = e
        self.xy = xy_out
        self._analyze_weights(self.xy)
        return self.e, self.xy

    # ------------------------------------------------------------------
    # 诊断工具
    # ------------------------------------------------------------------

    def tda_comparison(self, nstates=None) -> dict:
        """
        同时运行 TDA 和 full TDDFT，打印并返回两者激发能、Rabi 劈裂的比较。
        用于量化 B 矩阵修正的量级。
        """
        ns = nstates or self.nstates

        td_tda = QEDTDA(self._scf, self.cavity)
        td_tda.nstates  = ns
        td_tda.conv_tol = self.conv_tol
        td_tda.verbose  = 0
        e_tda, _ = td_tda.kernel()

        e_full, _ = self.kernel(nstates=ns)

        print(f"\n{'态':>4}  {'E_TDA/eV':>10}  {'E_TDDFT/eV':>12}  {'ΔE/meV':>10}")
        print("-" * 44)
        for i, (et, ef) in enumerate(zip(e_tda, e_full)):
            print(f"{i+1:>4}  {et*27.2114:>10.4f}  {ef*27.2114:>12.4f}  "
                  f"{(ef-et)*27211.4:>+10.2f}")

        def _rabi(td_obj):
            from ..analysis.polariton import PolaritonAnalysis
            pol = PolaritonAnalysis(td_obj).compute()
            return pol.rabi_splitting if pol.is_resonant(0.1) else float('nan')

        r_tda  = _rabi(td_tda)
        r_full = _rabi(self)
        for label, r in [("TDA", r_tda), ("TDDFT", r_full)]:
            if not np.isnan(r):
                print(f"  Rabi 劈裂 ({label}):  {r*1000:.2f} meV")
            else:
                print(f"  Rabi 劈裂 ({label}):  非共振")

        return {
            'e_tda_ev':       e_tda  * 27.2114,
            'e_tddft_ev':     e_full * 27.2114,
            'delta_e_meV':    (e_full - e_tda) * 27211.4,
            'rabi_tda_meV':   r_tda  * 1000 if not np.isnan(r_tda)  else float('nan'),
            'rabi_tddft_meV': r_full * 1000 if not np.isnan(r_full) else float('nan'),
        }
