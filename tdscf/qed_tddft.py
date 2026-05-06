# -*- coding: utf-8 -*-
"""
nqeddft.tdscf.qed_tddft
=======================
QED-TDDFT 线性响应（极化子谱）

扩展 Casida 矩阵（Tamm-Dancoff 近似，TDA）：

    [A_el   G  ] [X ]       [X ]
    [G^T   W_ph] [q ] = E * [q ]

    A_el : 电子-空穴激发矩阵（PySCF 原版）
    G_ia^a = sqrt(w_a/2) * <i|lam_a·d|a>   光子-激子耦合
    W_ph   = diag(w_a)                      光子频率

与 PySCF 的关键接口：
  - 继承 pyscf.tdscf.rhf.TDA，override gen_vind
  - 使用 pyscf.lib.davidson1 求解本征值问题
  - precond 使用对角能量差（标准做法）
"""
import numpy as np
from pyscf.tdscf import rhf as td_rhf
from pyscf import lib
from ..cavity import Cavity
from ..integrals.dipole import DipoleIntegrals


class QEDTDA(td_rhf.TDA):
    """
    QED Tamm-Dancoff Approximation。
    求解上/下极化子，输出激发能、振子强度、Hopfield 系数。

    Parameters
    ----------
    mf     : QEDRKS（已收敛）
    cavity : Cavity
    """

    def __init__(self, mf, cavity: Cavity):
        super().__init__(mf)
        self.cavity   = cavity
        self.dip_ints = DipoleIntegrals(mf.mol)
        self._g_ia    = None       # 光子-激子耦合矩阵元列表
        self._nocc    = 0
        self._nvir    = 0
        self._mo_o    = None
        self._mo_v    = None

    # ------------------------------------------------------------------
    # 构建耦合矩阵元
    # ------------------------------------------------------------------

    def _build_coupling(self):
        """
        构建 G_ia^a = sqrt(w_a/2) * <i|lam_a·d|a>，MO 基，shape (nocc, nvir)。
        在 kernel() 第一次调用时执行，结果缓存。
        """
        mf       = self._scf
        mo_coeff = mf.mo_coeff
        mo_occ   = mf.mo_occ
        mo_o     = mo_coeff[:, mo_occ > 0]    # (nao, nocc)
        mo_v     = mo_coeff[:, mo_occ == 0]   # (nao, nvir)
        dip_ao   = self.dip_ints.dip_ao       # (3, nao, nao)

        g_list = []
        for mode in self.cavity.modes:
            ld_ao = np.einsum('k,kpq->pq', mode.lambda_vec, dip_ao)  # (nao,nao)
            g_ia  = mo_o.T @ ld_ao @ mo_v    # (nocc, nvir)
            g_list.append(mode.sqrt_omega_half * g_ia)

        self._g_ia  = g_list
        self._nocc  = mo_o.shape[1]
        self._nvir  = mo_v.shape[1]
        self._mo_o  = mo_o
        self._mo_v  = mo_v
        return g_list

    # ------------------------------------------------------------------
    # Override gen_vind：注入光子-激子耦合块
    # ------------------------------------------------------------------

    def gen_vind(self, mf=None):
        """
        构建扩展矩阵向量乘积函数 vind(zs)。
        扩展向量 z = [z_ia (电子, n_ia), q_a (光子, n_ph)]
        """
        if mf is None:
            mf = self._scf
        if self._g_ia is None:
            self._build_coupling()

        # PySCF 原版电子响应 A·X（TDA 级别）
        vind_elec, _ = super().gen_vind(mf)

        nocc   = self._nocc
        nvir   = self._nvir
        n_ia   = nocc * nvir
        n_ph   = self.cavity.n_modes
        g_list = self._g_ia
        modes  = self.cavity.modes

        def vind_qed(zs):
            """
            返回扩展 Casida 矩阵与向量的乘积列表。
            输入 zs：list of ndarray, 每个 shape (n_ia+n_ph,)
            输出：同结构的列表
            """
            results = []
            for z in zs:
                z   = np.asarray(z, dtype=float)
                z_ia = z[:n_ia].reshape(nocc, nvir)
                z_ph = z[n_ia:] if len(z) > n_ia else np.zeros(n_ph)

                # 电子块：A·X + G·q
                ax = vind_elec([z_ia.ravel()])[0].reshape(nocc, nvir)
                for i, g in enumerate(g_list):
                    ax = ax + g * z_ph[i]

                # 光子块：W·q + G^T·X
                bq = np.zeros(n_ph)
                for i in range(n_ph):
                    bq[i] = (modes[i].omega * z_ph[i]
                              + float(np.einsum('ia,ia->', g_list[i], z_ia)))

                results.append(np.concatenate([ax.ravel(), bq]))
            return results

        return vind_qed

    # ------------------------------------------------------------------
    # kernel：构建 precond，调用 Davidson 求解器
    # ------------------------------------------------------------------

    def kernel(self, x0=None, nstates=None):
        """
        求解极化子本征态。
        返回 (energies_au, eigenvectors)。
        """
        if nstates is not None:
            self.nstates = nstates
        if self._g_ia is None:
            self._build_coupling()

        nocc  = self._nocc
        nvir  = self._nvir
        n_ia  = nocc * nvir
        n_ph  = self.cavity.n_modes
        n_tot = n_ia + n_ph
        ns    = getattr(self, 'nstates', 5)

        # ── 预条件子 precond：对角 Hessian 近似 ────────────────────────
        # 对电子部分：轨道能量差 e_a - e_i
        mo_e   = self._scf.mo_energy
        mo_occ = self._scf.mo_occ
        e_occ  = mo_e[mo_occ > 0]
        e_vir  = mo_e[mo_occ == 0]
        e_ia   = (e_vir[None, :] - e_occ[:, None]).ravel()   # (n_ia,)
        # 对光子部分：腔模频率
        e_ph   = np.array([m.omega for m in self.cavity.modes])
        diag   = np.concatenate([e_ia, e_ph])                 # (n_tot,)

        def precond(dx, e, x0_):
            """标准 Davidson 预条件：(diag - e)^{-1} * dx"""
            denom = diag - e
            # 避免除以过小值
            denom = np.where(np.abs(denom) > 1e-6, denom, 1e-6)
            return dx / denom

        # ── 初始猜测向量 ────────────────────────────────────────────────
        if x0 is None:
            idx = np.argsort(diag)[:ns]
            x0  = np.zeros((ns, n_tot))
            for k, i in enumerate(idx):
                x0[k, i] = 1.0

        # ── PySCF Davidson 求解 ─────────────────────────────────────────
        # 使用 lib.davidson1（PySCF >= 2.0 的稳定接口）
        vind  = self.gen_vind()
        tol   = getattr(self, 'conv_tol', 1e-6)
        max_c = getattr(self, 'max_cycle', 100)

        try:
            conv, e, v = lib.davidson1(
                vind, x0, precond,
                tol=tol,
                nroots=ns,
                max_space=max(ns * 8, 20),
                max_cycle=max_c,
                verbose=self.verbose,
            )
        except AttributeError:
            # 兼容旧版 pyscf：lib.davidson
            conv, e, v = lib.davidson(
                vind, x0, diag,
                tol=tol,
                nroots=ns,
                max_space=max(ns * 8, 20),
                max_cycle=max_c,
                verbose=self.verbose,
            )

        self.e  = np.asarray(e)
        v = np.array(v); self.xy = list(v) if v.ndim == 2 else [v]
        self._analyze_weights(self.xy, n_ia)
        return self.e, self.xy

    # ------------------------------------------------------------------
    # 极化子分析
    # ------------------------------------------------------------------

    def _analyze_weights(self, vecs, n_ia):
        """计算各本征态的光子权重和激子权重（Hopfield 系数）。"""
        self.photon_weight  = []
        self.exciton_weight = []
        for v in vecs:
            v = np.asarray(v)
            norm = float(np.dot(v, v))
            if norm < 1e-14:
                norm = 1.0
            ph = float(np.dot(v[n_ia:], v[n_ia:])) / norm
            ex = float(np.dot(v[:n_ia], v[:n_ia])) / norm
            self.photon_weight.append(ph)
            self.exciton_weight.append(ex)

    def oscillator_strength(self) -> np.ndarray:
        """
        各极化子态的振子强度 f_k = (2/3)*E_k*|<0|r|k>|^2。
        shape (nstates,)。

        注意：跃迁偶极矩用长度规范（r），结果依赖规范原点选取。
        对于中性分子，规范原点通常选在分子质心，此时结果与速度规范一致。
        严格规范不变性需要用 velocity_ao 做交叉验证（已在 DipoleIntegrals 中实现）。
        核贡献对激发态跃迁矩阵元无贡献（核坐标不变），无需额外修正。
        """
        if not hasattr(self, 'e'):
            raise RuntimeError("请先调用 kernel()")

        mo_o   = self._mo_o
        mo_v   = self._mo_v
        nocc   = self._nocc
        nvir   = self._nvir
        n_ia   = nocc * nvir
        dip_ao = self.dip_ints.dip_ao   # (3, nao, nao)

        # 偶极跃迁矩阵元 <i|r_k|a>，shape (3, nocc, nvir)
        dip_mo = np.einsum('kpq,pi,qa->kia', dip_ao, mo_o, mo_v)

        f_list = []
        for E, v in zip(self.e, self.xy):
            v    = np.asarray(v)
            x_ia = v[:n_ia].reshape(nocc, nvir)
            mu   = np.einsum('kia,ia->k', dip_mo, x_ia)
            f    = (2.0 / 3.0) * float(E) * float(np.dot(mu, mu))
            f_list.append(max(0.0, f))   # 防止数值噪声导致负值
        return np.array(f_list)

    def print_spectrum(self):
        """打印极化子谱表格。"""
        if not hasattr(self, 'e'):
            raise RuntimeError("请先调用 kernel()")
        f = self.oscillator_strength()
        print("=" * 62)
        print(f"{'态':>4} {'E/eV':>9} {'E/a.u.':>10} "
              f"{'振子强度':>12} {'光子%':>8} {'激子%':>8}")
        print("-" * 62)
        for i, E in enumerate(self.e):
            print(f"{i+1:>4} {E*27.2114:>9.4f} {E:>10.6f} "
                  f"{f[i]:>12.6f} "
                  f"{self.photon_weight[i]*100:>7.2f}% "
                  f"{self.exciton_weight[i]*100:>7.2f}%")
        print("=" * 62)


class QEDTDDFT(QEDTDA):
    """
    完整 QED-TDDFT（含退激发 B 矩阵）。

    警告：当前 B 矩阵（退激发耦合块）尚未实现。
    此类与 QEDTDA 完全等价，仅作占位符。
    若需要完整线性响应结果，请显式使用 QEDTDA 并知悉其 TDA 近似。
    完整 B 块实现在 Phase 2 路线中。
    """

    def kernel(self, x0=None, nstates=None):
        import warnings
        warnings.warn(
            "QEDTDDFT 的 B 矩阵（退激发块）尚未实现，"
            "当前结果与 QEDTDA（Tamm-Dancoff 近似）完全相同。"
            "如需完整线性响应，请等待 Phase 2 实现。",
            UserWarning,
            stacklevel=2,
        )
        return super().kernel(x0=x0, nstates=nstates)
