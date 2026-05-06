# -*- coding: utf-8 -*-
"""
nqeddft.analysis.spectrum  —  光谱生成与输出

将极化子本征态卷积为连续吸收光谱，支持 Lorentzian 和 Gaussian 展宽。
"""
import numpy as np


class AbsorptionSpectrum:
    """
    吸收光谱生成器。

    Parameters
    ----------
    td  : QEDTDA / QEDTDDFT（已调用 kernel()）
    """
    def __init__(self, td):
        self.td = td

    def compute(self, e_range: tuple = None, n_pts: int = 2000,
                fwhm: float = 0.05, lineshape: str = 'lorentzian') -> dict:
        """
        生成吸收光谱。

        Parameters
        ----------
        e_range   : (e_min, e_max) eV，默认自动确定
        n_pts     : 频率格点数
        fwhm      : 半高全宽，eV
        lineshape : 'lorentzian' 或 'gaussian'

        Returns
        -------
        dict: {'energy_ev': ..., 'intensity': ...}
        """
        td     = self.td
        if not hasattr(td, 'e'): raise RuntimeError("先调用 kernel()")
        e_ev   = td.e * 27.2114
        f      = td.oscillator_strength()
        gamma  = fwhm / 2.0

        if e_range is None:
            e_range = (max(0, e_ev.min() - 0.5), e_ev.max() + 0.5)
        grid   = np.linspace(e_range[0], e_range[1], n_pts)
        spec   = np.zeros(n_pts)

        for E_k, f_k in zip(e_ev, f):
            if lineshape == 'lorentzian':
                spec += f_k * gamma / (np.pi * ((grid - E_k)**2 + gamma**2))
            else:
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                spec += f_k * np.exp(-0.5*((grid-E_k)/sigma)**2) / (sigma*np.sqrt(2*np.pi))

        return {'energy_ev': grid, 'intensity': spec, 'peaks_ev': e_ev, 'osc': f}

    def save_csv(self, filename: str, **kwargs):
        """将光谱保存为 CSV 文件"""
        data = self.compute(**kwargs)
        np.savetxt(filename,
                   np.column_stack([data['energy_ev'], data['intensity']]),
                   delimiter=',', header='energy_eV,intensity',
                   comments='')
        print(f"光谱已保存至 {filename}")
