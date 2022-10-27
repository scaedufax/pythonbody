import numpy as np
from scipy.optimize import curve_fit

from . import distributions


class Kroupa:
    def __init__(
            self,
            breakpoints=[0.08, 0.5],
            alphas=[0.3, 1.3, 2.3],
            scale=1,
            *args, **kwargs):
        self.distrib = distributions.BrokenPowerLaw(
                breakpoints=breakpoints,
                alphas=alphas,
                scale=scale
                )

    def __call__(self, m, scale=None):
        return self.distrib.__call__(m, scale)
    
    def fit(self, m, dm: float = None, fit_mask=None, *args, **kwargs):
        if fit_mask is not None:
            if type(fit_mask) == str:
                fit_mask = eval(fit_mask)

            m = m[fit_mask]
        if dm is None:
            dm = (m.max()-m.min())/1000

        print(m.min())

        hist = np.histogram(
                m,
                density=True,
                bins=int((m.max() - m.min())/dm)
                )

        popt, pcov = curve_fit(self.__call__,
                               hist[1][1:],
                               hist[0],
                               *args,
                               **kwargs)
        return popt, pcov


kroupa = Kroupa()
