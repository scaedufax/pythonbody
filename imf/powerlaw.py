import numpy as np
from scipy.optimize import curve_fit

from . import distributions


class SimplePowerLaw:
    def __init__(
            self,
            alpha=2.3,
            scale=1,
            ):
        self.alpha = alpha,
        self.scale = scale,
        self.distrib = distributions.PowerLaw(
                alpha=alpha,
                scale=scale
                )

    def __call__(self, m, alpha: float = None, scale: float = None):
        return self.distrib.__call__(m, alpha, scale)

    def fit(self, m, dm: float = None, fit_mask=None, *args, **kwargs):
        if fit_mask is not None:
            if type(fit_mask) == str:
                fit_mask = eval(fit_mask)

            m = m[fit_mask]
        if dm is None:
            dm = (m.max()-m.min())/1000

        hist = np.histogram(
                m,
                # density=True,
                bins=int((m.max() - m.min())/dm)
                )

        popt, pcov = curve_fit(self.__call__,
                               hist[1][1:],
                               hist[0],
                               ) 
        return popt, pcov

simplepowerlaw = SimplePowerLaw()
