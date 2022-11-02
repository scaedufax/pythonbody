import numpy as np
from scipy.optimize import curve_fit

class IMF:
    def __init__(self):
        print(__name__)
        pass

    def __call__(self, m, *args, **kwargs):
        return self.distrib.__call__(m, *args, **kwargs)

    def draw(self,
             n: int,
             mmin: float = 0.03,
             mmax: float = 120,
             ):

        return self.distrib.draw(n, mmin, mmax)

    def fit(self,
            m,
            dm: float = None,
            fit_mask=None,
            fit_n_geq: float = 10,
            *args,
            **kwargs
            ):

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

        # fit only bins containing more than fit_n_geq
        n_geq_mask = hist[0] >= fit_n_geq

        popt, pcov = curve_fit(self.distrib.__call__,
                               hist[1][1:][n_geq_mask],
                               hist[0][n_geq_mask],
                               )
        return popt, pcov


from .salpeter import Salpeter
