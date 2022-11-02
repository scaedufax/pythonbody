import numpy as np
from scipy.optimize import curve_fit

from . import distributions
from .imf import IMF
#from .salpeter import Salpeter


class SimplePowerLaw(IMF):
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


simplepowerlaw = SimplePowerLaw()
