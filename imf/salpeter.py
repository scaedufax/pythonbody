import numpy as np
from scipy.optimize import curve_fit

from . import distributions
from .powerlaw import SimplePowerLaw


class Salpeter(SimplePowerLaw):
    def __call__(self, m, scale: float = None):
        return self.distrib.__call__(m, scale=scale)

salpeter = Salpeter()
