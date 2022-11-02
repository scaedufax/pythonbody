from . import distributions
from .imf import IMF


class Kroupa(IMF):
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


kroupa = Kroupa()
