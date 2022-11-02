from . import distributions
from .powerlaw import SimplePowerLaw


class Salpeter(SimplePowerLaw):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distrib.__call__ = self.distrib._call_fixed_alpha
        
    def __call__(self, m, scale: float = None):
        return self.distrib.__call__(m, scale=scale)
        # return self.distrib._call_fixed_alpha(m, scale=scale)


salpeter = Salpeter()
