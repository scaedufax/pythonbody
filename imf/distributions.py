import numpy as np


class PowerLaw:
    def __init__(self, alpha: float, scale: float = 1):
        self.alpha = alpha
        self.scale = scale

    def __call__(self, m, alpha: float = None, scale: float = None):
        if alpha is None:
            alpha = self.alpha
        if scale is None:
            scale = self.scale

        if type(m) in [int, float]:
            m = np.array([m]) 

        if type(m) != np.ndarray:
            m = np.array(m)

        return scale*m**(-alpha)

class BrokenPowerLaw:
    def __init__(self,
                 breakpoints: list,
                 alphas: list,
                 scale: float = 1,
                 ):

        # check for errors
        if len(breakpoints) + 1 != len(alphas):
            raise ValueError("Number of breakpoints + 1 must equal" +
                             "to number of Alphas!")
        if not (np.diff(breakpoints) > 0).all():
            raise ValueError("Breakpoints must be in sequential order" +
                             "from low to high!")
        
        self.breakpoints = breakpoints
        self.alphas = alphas
        self.scale = scale
       
        # introduce scaling between breakpoints
        self.alphas_scale = []
        for i in range(len(breakpoints) + 1, 0, -1):
            if i == len(self.breakpoints) + 1:
                self.alphas_scale = [1] + self.alphas_scale
                continue
            self.alphas_scale = ([
                    (self.breakpoints[i-1]**(-self.alphas[i])) 
                    / (self.breakpoints[i-1]**(-self.alphas[i-1]))
                    * self.alphas_scale[0]]

                                 + self.alphas_scale)

    def __call__(self, m, scale=None):
        if type(m) in [int, float]:
            m = [m]
        m = np.array(m)

        if scale is None:
            scale = self.scale

        ret = np.zeros(m.shape[0])

        for i in range(len(self.breakpoints) + 1, 0, -1):
            mask = None
            if i == 1:
                mask = m < self.breakpoints[0]
            elif i == len(self.breakpoints) + 1:
                mask = m > self.breakpoints[len(self.breakpoints) - 1]
            else:
                mask = (m < self.breakpoints[i-1]) & (m > self.breakpoints[i-2])

            if mask is None:
                raise ValueError("Something went terribly wrong as mask is None")

            ret[mask] = self.alphas_scale[i-1] * m[mask]**(-self.alphas[i-1])

        return scale * ret