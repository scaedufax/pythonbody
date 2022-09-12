import numpy as np
import warnings

from pythonbody.nbdf import nbdf

class singles(nbdf):
    def __init__(self,full,binaries,*args,**kwargs):
        if binaries.shape[0] != 0:
            super().__init__(full[~full["NAME"].isin(binaries["NAME1"]) & ~full["NAME"].isin(binaries["NAME2"])], *args, **kwargs)
        else:
            super().__init__(full, *args, **kwargs)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.filters = ["BH"]
    def filter(self,value):
        if value not in self.filters:
            raise KeyError(f"{value} is not a filter type. Available filters: {self.filters}")
        if value == "BH":
            return self[self["K*"] == 14]
