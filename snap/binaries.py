import warnings
import numpy as np

from pythonbody.nbdf import nbdf

class binaries(nbdf):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.filters = ["BH-BH", "BH-Any"]
    def calc_Eb(self, G=4.30091e-3):
        """
        see http://web.pd.astro.it/mapelli/colldyn3.pdf
        """
        self["Eb"] = ((0.5*self["M1"].values*self["M2"].values)/self[["M1","M2"]].sum(axis=1).values * np.linalg.norm(self[["relV1", "relV2", "relV3"]],axis=1)) - (G*(self["M1"].values*self["M2"].values)/(np.linalg.norm(self[["relX1", "relX2", "relX3"]],axis=1)))
        return self["Eb"]

    def filter(self,value):
        if value not in self.filters:
            raise KeyError(f"{value} is not a filter type. Available filters: {self.filters}")
        if value == "BH-BH":
            return self[(self["K*1"] == 14) & (self["K*2"] == 14)]
        elif value == "BH-Any":
            return self[(self["K*1"] == 14) | (self["K*2"] == 14)]
