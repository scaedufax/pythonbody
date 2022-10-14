import warnings
import numpy as np

from pythonbody.nbdf import nbdf

class Binaries(nbdf):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.filters = ["BH-BH", "BH-Any"]
    
    """def calc_Eb(self, G=4.30091e-3):
        """"""
        see http://web.pd.astro.it/mapelli/colldyn3.pdf
        """"""
        self["Eb"] = ((0.5*self["M1"].values*self["M2"].values)/self[["M1","M2"]].sum(axis=1).values * np.linalg.norm(self[["relV1", "relV2", "relV3"]],axis=1)**2) - (G*(self["M1"].values*self["M2"].values)/(np.linalg.norm(self[["relX1", "relX2", "relX3"]],axis=1)))
        return self["Eb"]"""
    def calc_Eb_spec(self):
        if "relEKIN_spec" not in self.columns:
            self.calc_relEKIN_spec()
        self["Eb_spec"] = self["relPOT"] + self["relEKIN_spec"]


    def calc_relLZ_spec(self):
        self["relLZ_spec"] = self["relX1"]*self["relV2"] - self["relX2"]*self["relV1"]

    def calc_relLZ(self):
        if "relLZ_spec" not in self.columns:
            self.calc_relLZ_spec()
        self["relLZ"] = self["relLZ_spec"] * (self["M1"].values*self["M2"].values)/self[["M1","M2"]].sum(axis=1).values
    
    def calc_cmLZ_spec(self):
        self["cmLZ_spec"] = self["cmX1"]*self["cmV2"] - self["cmX2"]*self["cmV1"]

    def calc_cmLZ(self):
        if "cmLZ_spec" not in self.columns:
            self.calc_relLZ_spec()
        self["cmLZ"] = self["cmLZ_spec"] * (self["M1"].values*self["M2"].values)/self[["M1","M2"]].sum(axis=1).values

    def calc_relEKIN_spec(self):
        self["relEKIN_spec"] = 0.5*np.linalg.norm(self[["relV1", "relV2", "relV3"]], axis=1)**2
    def calc_relEKIN(self):
        if "relEKIN_spec" not in self.columns:
            self.calc_relEKIN_spec()
        self["relEKIN"] = self["relEKIN_spec"] * (self["M1"].values*self["M2"].values)/self[["M1","M2"]].sum(axis=1).values

    def calc_cmEKIN_spec(self):
        self["cmEKIN_spec"] = 0.5*np.linalg.norm(self[["cmV1", "cmV2", "cmV3"]], axis=1)**2

    def calc_cmEKIN(self):
        if "cmEKIN_spec" not in self.columns:
            self.calc_cmEKIN_spec()
        self["cmEKIN"] = self["cmEKIN_spec"] * (self["M1"].values*self["M2"].values)/self[["M1","M2"]].sum(axis=1).values

    def filter(self,value):
        if value not in self.filters:
            raise KeyError(f"{value} is not a filter type. Available filters: {self.filters}")
        if value == "BH-BH":
            return self[(self["K*1"] == 14) & (self["K*2"] == 14)]
        elif value == "BH-Any":
            return self[(self["K*1"] == 14) | (self["K*2"] == 14)]
