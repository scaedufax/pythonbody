import warnings
import numpy as np

from pythonbody.nbdf import nbdf

class Binaries(nbdf):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.filters = ["BH-BH", "BH-Any"]

    # old calculation of binary Eb
    """def calc_Eb(self, G=4.30091e-3):
        """"""
        see http://web.pd.astro.it/mapelli/colldyn3.pdf
        """"""
        self["Eb"] = ((0.5*self["M1"].values*self["M2"].values)/self[["M1","M2"]].sum(axis=1).values * np.linalg.norm(self[["relV1", "relV2", "relV3"]],axis=1)**2) - (G*(self["M1"].values*self["M2"].values)/(np.linalg.norm(self[["relX1", "relX2", "relX3"]],axis=1)))
        return self["Eb"]"""
    def pd_df_to_class(self, ret):
        """
        if type of a return value is pandas dataframe, return
        a class instance of self.
        """

        return Binaries(ret)

    def calc_relPOT(self, G=4.30091e-3):
        self["relPOT"] = -G*(self["M1"] + self["M2"])/np.linalg.norm(self[["relX1", "relX2", "relX3"]], axis=1)

    def calc_relEb_spec(self):
        if "relEKIN_spec" not in self.columns:
            self.calc_relEKIN_spec()
        self["relEb_spec"] = self["relPOT"] + self["relEKIN_spec"]
    
    def calc_cmEb_spec(self):
        if "cmEKIN_spec" not in self.columns:
            self.calc_cmEKIN_spec()
        self["cmEb_spec"] = self["POT_snap"] + self["cmEKIN_spec"]

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

    def calc_Eb_spec(self):
        self["Eb_spec"] = self["relPOT"] + self["POT_snap"] + 0.5*((self["cmV1"] + self["relV1"])**2 + (self["cmV2"] + self["relV2"])**2 + (self["cmV3"] + self["relV3"])**2)

    def calc_spherical_coords(self):
        """
        Redefinition for binaries as the cols are now named cmX1, cmX2 and cmX3
        See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        """

        self["R"] = np.sqrt(self["cmX1"]**2 + self["cmX2"]**2 + self["cmX3"]**2)
        self["THETA"] = np.arccos(self["cmX3"]/self["R"])

        mask = self["cmX1"] > 0
        self.loc[mask,"PHI"] = np.arctan(self.loc[mask,"cmX2"]/self.loc[mask,"cmX1"])
        mask = (self["cmX1"] < 0) & (self["cmX2"] >= 0)
        self.loc[mask,"PHI"] = np.arctan(self.loc[mask,"cmX2"]/self.loc[mask,"cmX1"]) + np.pi
        mask = (self["cmX1"] < 0) & (self["cmX2"] < 0)
        self.loc[mask,"PHI"] = np.arctan(self.loc[mask,"cmX2"]/self.loc[mask,"cmX1"]) - np.pi
        mask = (self["cmX1"] == 0) & (self["cmX2"] > 0)
        self.loc[mask,"PHI"] = np.pi/2
        mask = (self["cmX1"] == 0) & (self["cmX2"] < 0)
        self.loc[mask,"PHI"] = -np.pi/2
        self["R/Rt"] = self["R"]/self["R"].max()

        self.sort_values("R",ignore_index=True,inplace=True)
    def filter(self,value):
        if value not in self.filters:
            raise KeyError(f"{value} is not a filter type. Available filters: {self.filters}")
        if value == "BH-BH":
            return self[(self["K*1"] == 14) & (self["K*2"] == 14)]
        elif value == "BH-Any":
            return self[(self["K*1"] == 14) | (self["K*2"] == 14)]
