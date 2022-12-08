import pandas as pd
import numpy as np

from pythonbody.ffi import ffi

class nbdf(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded dataframe, otherwise returns snap list data
        """
        ret = None
        try:
            ret = super().__getitem__(value)
            #return ret
        except:
            ret = None
        if ret is None:
            try:
                ret = super().loc[value.values[:,0]]
                #return ret
            except:
                ret = None
        if ret is None:
            try:
                ret = super().loc[:,value]
                #return ret
            except:
                ret = None
        
        # Check if item is calulatable
        if ret is None:
            if type(value) != list:
                value = [value]

            missing_list = []
            for val in value:
                if val not in self.columns:
                    missing_list += [val]
                
            if len(missing_list) == 0:
                ret = super().__getitem__(value)
            elif len(missing_list) > 0 and np.sum([f"calc_{val}".replace("/","_over_") not in dir(self) for val in missing_list]) == 0:
                for missing in missing_list:
                    if missing not in self.columns:
                        eval(f"self.calc_{missing}()".replace("/","_over_"))
                ret = self[value[0] if len(value) == 1 else value]
            else:
                raise KeyError(f"Couldn't get key(s) {missing_list}")

        if type(ret) == pd.core.frame.DataFrame:
            return self.pd_df_to_class(ret)
        return ret
    
    def pd_df_to_class(self, ret):
        """
        Changes pandas DataFrame to instalce of self
        """
        return nbdf(ret)
    
    def __repr__(self):
        return super().__repr__()
    
    def _repr_html_(self):
        return super()._repr_html_()

    def calc(self, *args):
        if len(args) != 0:
            for arg in args:
                if f"calc_{arg}" in dir(self):
                    eval(f"self.calc_{arg}()".replace("/","_over_"))
                else:
                    raise KeyError(f"Couldn't calculate {arg}")
        else:
            methods = dir(self)
            for method in methods:
                if "calc_" in method:
                    eval(f"self.calc_{method}()".replace("/","_over_"))

    def calc_all(self):
        for func in [func for func in dir(self) if "calc_" in func and func not in ["calc_R", "calc_THETA", "calc_PHI", "calc_all"]]:
            eval(f"self.{func}()")

    def calc_spherical_coords(self):
        """
        calculates spherical coordinates from cartesian ones.
        See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        """

        self["R"] = np.sqrt(self["X1"]**2 + self["X2"]**2 + self["X3"]**2)
        self["THETA"] = np.arccos(self["X3"]/self["R"])
        
        mask = self["X1"] > 0
        self.loc[mask,"PHI"] = np.arctan(self.loc[mask,"X2"]/self.loc[mask,"X1"])
        mask = (self["X1"] < 0) & (self["X2"] >= 0)
        self.loc[mask,"PHI"] = np.arctan(self.loc[mask,"X2"]/self.loc[mask,"X1"]) + np.pi
        mask = (self["X1"] < 0) & (self["X2"] < 0)
        self.loc[mask,"PHI"] = np.arctan(self.loc[mask,"X2"]/self.loc[mask,"X1"]) - np.pi
        mask = (self["X1"] == 0) & (self["X2"] > 0)
        self.loc[mask,"PHI"] = np.pi/2
        mask = (self["X1"] == 0) & (self["X2"] < 0)
        self.loc[mask,"PHI"] = -np.pi/2
        self["R/Rt"] = self["R"]/self["R"].max()

        self.sort_values("R",ignore_index=True,inplace=True)

    def calc_R(self):
        return self.calc_spherical_coords()

    def calc_THETA(self):
        return self.calc_spherical_coords()

    def calc_PHI(self):
        return self.calc_spherical_coords()

    def calc_EKIN(self):
        self["EKIN"] = 0.5*self["M"]*np.linalg.norm(self[["V1", "V2", "V3"]], axis=1)**2

    def calc_EKIN_spec(self):
        self["EKIN_spec"] = 0.5*np.linalg.norm(self[["V1", "V2", "V3"]], axis=1)**2

    def calc_Eb_spec(self):
        if "EKIN_spec" not in self.columns:
            self.calc_EKIN_spec()
        self["Eb_spec"] = self["EKIN_spec"] + self["POT"]

    def calc_Eb(self):
        if "Eb_spec" not in self.columns:
            self.calc_Eb_spec()
        self["Eb"] = self["Eb_spec"] * self["M"]

    def calc_LZ_spec(self):
        self["LZ_spec"] = self["X1"] * self["V2"] - self["X2"] * self["V1"]

    def calc_LZ(self):
        if "LZ_spec" not in self.columns:
            self.calc_LZ_spec()
        self["LZ"] = self["M"] * self["LZ_spec"]

    def calc_L_spec(self):
        R = self[["X1", "X2", "X3"]]
        V = self[["V1", "V2", "V3"]]
        L_spec = np.cross(R, V) 
        self["L_spec"] = np.linalg.norm(L_spec, axis=1) 
        self["LX_spec"] = L_spec[:, 0]
        self["LY_spec"] = L_spec[:, 1]
        self["LZ_spec"] = L_spec[:, 2]

    def calc_L(self, normalize=False):
        """
        calculate angular momentum L, and stores the values in LX, LY, LZ,
        and the norm into L.

        Parameters:
            normalize (str or bool): can be 'unit', 'system' or False. 'unit'
                normalizes each vector to one, and system normalizes such that
                the total angular momentum of the system is 1.

        """

        if normalize not in ["unit", "system", False]:
            raise ValueError(f"normalize must be 'unit', 'system' or False, but is {unit}")

        R = self[["X1", "X2", "X3"]]
        V = self[["V1", "V2", "V3"]]

        L = np.cross(R, V) * self[["M"]].values
        self["L"] = np.linalg.norm(L, axis=1) 
        self["LX"] = L[:, 0]
        self["LY"] = L[:, 1]
        self["LZ"] = L[:, 2]

        if normalize == "unit":
            self.loc[:, ["LX", "LY", "LZ"]] = self.loc[:, ["LX", "LY", "LZ"]] / np.full((3,self.shape[0]), self.loc[:,"L"].values).T
            self.loc[:, "L"] = 1
        if normalize == "system":
            self.loc[:, ["LX", "LY", "LZ"]] = self.loc[:, ["LX", "LY", "LZ"]] / self.loc[:, ["L"]].values.sum()
            self.loc[:, "L"] = self.loc[:, "L"]/self.loc[:, "L"].sum()



    def calc_M_over_MT(self):
        if "R" not in self.columns:
            self.calc_R()

        self["M/MT"] = self["M"].cumsum()/self["M"].sum()

    def calc_R_over_RT(self):
        if "R" not in self.columns:
            self.calc_R()
        self["R/RT"] = self["R"].cumsum()/self["R"].sum()

    def calc_VROT(self):
        if "R" not in self.columns:
            self.calc_R()
        
        """
        # nbody style
        rvxy = self["X1"]*self["V1"] + self["X2"] * self["V2"]
        rxy2 = self["X1"]**2 + self["X2"]**2
        vrot1 = self["V1"] - rvxy * self["X1"]/rxy2
        vrot2 = self["V2"] - rvxy * self["X2"]/rxy2
        self["VROT"] = np.sqrt(vrot1**2 + vrot2**2)
        mask = (vrot1*self["X2"] - vrot2*self["X1"]) < 0
        self.cluster_data.loc[mask,"VROT"] = - self.cluster_data.loc[mask,"VROT"]
        """

        # pythonbody style
        VROT = np.cross( 
                    np.cross(
                        self.loc[:,["X1","X2","X3"]],
                        self.loc[:,["V1","V2","V3"]]
                    )/(self["R"]**2).values.reshape(self.shape[0],1),
                    self.loc[:,["X1","X2","X3"]]
                )
        XSIGN = np.sign(VROT[:,0]*self["X2"]/np.sqrt(self["X1"]**2 + self["X2"]**2) \
                            - VROT[:,1]*self["X1"]/np.sqrt(self["X1"]**2 + self["X2"]**2))
        self["VROT"] = XSIGN * np.sqrt(VROT[:,0]**2 + VROT[:,1]**2)

    def calc_VROT_CUMMEAN(self):
        if "VROT" not in self.columns:
            self.calc_VROT()

        self["VROT_CUMMEAN"] = ffi.cummean(self["VROT"].values)
