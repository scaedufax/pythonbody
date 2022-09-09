import pandas as pd
import numpy as np

class nbdf(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded dataframe, otherwise returns snap list data
        """
        try:
            return super().__getitem__(value)
        except:
            pass
        try:
            return super().loc[value.values[:,0]]
        except:
            pass
        try:
            return super().loc[:,value]
        except:
            pass
        
        # Check if item is calulatable
        if type(value) != list:
            value = [value]

        missing_list = []
        for val in value:
            if val not in self.columns:
                missing_list += [val]
            
        if len(missing_list) == 0:
            return super().__getitem__(value)
        elif len(missing_list) > 0 and np.sum([f"calc_{val}".replace("/","_over_") not in dir(self) for val in missing_list]) == 0:
            for missing in missing_list:
                if missing not in self.columns:
                    eval(f"self.calc_{missing}()".replace("/","_over_"))
            return self[value[0] if len(value) == 1 else value]
        else:
            raise KeyError(f"Couldn't get key(s) {missing_list}")

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

    def calc_M_over_MT(self,M_col = "M"):
        if M_col not in self.columns:
            raise KeyError("Couldn't calculate M/MT due to missing mass column")

        self["M/MT"] = (self[M_col]/self[M_col].sum()).cumsum()
        self.sort_values("M/MT", inplace=True)
