import pandas as pd
import numpy as np

from pythonbody.ffi import ffi

class nbdf(pd.DataFrame):
    """
    Basic DataFrame used most of the time. Extends pandas.DataFrame, adds a lot
    of calculatable values from the raw data.

    This is the base class for handling an entire cluster with all it's data

    :param \*args: see pandas.DataFrame documentation
    :param \*\*kwargs: see pandas.DataFrame documentation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded dataframe, otherwise returns snap list data
        """
        ret = None
        try:
            ret = super().__getitem__(value)
        except:
            ret = None
        if ret is None:
            try:
                ret = super().loc[value.values[:,0]]
            except:
                ret = None
        if ret is None:
            try:
                ret = super().loc[:,value]
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
        Changes pandas DataFrame to instalce of nbody_data_frame
        """
        return nbdf(ret)

    @property
    def COM(self):
        """
        return: center of mass
        rtype: float[3]
        """
        return 1/self.loc[:, "M"].sum() * np.sum(self.loc[:, ["X1", "X2", "X3"]].multiply(self.loc[:, "M"], axis=0))
    
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

        | Required columns: ``X1``, ``X2``, ``X3``        
        | Output columns: ``R``, ``THETA``, ``PHI``
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
        """
        maps to calc_spherical_coords
        """
        return self.calc_spherical_coords()

    def calc_THETA(self):
        """
        maps to calc_spherical_coords
        """
        return self.calc_spherical_coords()

    def calc_PHI(self):
        """
        maps to calc_spherical_coords
        """
        return self.calc_spherical_coords()

    def calc_EKIN(self):
        """
        calculates kinetic energy into ``EKIN`` column. 

        | Required columns: ``M``, ``V1``, ``V2``, ``V3``        
        | Output columns: ``EKIN``
        """
        self["EKIN"] = 0.5*self["M"]*np.linalg.norm(self[["V1", "V2", "V3"]], axis=1)**2

    def calc_EKIN_spec(self):
        """
        calculates specific kinetic energy into ``EKIN_spec`` column. 

        | Required columns: ``V1``, ``V2``, ``V3``        
        | Output columns: ``EKIN_spec``
        """
        self["EKIN_spec"] = 0.5*np.linalg.norm(self[["V1", "V2", "V3"]], axis=1)**2

    def calc_EROT_spec(self, kwargs_calc_vrot: dict = {}):
        """
        calculates specific rotational energy from VROT

        | Required columns: ``X1``, ``X2``, ``X3``, ``V1``, ``V2``, ``V3``
        | Intermediary columns: ``VROT``
        | Output columns: ``EROT_spec``

        :param kwargs_calc_vrot: parameters for calc_VROT()
            e.g. { "method": "nbody", "sign_nbody": True}
        :type kwargs_calc_vrot: dict
        """
        if "VROT" not in self.columns:
            self.calc_VROT(**kwargs_calc_vrot)
        
        self["EROT_spec"] = 0.5*self.loc[:, "VROT"]**2
    
    def calc_EROT(self, kwargs_calc_vrot: dict = {}):
        """
        calculates rotational energy from VROT

        | Required columns: ``M``, ``X1``, ``X2``, ``X3``, ``V1``, ``V2``, ``V3``
        | Intermediary columns: ``VROT``
        | Output columns: ``EROT``

        :param kwargs_calc_vrot: parameters for calc_VROT()
            e.g. { "method": "nbody", "sign_nbody": True}
        :type kwargs_calc_vrot: dict
        """
        if "VROT" not in self.columns:
            self.calc_VROT(**kwargs_calc_vrot)
        
        self["EROT"] = 0.5*self.loc[:, "VROT"]**2 * self.loc[:, "M"]

    def calc_EPOT(self, G: float = 1):
        """
        calculates potential energy.

        :param G: gravitational constant
        :type G: float

        | Required columns: ``M``, ``X1``, ``X2``, ``X3``
        | Output columns: ``EPOT``
        """
        self["EPOT"] = G * ffi.grav_pot(self.loc[:, ["M", "X1", "X2", "X3"]])
    
    def calc_EPOT_spec(self, G: float = 1):
        """
        calculates specific potential energy.

        :param G: gravitational constant
        :type G: float

        | Required columns: ``M``, ``X1``, ``X2``, ``X3``
        | Intermediary columns: ``EPOT``
        | Output columns: ``EPOT_spec``
        """
        if "EPOT" not in self.columns:
            self.calc_EPOT(G)

        self["EPOT_spec"] = self.loc[:, "EPOT"].values \
                                / self.loc[:, "M"].values

    def calc_Eb_spec(self):
        """
        calculate specific binding energy (E_b = E_kin + E_pot).

        | Required columns: ``V1``, ``V2``, ``V3``, ``POT``
        | Intermediary columns: ``EKIN_spec``
        | Output columns ``Eb_spec``
        """
        if "EKIN_spec" not in self.columns:
            self.calc_EKIN_spec()
        self["Eb_spec"] = self["EKIN_spec"] + self["POT"]

    def calc_Eb(self):
        """
        calculate binding energy (E_b = E_kin + E_pot), and uses Eb_spec
        as intermediate result.

        | Required columns: ``V1``, ``V2``, ``V3``, ``POT``
        | Intermediary columns: ``Eb_spec``, ``EKIN_spec``
        | Output columns: ``Eb``
        """

        if "Eb_spec" not in self.columns:
            self.calc_Eb_spec()
        self["Eb"] = self["Eb_spec"] * self["M"]

    def calc_LZ_spec(self):
        """
        calculates specific angular Momentum in z-Direction

        | Required columns: ``X1``, ``X2``, ``V1``, ``V2``
        | Output columns: ``LZ_spec``
        """
        X1 = self["X1"]
        X2 = self["X2"]
        V1 = self["V1"]
        V2 = self["V2"]

        if not np.allclose(self.COM, (0, 0, 0)):
            X1 = X1 - self.COM.values[0]
            X2 = X2 - self.COM.values[1]

        self["LZ_spec"] = X1 * V2 - X2 * V1

    def calc_LZ(self):
        """
        calculates specific angular Momentum in z-Direction, using LZ_spec
        as an intermediate result.

        | Required columns: ``X1``, ``X2``, ``V1``, ``V2``
        | Intermediate columns: ``LZ_spec``
        | Output columns: ``LZ``
        """
        if "LZ_spec" not in self.columns:
            self.calc_LZ_spec()
        self["LZ"] = self["M"] * self["LZ_spec"]

    def calc_L_spec(self):
        """
        calculate full specific angular Momentum vector.

        | Required columns: ``X1``, ``X2``, ``X3``, ``V1``, ``V2``, ``V3``
        | Output columns: ``L_spec`` (norm), ``LX_spec``, ``LY_spec``, 
          ``LZ_spec``
        """
        R = self[["X1", "X2", "X3"]]
        V = self[["V1", "V2", "V3"]]
        
        if not np.allclose(self.COM, (0, 0, 0)):
            R = R - self.COM

        L_spec = np.cross(R, V) 
        self["L_spec"] = np.linalg.norm(L_spec, axis=1) 
        self["LX_spec"] = L_spec[:, 0]
        self["LY_spec"] = L_spec[:, 1]
        self["LZ_spec"] = L_spec[:, 2]

    def calc_L(self, normalize: str = None):
        """
        calculate angular momentum L, and stores the values in LX, LY, LZ,
        and the norm into L.
        
        | Required columns: ``X1``, ``X2``, ``X3``, ``V1``, ``V2``, ``V3``
        | Output columns: ``L`` (norm), ``LX``, ``LY``, ``LZ``

        :param normalize: Optional noramlize angular momentum. Can be ``unit``,
            ``system``, ``mean`` or ``None``.            
            
            ``unit`` normalizes each vector to one, system normalizes such that
            the total angular momentum of the system is 1
            
            ``system`` normalizes the entire system to a total angular momentum
            of one            
            
            ``mean`` normalizes such that the mean of the enitre system is one.
            
            ``None`` leave everything as is
        
        :type normalize: str or None 
        """

        if normalize not in ["unit", "system", "mean", None]:
            raise ValueError(f"normalize must be 'unit', 'system' or False, but is {normalize}")

        R = self[["X1", "X2", "X3"]].values
        V = self[["V1", "V2", "V3"]].values
        
        if not np.allclose(self.COM, (0, 0, 0)):
            R = R - self.COM.values

        L = np.cross(R, V) * self[["M"]].values
        self["L"] = np.linalg.norm(L, axis=1) 
        self["LX"] = L[:, 0]
        self["LY"] = L[:, 1]
        self["LZ"] = L[:, 2]

        if normalize == "unit":
            self.loc[:, ["LX", "LY", "LZ"]] = self.loc[:, ["LX", "LY", "LZ"]] / np.full((3,self.shape[0]), self.loc[:,"L"].values).T
            self.loc[:, "L"] = 1
        elif normalize == "system":
            self.loc[:, ["LX", "LY", "LZ"]] = self.loc[:, ["LX", "LY", "LZ"]] / self.loc[:, ["L"]].values.sum()
            self.loc[:, "L"] = self.loc[:, "L"]/self.loc[:, "L"].sum()
        elif normalize == "mean":
            self.loc[:, ["LX", "LY", "LZ", "L"]] = self.loc[:, ["LX", "LY", "LZ", "L"]] / self.loc[:, "L"].mean()



    def calc_M_over_MT(self):
        """
        calculate M/M_T within the shell below

        | Required columns: ((``X1``, ``X2``, ``X3``) or ``R``), ``M``
        | Intermediate columns: ``R``
        | Output columns: ``M/MT``
        """
        if "R" not in self.columns:
            self.calc_R()

        self["M/MT"] = self["M"].cumsum()/self["M"].sum()

    def calc_R_over_RT(self):
        """
        calculate R/R_T within the shell below

        | Required columns: (``X1``, ``X2``, ``X3``) or ``R``
        | Intermediate columns: ``R``
        | Output columns: ``R/RT``
        """
        if "R" not in self.columns:
            self.calc_R()
        self["R/RT"] = self["R"].cumsum()/self["R"].sum()

    def calc_VROT(self, method: str = "pythonbody", sign_nbody: bool = True):
        """
        calculate rotational velocity.

        | Required columns: ``X1``, ``X2``, ``X3``, ``V1``, ``V2``, ``V3``
        | Intermediate columns: ``R``
        | Output columns: ``VROT``
        
        :param method: Optional which method to use, either standard ``nbody`` way or
            the ``pythonbody`` way, which does not require the system to rotate
            along the z-axis.
        :type method: str ["pythonbody" or "nbody"]
        
        :param sign_nbody: Use the sign convention for VROT from nbody
        :type sign_nbody: bool
        """

        if "R" not in self.columns:
            self.calc_R()

        if method == "pythonbody":
            return self._calc_VROT_pythonbody(sign_nbody)
        elif method == "nbody":
            return self._calc_VROT_nbody(sign_nbody)
        else:
            raise ValueError(f"Method must be either 'nbody' or 'pythonbody' but is {method}")
        
    def _calc_VROT_nbody(self, sign_nbody):
        rvxy = self["X1"]*self["V1"] + self["X2"] * self["V2"]
        rxy2 = self["X1"]**2 + self["X2"]**2
        vrot1 = self["V1"] - rvxy * self["X1"]/rxy2
        vrot2 = self["V2"] - rvxy * self["X2"]/rxy2
        self["VROT"] = np.sqrt(vrot1**2 + vrot2**2)
        if sign_nbody:
            mask = (vrot1*self["X2"] - vrot2*self["X1"]) < 0
            self.cluster_data.loc[mask,"VROT"] = - self.cluster_data.loc[mask,"VROT"]

    def _calc_VROT_pythonbody(self, sign_nbody):
        VROT = np.cross( 
                    np.cross(
                        self.loc[:,["X1","X2","X3"]],
                        self.loc[:,["V1","V2","V3"]]
                    )/(self["R"]**2).values.reshape(self.shape[0],1),
                    self.loc[:,["X1","X2","X3"]]
                )
        self["VROTX"] = VROT[:,0]
        self["VROTY"] = VROT[:,1]
        self["VROTZ"] = VROT[:,2]
        XSIGN = 1
        if sign_nbody:
            XSIGN = np.sign(VROT[:,0]*self["X2"]/np.sqrt(self["X1"]**2 + self["X2"]**2) \
                                - VROT[:,1]*self["X1"]/np.sqrt(self["X1"]**2 + self["X2"]**2))
        self["VROT"] = XSIGN * np.linalg.norm(VROT, axis=1)

    def calc_VROT_CUMMEAN(self):
        if "VROT" not in self.columns:
            self.calc_VROT()

        self["VROT_CUMMEAN"] = ffi.cummean(self["VROT"].values)
