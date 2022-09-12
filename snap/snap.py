import pandas as pd
import numpy as np
import glob
import logging
import h5py
import sys
from tqdm import tqdm
import re
import pathlib

#from pythonbody.utils import cummean
from pythonbody.ffi import ffi
from pythonbody.nbdf import nbdf
from pythonbody.snap.binaries import binaries
from pythonbody.snap.singles import singles
        
class snap(pd.DataFrame):
    def __init__(self, data_path = None):
        super().__init__(columns=["time","file","step"])
        if not pathlib.Path(data_path).is_dir():
            raise IOError(f"Couldn't find {data_path}. Does it exist?")
        self.data_path = data_path

        
        self.files = None
        self.time = None
        if self.data_path is not None:
            self.files = sorted(glob.glob(self.data_path + "/snap*"))
            self._load_files()

        self.cluster_data = None
        self.binary_data = None 
        self.singles_data = None
        self.time_evolution_data = None

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded cluster data, otherwise returns snap list data
        """
        if type(value) != list:
            value = [value]

        missing_list = []
        if self.cluster_data is not None:
            for val in value:
                if val not in self.cluster_data.columns:
                    missing_list += [val]
            
            if len(missing_list) == 0:
                return self.cluster_data[value]
            elif len(missing_list) > 0 and np.sum([f"calc_{val}".replace("/","_over_") not in dir(self) for val in missing_list]) == 0:
                for missing in missing_list:
                    if missing not in self.cluster_data.columns:
                        eval(f"self.calc_{missing}()".replace("/","_over_"))
                return self.cluster_data[value]
            else:
                return super().__getitem__(value)
        else:
            return super().__getitem__(value)

    """def __repr__(self):
        if self.cluster_data is not None:
            return self.cluster_data.__repr__()
        else:
            super().__repr__()"""

    @property
    def reduced(self):
        if self.shape == (0,3):
            self._load_files()
        return self[self["time"] == self["time"].values.astype(int)]

    @property
    def binaries(self, t=0):
        if self.binary_data is None:
            self.load_cluster(t)
        return self.binary_data
    @property
    def singles(self, t=0):
        if self.singles_data is None:
            self.load_cluster(t)
        return self.singles_data
    @property
    def time_evolution(self):
        if self.time_evolution_data is None:
            self.calc_time_evolution_data()
        return self.time_evolution_data

    def calculate_time_evolution(self, RLAGRS=None):
        if RLAGRS is None:
            RLAGRS = [0.001,
                     0.003,
                     0.005,
                     0.01,
                     0.03,
                     0.05,
                     0.1,
                     0.2,
                     0.3,
                     0.4,
                     0.5,
                     0.6,
                     0.7,
                     0.8,
                     0.9,
                     0.95,
                     0.99,
                     1.0]
        self.time_evolution_data = {
                "RLAGR_BH": nbdf(),
                "E": nbdf(),
                }
        for i,r in tqdm(self.iterrows(), total=self.shape[0]):
            self.load_cluster(i)
            self.calc_R()
            self.calc_M_over_MT()
            self.singles.calc_R()
            self.singles.calc_M_over_MT()
            self.binaries.calc_Eb()
            for rlagr in RLAGRS:
                self.time_evolution_data["RLAGR_BH"].loc[i,str(rlagr)] = float(self.singles.filter("BH")[self.singles.filter("BH")["M/MT"] < rlagr]["R"].max())
                self.time_evolution_data["E"].loc[i,"BH-BH_N"] = self.binaries.filter("BH-BH").shape[0]
                self.time_evolution_data["E"].loc[i,"BH-BH_Eb_tot"] = self.binaries.filter("BH-BH")["Eb"].sum()
                self.time_evolution_data["E"].loc[i,"BH-BH_Eb_mean"] = self.binaries.filter("BH-BH")["Eb"].mean()
                self.time_evolution_data["E"].loc[i,"BH-BH_Eb_std"] = self.binaries.filter("BH-BH")["Eb"].std()


    def load_cluster(self, time):
        if self.shape == (0,3):
            self._load_files()

        self.time = time
        
        add_cols = {
                    "K*": "031 KW",
                    "NAME": "032 Name",
                    "Type": "033 Type",
                   }

        f = h5py.File(self.loc[time]["file"],"r")
        self.cluster_data =  pd.DataFrame({
            "M":  f["Step#" + self.loc[time]["step"]]["023 M"],
            "X1": f["Step#" + self.loc[time]["step"]]["001 X1"],
            "X2": f["Step#" + self.loc[time]["step"]]["002 X2"],
            "X3": f["Step#" + self.loc[time]["step"]]["003 X3"],
            "V1": f["Step#" + self.loc[time]["step"]]["004 V1"],
            "V2": f["Step#" + self.loc[time]["step"]]["005 V2"],
            "V3": f["Step#" + self.loc[time]["step"]]["006 V3"],
            })
        for col in add_cols.keys():
            if add_cols[col] in f["Step#" + self.loc[time]["step"]].keys():
                self.cluster_data[col] = f["Step#" + self.loc[time]["step"]][add_cols[col]][:]
        
        self.binary_data =  binaries({
            "M1": f["Step#" + self.loc[time]["step"]]["123 Bin M1*"],
            "M2": f["Step#" + self.loc[time]["step"]]["124 Bin M2*"],
            "cmX1": f["Step#" + self.loc[time]["step"]]["101 Bin cm X1"],
            "cmX2": f["Step#" + self.loc[time]["step"]]["102 Bin cm X2"],
            "cmX3": f["Step#" + self.loc[time]["step"]]["103 Bin cm X3"],
            "cmV1": f["Step#" + self.loc[time]["step"]]["104 Bin cm V1"],
            "cmV2": f["Step#" + self.loc[time]["step"]]["105 Bin cm V2"],
            "cmV3": f["Step#" + self.loc[time]["step"]]["106 Bin cm V3"],
            "relX1": f["Step#" + self.loc[time]["step"]]["125 Bin rel X1"],
            "relX2": f["Step#" + self.loc[time]["step"]]["126 Bin rel X2"],
            "relX3": f["Step#" + self.loc[time]["step"]]["127 Bin rel X3"],
            "relV1": f["Step#" + self.loc[time]["step"]]["128 Bin rel V1"],
            "relV2": f["Step#" + self.loc[time]["step"]]["129 Bin rel V2"],
            "relV3": f["Step#" + self.loc[time]["step"]]["130 Bin rel V3"],
            "K*1": f["Step#" + self.loc[time]["step"]]["158 Bin KW1"],
            "K*2": f["Step#" + self.loc[time]["step"]]["159 Bin KW2"],
            "NAME1": f["Step#" + self.loc[time]["step"]]["161 Bin Name1"],
            "NAME2": f["Step#" + self.loc[time]["step"]]["162 Bin Name2"], 
            })

        self.singles_data = singles(self.cluster_data, self.binary_data)

        return self.cluster_data

    def calc_spherical_coords(self):
        """
        calculates spherical coordinates from cartesian ones.
        See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        """

        self.cluster_data["R"] = np.sqrt(self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2 + self.cluster_data["X3"]**2)
        self.cluster_data["THETA"] = np.arccos(self.cluster_data["X3"]/self.cluster_data["R"])
        
        mask = self.cluster_data["X1"] > 0
        self.cluster_data.loc[mask,"PHI"] = np.arctan(self.cluster_data.loc[mask,"X2"]/self.cluster_data.loc[mask,"X1"])
        mask = (self.cluster_data["X1"] < 0) & (self.cluster_data["X2"] >= 0)
        self.cluster_data.loc[mask,"PHI"] = np.arctan(self.cluster_data.loc[mask,"X2"]/self.cluster_data.loc[mask,"X1"]) + np.pi
        mask = (self.cluster_data["X1"] < 0) & (self.cluster_data["X2"] < 0)
        self.cluster_data.loc[mask,"PHI"] = np.arctan(self.cluster_data.loc[mask,"X2"]/self.cluster_data.loc[mask,"X1"]) - np.pi
        mask = (self.cluster_data["X1"] == 0) & (self.cluster_data["X2"] > 0)
        self.cluster_data.loc[mask,"PHI"] = np.pi/2
        mask = (self.cluster_data["X1"] == 0) & (self.cluster_data["X2"] < 0)
        self.cluster_data.loc[mask,"PHI"] = -np.pi/2
        self.cluster_data["R/Rt"] = self.cluster_data["R"]/self.cluster_data["R"].max()

        self.cluster_data.sort_values("R",ignore_index=True,inplace=True)

    def calc_R(self):
        return self.calc_spherical_coords()
    def calc_THETA(self):
        return self.calc_spherical_coords()
    def calc_PHI(self):
        return self.calc_spherical_coords()

    def calc_M_over_MT(self):
        if not "R" in self.cluster_data.columns:
            self.calc_R()

        self.cluster_data["M/MT"] = self.cluster_data["M"].cumsum()

    def calc_VROT(self):
        if not "R" in self.cluster_data.columns:
            self.calc_R()

        """rvxy = self.cluster_data["X1"]*self.cluster_data["V1"] + self.cluster_data["X2"]*self.cluster_data["V2"]
        rxy2 = self.cluster_data["X1"] **2 + self.cluster_data["X2"]**2
        vrot1 = self.cluster_data["V1"] - rvxy* self.cluster_data["X1"]/rxy2
        vrot2 = self.cluster_data["V2"] - rvxy* self.cluster_data["X2"]/rxy2
        xsign = np.sign(vrot1*self.cluster_data["X2"]-vrot2*self.cluster_data["X1"])
        self.cluster_data["VROT"] = xsign*np.sqrt(vrot1**2+vrot2**2)"""

        """RR12 = self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2
        XR12 = (self.cluster_data["V1"] * (self.cluster_data["X1"] - self.RDENS[0])) + \
                (self.cluster_data["V2"] * (self.cluster_data["X2"] -  self.RDENS[1]))

        VROT1 = self.cluster_data["V1"] - XR12/RR12 * self.cluster_data["X1"]
        VROT2 = self.cluster_data["V2"] - XR12/RR12 * self.cluster_data["X2"]

        VROTM = np.sqrt(VROT1**2 + VROT2**2)
        XSIGN = np.sign(VROT1*self.cluster_data["X2"]/np.sqrt(RR12) - VROT2*self.cluster_data["X1"]/np.sqrt(RR12))

        self.cluster_data["VROT"] = XSIGN*self.cluster_data["M"]*VROTM"""
        VROT = np.cross( 
                    np.cross(
                        self.cluster_data.loc[:,["X1","X2","X3"]],
                        self.cluster_data.loc[:,["V1","V2","V3"]]
                    )/(self.cluster_data["R"]**2).values.reshape(self.cluster_data.shape[0],1),
                    self.cluster_data.loc[:,["X1","X2","X3"]]
                )
        XSIGN = np.sign(VROT[:,0]*self.cluster_data["X2"]/np.sqrt(self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2) \
                            - VROT[:,1]*self.cluster_data["X1"]/np.sqrt(self.cluster_data["X1"]**2 + self.cluster_data["X2"]**2))
        self.cluster_data["VROT"] = XSIGN * np.sqrt(VROT[:,0]**2 + VROT[:,1]**2)


        """self.cluster_data["VROT"] = np.cross(
                self.cluster_data.loc[:,["X1","X2","X3"]],
                self.cluster_data.loc[:,["V1","V2","V3"]]
            )[:,2]/(self.cluster_data["R"]**2)"""

        
        """
        self.cluster_data["VROT"] = np.linalg.norm(np.cross(
                self.cluster_data.loc[:,["X1","X2","X3"]],
                self.cluster_data.loc[:,["V1","V2","V3"]]
            ), axis=1)/(self.cluster_data["R"]**2)
        """

    def calc_VROT_CUMMEAN(self):
        if not "VROT" in self.cluster_data.columns:
            self.calc_VROT()

        self.cluster_data["VROT_CUMMEAN"] = ffi.cummean(self.cluster_data["VROT"].values)
        #self.cluster_data["VROT_CUMMEAN"] = self.cluster_data["VROT"]


    def _load_files(self):
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0
        for file in self.files:
            # if else clause should do the same, but kepler complains about setting new row with loc
            # newer python versions complain about appending... so either way...
            # TODO: change python ver to pandas ver
            if sys.version_info.minor >= 10:
                self.loc[float(file[file.rfind("/")+1:].replace("snap.40_","").replace(".h5part",""))] = [float(file[file.rfind("/") + 1:].replace("snap.40_","").replace(".h5part","")),file, "0"]
            else:
                super().__init__(self.append({"time": float(file[file.rfind("/")+1:].replace("snap.40_","").replace(".h5part","")),
                                              "file": file,
                                              "step": "0"}, ignore_index=True))
                self.index = self["time"].values
        self.sort_index(inplace=True)

    def _analyze_files(self):
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0

        super().__init__(columns=["time","file","step"])

        for file in tqdm(self.files):
            f = h5py.File(file,"r")

            for step in f.keys():

                # if else clause should do the same, but kepler complains about setting new row with loc
                # newer python versions complain about appending... so either way...
                # TODO: change python ver to pandas ver
                if sys.version_info.minor >= 10:
                    self.loc[f[step]['000 Scalars'][0]] = [f[step]['000 Scalars'][0],file, step.replace("Step#","")]
                else:
                    super().__init__(self.append({"time": f[step]['000 Scalars'][0],
                                                  "file": file,
                                                  "step": step.replace("Step#","")}, ignore_index=True))
                    self.index = self["time"].values
            f.close() 
