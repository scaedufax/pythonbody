import pandas as pd
import numpy as np
import glob
import logging
import h5py
import sys
from tqdm import tqdm
import re

from pythonbody.utils import cummean
        
import time

class snap(pd.DataFrame):
    def __init__(self, data_path = None):
        super().__init__(columns=["time","file","step"])
        self.data_path = data_path
        
        self.files = None
        if self.data_path is not None:
            self.files = sorted(glob.glob(self.data_path + "/snap*"))
            self._load_files()

        self.cluster_data = None;
            
        #if self.files is not None:
        #    self._load_files()
        #    self.sort_values("time",inplace=True)

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded cluster data, otherwise returns snap list data
        """
        if type(value) != list:
            value = [value]

        missing_list = []
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


    @property
    def reduced(self):
        if self.shape == (0,3):
            self._load_files()
        return self[self["time"] == self["time"].values.astype(int)]

    def load_cluster(self, time):
        if self.shape == (0,3):
            self._load_files()

        f = h5py.File(self.loc[time]["file"],"r")
        self.cluster_data =  pd.DataFrame({
            "M":  f["Step#" + self.loc[time]["step"]]["023 M"][:],
            "X1": f["Step#" + self.loc[time]["step"]]["001 X1"][:],
            "X2": f["Step#" + self.loc[time]["step"]]["002 X2"][:],
            "X3": f["Step#" + self.loc[time]["step"]]["003 X3"][:],
            "V1": f["Step#" + self.loc[time]["step"]]["004 V1"][:],
            "V2": f["Step#" + self.loc[time]["step"]]["005 V2"][:],
            "V3": f["Step#" + self.loc[time]["step"]]["006 V3"][:],
            })
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

        """self.cluster_data["VROT"] = np.cross(
                self.cluster_data.loc[:,["X1","X2","X3"]],
                self.cluster_data.loc[:,["V1","V2","V3"]]
            )[:,2]/(self.cluster_data["R"]**2)"""
        self.cluster_data["VROT"] = np.linalg.norm(np.cross(
                self.cluster_data.loc[:,["X1","X2","X3"]],
                self.cluster_data.loc[:,["V1","V2","V3"]]
            ), axis=1)/(self.cluster_data["R"]**2)

    def calc_VROT_CUMMEAN(self):
        if not "VROT" in self.cluster_data.columns:
            self.calc_VROT()

        self.cluster_data["VROT_CUMMEAN"] = cummean(self.cluster_data["VROT"].values)


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
