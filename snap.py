import pandas as pd
import numpy as np
import glob
import logging
import h5py
import sys
from tqdm import tqdm
        
import time

class snap(pd.DataFrame):
    def __init__(self, data_path = None):
        super().__init__(columns=["time","file","step"])
        self.data_path = data_path
        
        self.files = None
        if self.data_path is not None:
            self.files = sorted(glob.glob(self.data_path + "/snap*"))
            
        #if self.files is not None:
        #    self._load_files()
        #    self.sort_values("time",inplace=True)

    @property
    def reduced(self):
        if self.shape == (0,3):
            self._load_files()
        return self[self["time"] == self["time"].values.astype(int)]

    def load_cluster(self, time):
        if self.shape == (0,3):
            self._load_files()

        f = h5py.File(self.loc[time]["file"],"r")
        return pd.DataFrame({
            "M":  f["Step#" + self.loc[time]["step"]]["023 M"][:],
            "X1": f["Step#" + self.loc[time]["step"]]["001 X1"][:],
            "X2": f["Step#" + self.loc[time]["step"]]["002 X2"][:],
            "X3": f["Step#" + self.loc[time]["step"]]["003 X3"][:],
            "V1": f["Step#" + self.loc[time]["step"]]["004 V1"][:],
            "V2": f["Step#" + self.loc[time]["step"]]["005 V2"][:],
            "V3": f["Step#" + self.loc[time]["step"]]["006 V3"][:],
            })

    def _load_files(self):
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0
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
