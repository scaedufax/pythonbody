import numpy as np
import pandas as pd
import logging
import warnings

from pythonbody.snap import snap
from pythonbody.ffi import ffi

# TODO: remove time_dependencies and load everything

class cluster(pd.DataFrame):
    """
    Stores information of a cluster for a given time step

    Attributes:
        data_path (str): path to nbody run data
        time (float): time step

    """
    def __init__(self,
            data_path: str,
            time: float = None,
            s: snap = None):
        """
        cluster constructor

        Parameters:
            data_path: path to nbody run data
            time: nbody time step to load cluster data from
            s (snap): snap module if parent object already has one
        """
        super().__init__(columns=["M", "X1", "X2", "X3", "V1", "V2", "V3"])

        # stop warning for self.snap assignments can not be done
        warnings.simplefilter(action='ignore', category=UserWarning)
        self.data_path = data_path
        self.snap = s
        self.time = None

        if time is not None:
            self.time = time
            self.load(self.time)

    def load(self, time: float):
        """
        loads cluster information for a given time step

        Parameters:
            time: time step to load data from

        """
        if self.snap is None:
            if self.data_path is None:
                logging.error("You need to specify either a snap instance or a datapath")
                return 0
            self.snap = snap(self.data_path)

        self.drop(self.index,inplace=True)
        self.time = time

        data = self.snap.load_cluster(time)
        super().__init__(data)

    @property
    def EKIN(self):
        """
        calculate kinetic energy at the current time step
        """
        if not "EKIN" in self.columns:
            self["EKIN"] = 0.5*self["M"]*(self["V1"]**2 + self["V2"]**2 + self["V3"]**2)
        return self["EKIN"].sum()

    @property
    def EPOT(self):
        """
        calculates potential energy at the current time step

        careful: depending on hardware and the size of the cluster,
        this may take some time.
        """
        if not "EPOT" in self.columns:
            self["EPOT"] = ffi.grav_pot(self)
        return self["EPOT"].sum()


