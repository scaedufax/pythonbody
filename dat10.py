import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from pythonbody.utils import grav_pot

class dat10():
    def __init__(self, file_path: str = None, G = 1):
        self._setup_logger()
        self._data = None
        self._com = None

        self._ETOT = None
        self._EKIN = None
        self._EPOT = None

        self._default_cols = ["M", "X1", "X2", "X3", "V1", "V2", "V3"]
        self._G = G

        if file_path is not None:
            self.file_path = file_path
            self.load(self.file_path)

    def load(self, file: str):
        self._data = pd.read_csv(
                file,
                header=None,
                index_col=False,
                delimiter=" ",
                names=self._default_cols
                )
    def save(self, file:str):
        self._data[self._default_cols].to_csv(
            file,
            sep = " ",
            header = False,
            index = False,
            float_format="%.6f"
        )
    
    def __repr__(self):
        return self._data.__repr__()

    @property
    def iloc(self):
        return self._data.iloc
    @property
    def shape(self):
        return self._data.shape
    @property
    def index(self):
        return self._data.index
    @index.setter
    def index(self, idx):
        self._data.index = idx

    @property
    def ETOT(self):
        if self._ETOT is None:
            self._ETOT = self.EKIN + self.EPOT
        return self._ETOT
    @property
    def EKIN(self):
        if self._EKIN is None:
            self._data["EKIN"] = 1/2 * self.iloc[:,0]*np.linalg.norm(self.iloc[:,4:7],axis=1)**2
            self._EKIN = self._data["EKIN"].sum()
        return self._EKIN
    @property
    def EPOT(self, G=1):
        #if self._EPOT is None:
        #    self._data["EPOT"] = self._G * grav_pot(self._data[["M", "X1", "X2", "X3"]])
        #    self._EPOT = self._data["EPOT"].sum()
        #return self._EPOT
        return np.sum(self._G * grav_pot(self._data[["M", "X1", "X2", "X3"]]))

    @property
    def ZMBAR(self):
        return self._data.iloc[:,0].mean() 
    @property
    def RBAR(self):
        return np.linalg.norm(self.iloc[:,1:4],axis=1).mean()
        
    def __setitem__(self, key, item):
        self._data[key] = item

    def __getitem__(self, key):
        if type(key) in [int, list]:
            return self._data.iloc[:,key]
        else:
            return self._data[key]

    def _check_non_empty(self):
        if not self._data:
            self.logger.warning("Data is empty, have you loaded the file?")
            
    def len(self):
        return self._data.shape[0]
    
    @property
    def com(self):
        if self._com is None:
            self._com = 1/self.iloc[:,0].sum() * np.sum(self.iloc[:,[1,2,3]].multiply(self.iloc[:,0],axis=0))
        return self._com
    
    def adjust_com(self):        
        self._data.iloc[:,[1,2,3]] = self._data.iloc[:,[1,2,3]] - self.com
        self._com = None
    
    def mean_mass(self):
        return self._data.iloc[:,0].mean()   
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(10)
        handler = logging.StreamHandler()
        handler.setLevel(10)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        if len(self.logger.handlers) == 0:
            self.logger.addHandler(handler)

