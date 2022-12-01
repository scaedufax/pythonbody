import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from pythonbody.ffi import ffi

# TODO: Do something more reasonable with G.

class dat10():
    """
    Class for reading and modifying dat.10 files
    for nbody.

    Attributes:
        com (float): center of mass
        EKIN (float): kinetic energy
        EPOT (float): potential energy
        ETOT (float) EPOT+EKIN

        ZMBAR: Calculates ZMBAR for use in nbody input files
        RBAR: Calculates RBAR for use in nbody input files
        AVMASS: prints average Mass of system

    """
    def __init__(self, file_path: str = None, G=1):
        """
        dat10 constructor
        Parameters:
            file_path (str): path to dat.10 file
        """
        self._setup_logger()
        self._data = None
        self._com = None

        self._ETOT = None
        self._EKIN = None
        self._EPOT = None

        self._default_cols = ["M", "X1", "X2", "X3", "V1", "V2", "V3"]
        self._G = G

        self._L = None

        if file_path is not None:
            self.file_path = file_path
            self.load(self.file_path)

    def load(self, file_path: str):
        """
        function to load data from dat.10 file

        Parameters:
            file_path (str): path to dat.10 file
        """
        self._data = pd.read_csv(
                file_path,
                header=None,
                index_col=False,
                #delimiter=" ",
                delim_whitespace=True,
                usecols=range(0,7),
                names=self._default_cols
                )
    def save(self, file: str):
        """
        save (modified) dat.10 file

        Will only save the relevant cols for nbody, not any other cols you've
        created

        Parameters:
            file (str): path to file to save output to.
        """
        self._data[self._default_cols].to_csv(
            file,
            sep=" ",
            header=False,
            index=False,
            float_format="%.6f"
        )
    
    def __repr__(self):
        return self._data.__repr__()

    def _repr_html_(self):
        return self._data._repr_html_()

    @property
    def loc(self):
        return self._data.loc

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
            self._data["EKIN"] = 1/2 * self.loc[:,"M"]*np.linalg.norm(self.loc[:, ["V1", "V2", "V3"]], axis=1)**2
            self._EKIN = self._data["EKIN"].sum()
        return self._EKIN
    
    @property
    def EPOT(self, G=1):
        """if self._EPOT is None:
            self._data["EPOT"] = self._G * grav_pot(self._data[["M", "X1", "X2", "X3"]])
            self._EPOT = self._data["EPOT"].sum()
        return self._EPOT"""
        return np.sum(self._G * ffi.grav_pot(self._data[["M", "X1", "X2", "X3"]]))

    @property
    def ZMBAR(self):
        return self._data.loc[:, "M"].mean()
    
    @property
    def RBAR(self):
        return np.linalg.norm(self.loc[:, ["X1", "X2", "X3"]], axis=1).mean()
    
    @property
    def COM(self):
        if self._com is None:
            self._com = 1/self.loc[:, "M"].sum() * np.sum(self.loc[:, ["X1", "X2", "X3"]].multiply(self.loc[:, "M"], axis=0))
        return self._com
    
    @property
    def AVMASS(self):
        return self._data.loc[:, "M"].mean()   
        
    def __setitem__(self, key, item):
        self._data[key] = item

    def __getitem__(self, key):
        #if type(key) == int, list]:
        #    return self._data.iloc[:,key]
        #else:
        return self._data[key]

    @property
    def L(self):
        if self._L is None:
            self._calc_L()
        return self._L

    def _calc_L(self):
        R = self[["X1","X2","X3"]] - self.COM
        L = np.cross(R, self[["V1","V2","V3"]].values * self[["M"]].values)
        self._L = L.mean(axis=0)
        self._data["Lx"] = L[:,0]
        self._data["Ly"] = L[:,1]
        self._data["Lz"] = L[:,2]
        self._data["L"] = np.linalg.norm(L,axis=1)
        pass


    def _check_non_empty(self):
        if not self._data:
            self.logger.warning("Data is empty, have you loaded the file?")
    
    def rotate(self, yaw: float = 0, pitch: float = 0, roll: float = 0):
        """
        Rotates position (and velocities if available)

        Parameters:
            yaw (float): yaw in radians
            pitch (float): pitch in radians
            roll (float): roll in radians
        """

        # initialize rotation matrices
        rot_mat_yaw = np.zeros((3, 3))
        rot_mat_pitch = np.zeros((3, 3))
        rot_mat_roll = np.zeros((3, 3))

        rot_mat_yaw[0][0] = np.cos(yaw)
        rot_mat_yaw[0][1] = -np.sin(yaw)
        rot_mat_yaw[1][0] = np.sin(yaw)
        rot_mat_yaw[1][1] = np.cos(yaw)
        rot_mat_yaw[2][2] = 1

        rot_mat_pitch[0][0] = np.cos(pitch)
        rot_mat_pitch[0][2] = np.sin(pitch)
        rot_mat_pitch[1][1] = 1
        rot_mat_pitch[2][0] = -np.sin(pitch)
        rot_mat_pitch[2][2] = np.cos(pitch)

        rot_mat_roll[0][0] = 1
        rot_mat_roll[1][1] = np.cos()
        rot_mat_roll[1][2] = -np.sin(roll)
        rot_mat_roll[2][1] = np.sin(roll)
        rot_mat_roll[2][2] = np.cos(roll)

        rot_mat = np.matmul(np.matmul(rot_mat_yaw, rot_mat_pitch), rot_mat_roll)

        self[["V1", "V2", "V3"]] = rot_mat.dot(self[["V1", "V2", "V3"]].T.values).T
        # rotate positions
        self[["X1", "X2", "X3"]] = rot_mat.dot(self[["X1", "X2", "X3"]].T.values).T
        self._L = None

    def rotate_axis(self, axis: np.array, angle: float):
        """
        Rotates position (and velocities if available) around a given axis by angle

        Parameters:
            axis (np.array, list): axis to rotate around, must contain 3 values
            angle (float): angle to rotate by
        """
        if type(axis) == list:
            axis = np.array(axis)
        if axis.shape != (3,):
            raise ValueError(f"Axis must have shape (3,) but is {axis.shape}")

        # initialize rotation matrix
        rot_mat = np.zeros((3,3))

        # see e.g. https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        rot_mat[0][0] = np.cos(angle) + axis[0]**2*(1-np.cos(angle))
        rot_mat[0][1] = axis[0]*axis[1]*(1-np.cos(angle))-axis[2]*np.sin(angle)
        rot_mat[0][2] = axis[0]*axis[2]*(1-np.cos(angle))+axis[1]*np.sin(angle)
        rot_mat[1][0] = axis[1]*axis[0]*(1-np.cos(angle))+axis[2]*np.sin(angle)
        rot_mat[1][1] = np.cos(angle)+axis[1]**2*(1-np.cos(angle))
        rot_mat[1][2] = axis[1]*axis[2]*(1-np.cos(angle))-axis[0]*np.sin(angle)
        rot_mat[2][0] = axis[2]*axis[0]*(1-np.cos(angle))-axis[1]*np.sin(angle)
        rot_mat[2][1] = axis[2]*axis[1]*(1-np.cos(angle))+axis[0]*np.sin(angle)
        rot_mat[2][2] = np.cos(angle)+axis[2]**2*(1-np.cos(angle))
        
        # rotate velocities
        self[["V1", "V2", "V3"]] = rot_mat.dot(self[["V1", "V2", "V3"]].T.values).T
        # rotate positions
        self[["X1", "X2", "X3"]] = rot_mat.dot(self[["X1", "X2", "X3"]].T.values).T
        self._L = None

    def rotate_angular_momentum_along_zaxis(self):
        """
        Rotates position and velocities such that the angular momentum is
        aligned with z-axis
        """

        momentum = self.L
        # calculate normal between angular momentum and z-axis.
        normal = np.cross(self.L, (0, 0, 1))

        # normalize angular momentum and normal
        momentum = momentum/np.linalg.norm(momentum)
        normal = normal/np.linalg.norm(normal)

        # calculate angle to rotate along normal by
        angle = np.arccos(momentum.dot((0, 0, 1)))

        self.rotate_axis(normal, angle)

    def len(self):
        return self._data.shape[0]
     
    def adjust_com(self):
        """
        Changes positions into center of mass system
        """
        self._data.loc[:, ["X1", "X2", "X3"]] = self._data.loc[:, ["X1", "X2", "X3"]] - self.COM
        self._com = None
    
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(10)
        handler = logging.StreamHandler()
        handler.setLevel(10)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        if len(self.logger.handlers) == 0:
            self.logger.addHandler(handler)

