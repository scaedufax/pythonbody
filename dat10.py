import pandas as pd
import numpy as np
import logging
import warnings

from .ffi import ffi
from .nbdf import nbdf



class dat10(nbdf):
    """
    Class for reading and modifying dat.10 files for nbody.
        
    :param file_path: path to dat.10 file
    :type file_path: str or None
    """

    def __init__(self, file_path: str = None, *args, **kwargs):
        warnings.filterwarnings(action='ignore', category=UserWarning)
        if file_path is not None:
            default_cols = ["M", "X1", "X2", "X3", "V1", "V2", "V3"]
            pd_df = pd.read_csv(
                file_path,
                header=None,
                index_col=False,
                delim_whitespace=True,
                usecols=range(0, 7),
                names=default_cols
                )
            super().__init__(data=pd_df, *args, **kwargs)
        self._default_cols = ["M", "X1", "X2", "X3", "V1", "V2", "V3"]

    def load(self, file_path: str):
        """
        function to load data from dat.10 file into class

        :param file_path: path to dat.10 file
        :type file_path: str
        """
        pd_df = pd.read_csv(
                file_path,
                header=None,
                index_col=False,
                delim_whitespace=True,
                usecols=range(0, 7),
                names=self._default_cols
                )
        super().__init__(data=pd_df)

    def save(self, file: str):
        """
        save (modified) dat.10 file

        Will only save the relevant cols for nbody, not any other cols you've
        created

        :param file: path to file to save output to.
        :type file: str
        """
        self[self._default_cols].to_csv(
            file,
            sep=" ",
            header=False,
            index=False,
            float_format="%.6f"
        )
    
    @property
    def ETOT(self):
        """
        :return: Total energy (kinetic + potential)
        :rtype: float
        """

        return self.EKIN + self.EPOT
    
    @property
    def EKIN(self):
        """
        :return: Kinetic energy
        :rtype: float
        """
        if "EKIN" not in self.columns:
            self["EKIN"] = 1/2 * self.loc[:, "M"]*np.linalg.norm(self.loc[:, ["V1", "V2", "V3"]], axis=1)**2
        return self["EKIN"].sum()
    
    @property
    def EPOT(self):
        """
        :return: Potential energy (needs to be calculated)
        :rtype: float
        """
        if "EPOT" not in self.columns:
            self.calc_EPOT()
        return self.loc[:,"EPOT"].sum()

    @property
    def ZMBAR(self):
        """
        :return: Average mass
        :rtype: float
        """
        return self.loc[:, "M"].mean()
    
    @property
    def RBAR(self):
        """
        :return: Average radius
        :rtype: float
        """
        return np.linalg.norm(self.loc[:, ["X1", "X2", "X3"]], axis=1).mean()
     
    @property
    def AVMASS(self):
        """
        :return: Average Mass
        :rtype: float
        """
        return self.loc[:, "M"].mean()   
    
    @property
    def L(self):
        """
        :return: Angular momentum vector
        :rtype: float[3]
        """
        if "L" not in self.columns:
            self.calc_L()
        return self.loc[:, ["LX", "LY", "LZ"]].mean()
    
    @property
    def L_norm(self):
        """
        :return: Normalized (to 1) angular momentum vector
        :rtype: float[3]
        """
        return self.L/np.linalg.norm(self.L)
 
    def rotate(self, yaw: float = 0, pitch: float = 0, roll: float = 0):
        """
        Rotates positions (and velocities if available) by yaw, pitch and
        roll.

        :param yaw: yaw in radians
        :type yaw: float
        :param pitch: pitch in radians
        :type pitch: float
        :param roll: roll in radians
        :type roll: float
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

        # drop Angular Momentum after rotation
        for i in ["LX", "LY", "LZ", "L"]: self.drop(i,axis=1,inplace=True) if i in self.columns else None

    def rotate_axis(self, axis: np.array, angle: float):
        """
        Rotates position (and velocities if available) around a given `axis`
        by `angle`.
        
        :param axis: Axis to rotate around.
        :type axis: float[3]
        :param angle: Angle to rotate around axis by.
        :type angle: float
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
        # drop Angular Momentum after rotation
        for i in ["LX", "LY", "LZ", "L"]: self.drop(i,axis=1,inplace=True) if i in self.columns else None

    def rotate_angular_momentum_along_zaxis(self):
        """
        Rotates position and velocities such that the angular momentum is
        aligned with z-axis
        """

        momentum = self.L.values
        # calculate normal between angular momentum and z-axis.
        normal = np.cross(momentum, (0, 0, 1))

        # normalize angular momentum and normal
        momentum = momentum/np.linalg.norm(momentum)
        normal = normal/np.linalg.norm(normal)

        # calculate angle to rotate along normal by
        angle = np.arccos(momentum.dot((0, 0, 1)))

        print(normal,angle)

        self.rotate_axis(normal, angle)
 
    def adjust_com(self):
        """
        Changes positions into center of mass system
        """
        self.loc[:, ["X1", "X2", "X3"]] = self.loc[:, ["X1", "X2", "X3"]] - self.COM
    
    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(10)
        handler = logging.StreamHandler()
        handler.setLevel(10)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        if len(self.logger.handlers) == 0:
            self.logger.addHandler(handler)

