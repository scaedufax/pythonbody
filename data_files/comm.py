import pandas as pd
import numpy as np
import pathlib
import glob
import struct

from ..nbdf import nbdf


class comm():
    """
    Base class for handling comm.2_* files
    
    :param data_path: path to (conf) data, usually just the nbody project run.
    :type data_path: str or None

    Usage
    -----

    .. code-block:: python
        
        >>> from pythonbody.data_files import comm
        # load entire project
        >>> c = comm("/path/to/nbody/run")
        # look at available files
        >>> c.files
              time                                            file 
        0.0    0.0   /run/media/uli/ULIEXT/nbody_runs/N1k/comm.2_0 
        10.0  10.0  /run/media/uli/ULIEXT/nbody_runs/N1k/comm.2_10 
        20.0  20.0  /run/media/uli/ULIEXT/nbody_runs/N1k/comm.2_20 
        30.0  30.0  /run/media/uli/ULIEXT/nbody_runs/N1k/comm.2_30
        >>> c.load(10) # loads comm.2_10
        # Alternatively load just a file
        >>> c = conf("/path/to/nbody/run/comm.2_0")
        >>> t.scalars # show scalar values
        ...
        >>> t.data # show cluster data
                   X1        X2        X3      X0_1      X0_2      X0_3      V0_1      V0_2      V0_3        V1        V2        V3        F1        F2        F3         M 
        0   -0.073043  0.484168 -0.481146 -0.073043  0.484168 -0.481146  0.154465 -0.826052 -0.629742  0.154465 -0.826052 -0.629742  0.048468 -0.164268  0.289745  0.086948 
        1   -0.373770  0.219240  0.614535 -0.373770  0.219240  0.614535  0.433374  0.202484  0.236858  0.433374  0.202484  0.236858  0.444630 -0.071017 -0.244033  0.044023 
        2   -2.536047 -0.354711  0.131514 -2.536047 -0.354711  0.131514  0.002954 -0.209583  0.420556  0.002954 -0.209583  0.420556  0.062484  0.015900 -0.003916  0.032648 
        3   -0.244546  0.179264  0.205558 -0.244546  0.179264  0.205558  0.443664  1.051663  0.061355  0.443664  1.051663  0.061355  0.473523 -0.138025 -0.122024  0.022738 
        4   -0.076585 -0.395928  0.246906 -0.076585 -0.395928  0.246906  0.256862  0.116328 -0.392418  0.256862  0.116328 -0.392418  0.203803  0.691925  0.017607  0.019321 
        ..        ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ... 
        995  0.473128  0.365016  0.460594  0.473128  0.365016  0.460594 -0.475344  0.367096 -0.185623 -0.475344  0.367096 -0.185623 -0.112853 -0.172985 -0.084799  0.000134 
        996  0.394393  0.888119  0.134303  0.394393  0.888119  0.134303  0.026954 -0.162636 -0.209924  0.026954 -0.162636 -0.209924 -0.159191 -0.315613 -0.075223  0.000134 
        997 -0.335770 -0.674808  0.613585 -0.335770 -0.674808  0.613585 -0.527224  0.290886  0.010729 -0.527224  0.290886  0.010729  0.152269  0.227155 -0.173811  0.000134 
        998 -0.108430  0.059100  0.258102 -0.108430  0.059100  0.258102  0.517059  0.226135 -0.205559  0.517059  0.226135 -0.205559 -0.053115 -0.116155 -0.431076  0.000134 
        999 -0.104434 -0.940669 -0.650883 -0.104434 -0.940669 -0.650883  0.480972  0.098430  0.040469  0.480972  0.098430  0.040469  0.000723  0.159018  0.147055  0.000133

    """
    AUTO_LOAD = False
    def __init__(self, data_path):
        self._files = pd.DataFrame(columns=["time", "file"])
        if data_path is not None and not pathlib.Path(data_path).is_dir() and not pathlib.Path(data_path).is_file():
            raise IOError(f"Couldn't find {data_path}. Does it exist?")
        self.data_path = data_path

        self.time = None
        
        if self.data_path is not None:
            files = sorted(glob.glob(self.data_path + "/comm.2_*"))
            time = [float(file[file.rfind("/")+1:].replace("comm.2_", "")) for file in files]
            self._files["file"] = files
            self._files["time"] = time
            self._files.index = time
            self._files = self.files.sort_index()

        # mapping of integer values at beginning
        self._imap = {
                4: "NMAX",
                8: "KMAX",
                12: "LMAX",
                16: "MMAX",
                20: "MLD",
                24: "MLR",
                28: "MLV",
                32: "MCL",
                36: "NCMAX",
                40: "NTMAX",
                52: "ntot",
                56: "npairs",
                60: "nttot",
                }
        self._fmap = None
        self._byte_map = None 

        self._data_offset = 16

        self._comm_data = None
        self._comm_scalars = None
        
        if pathlib.Path(self.data_path).is_file():
            self._load_file(file=self.data_path)

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded cluster data,
        otherwise returns snap list data
        """
        try:
            return self._comm_data[value]
        except:
            pass
        try:
            return self._comm_scalars[value]
        except:
            pass
        try:
            return self._files[value]
        except:
            pass

        raise ValueError(f"Couldn't find data for {value}")

    def __repr__(self):
        if self._comm_data is not None:
            return self._comm_data.__repr__()
        return self._files.__repr__()
    
    def _repr_html_(self):
        if self._comm_data is not None:
            return self._comm_data._repr_html_()
        return self._files._repr_html_()

    @property
    def data(self):
        return self._comm_data
    
    @property
    def scalars(self):
        return self._comm_scalars

    @property
    def files(self):
        return self._files

    def _gen_fmap(self, f_offset: int = 64):
        if self.scalars is None:
            raise KeyError("scalars are not available! did you load some data?")
        if self._fmap is None:
            self._fmap = {}

        self._foutput = [
                ["ia", "i", 85],
                ["b", "f", 168],
                ["c", "f", 530],
                ["d", "f", 381 + self.scalars["MLR"] + self.scalars["MLD"] + self.scalars["MLV"]],
                ["e","f",24],
                ["g", "f", 132],
                ["l", "f", 99],
                ["m", "f", 40],
                ["o", "f", 20 * self.scalars["MCL"] + 16],
                ["p", "f", 32 * self.scalars["NTMAX"]],
                ["q", "f", 31 * self.scalars["MMAX"]],
                ["s", "f", 44 * self.scalars["MMAX"]],
                ]
        for i, d in enumerate(self._foutput):
            current_offset = 0
            for j, dj in enumerate(self._foutput[:i]):
                current_offset += dj[2]*4
            self._fmap[f_offset + current_offset] = d

    def _load_file(self, file: str, time: float = None):
        if time is not None:
            self.time = time
        else:
            self.time = float(file[file.rfind("/")+1:].replace("comm.2_", ""))

        self._comm_scalars = {}

        data = None
        with open(file, "rb") as comm_file:
            data = comm_file.read()

        for scalar in self._imap.keys():
            val = struct.unpack("i", data[scalar:scalar+1*4])[0]
            self._comm_scalars[self._imap[scalar]] = val

        self._gen_fmap()
        
        for f in self._fmap.keys():
            val = np.array(struct.unpack(self._fmap[f][1]*self._fmap[f][2], data[f: f + self._fmap[f][2]*4]))
            self._comm_scalars[self._fmap[f][0]] = val
                

        self._data_offset =  ( 16 + 4 # unknown bytes 0-4, 44-52 (Between write statements), maybe also after the b,c,..,q,s
                              + 4*13
                              + 4*(
                                  85 # ia
                                  + 168 #b
                                  + 530 #c
                                  + 381 + self.scalars["MLR"] + self.scalars["MLD"] + self.scalars["MLV"] # d
                                  + 24 #e
                                  + 132 #g
                                  + 99 #l
                                  + 40 #m
                                  + 20 * self.scalars["MCL"] + 16 #o
                                  + 32 * self.scalars["NTMAX"] #p
                                  + 31 * self.scalars["MMAX"] #q
                                  + 44 * self.scalars["MMAX"] #s
                                  )
                              )
        ntot = self.scalars["ntot"]
        offset = self._data_offset
        self._byte_map = {
                "X123": (offset, offset + ntot*3*8),
                "X0123": (offset + 1*3*8*ntot, offset + (1+1)*3*8*ntot),
                "V0123": (offset + 2*3*8*ntot, offset + (2+1)*3*8*ntot),
                "V123": (offset + 3*3*8*ntot, offset + (3+1)*3*8*ntot),
                "F123": (offset + 4*3*8*ntot, offset + (4+1)*3*8*ntot),
                "FDOT123": (offset + 5*3*8*ntot, offset + (5+1)*3*8*ntot),
                "M": (offset + 6*3*8*ntot, offset + (6)*3*8*ntot + 1*8*ntot),
                }
        
        X123 = np.array(struct.unpack("d" * self.scalars["ntot"] * 3, data[self._byte_map["X123"][0]:self._byte_map["X123"][1]]))
        X1 = X123[0::3]
        X2 = X123[1::3]
        X3 = X123[2::3]
        
        X0123 = np.array(struct.unpack("d" * self.scalars["ntot"] * 3, data[self._byte_map["X0123"][0]:self._byte_map["X0123"][1]]))
        X01 = X0123[0::3]
        X02 = X0123[1::3]
        X03 = X0123[2::3]
        
        V0123 = np.array(struct.unpack("d" * self.scalars["ntot"] * 3, data[self._byte_map["V0123"][0]:self._byte_map["V0123"][1]]))
        V01 = V0123[0::3]
        V02 = V0123[1::3]
        V03 = V0123[2::3]
        
        V123 = np.array(struct.unpack("d" * self.scalars["ntot"] * 3, data[self._byte_map["V123"][0]:self._byte_map["V123"][1]]))
        V1 = V123[0::3]
        V2 = V123[1::3]
        V3 = V123[2::3]
        
        F123 = np.array(struct.unpack("d" * self.scalars["ntot"] * 3, data[self._byte_map["F123"][0]:self._byte_map["F123"][1]]))
        F1 = F123[0::3]
        F2 = F123[1::3]
        F3 = F123[2::3]

        M = np.array(struct.unpack("d" * ntot, data[self._byte_map["M"][0]:self._byte_map["M"][1]]))

        self._comm_data = pd.DataFrame(
                {
                    "X1": X1, "X2": X2, "X3": X3,
                    "X0_1": X01, "X0_2": X02, "X0_3": X03,
                    "V0_1": V01, "V0_2": V02, "V0_3": V03,
                    "V1": V1, "V2": V2, "V3": V3,
                    "F1": F1, "F2": F2, "F3": F3,
                    "M": M
                }
                )

        return self.data

    def load(self, time: float):
        """
        load conf file at a given time step

        :param time: time in Nbody units to load
        """
        if time not in self._files.index:
            raise KeyError(f"Couldn't find file according to time {time} see conf.files for files")
        if not pathlib.Path(self._files.loc[time,"file"]).is_file():
            raise IOError(f"Couldn't find {self._files.loc[time,'file']}. Does it exist?")

        return self._load_file(self._files.loc[time,'file'], time)

