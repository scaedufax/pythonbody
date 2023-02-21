import pandas as pd
import numpy as np
import pathlib
import glob
import struct
import re
import copy

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
        >>> c = comm("/path/to/nbody/run/comm.2_0")
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
        self._byte_data = None
        
        if self.data_path is not None:
            files = sorted(glob.glob(self.data_path + "/comm.2_*"))
            files = [file for file in files if re.match(".*comm\.2_[0-9]*$", file)]
            time = [float(file[file.rfind("/")+1:].replace("comm.2_", "")) for file in files]
            self._files["file"] = files
            self._files["time"] = time
            self._files.index = time
            self._files = self.files.sort_index()
        
        # mapping of integer values at beginning
        # for later values we need the data here!
        # see _update_byte_map_with_scalars()
        self._init_value_map = [
                {"name": "fortran_header_1", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "NMAX", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "KMAX", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "LMAX","n": 1, "type": "i", "save_in": "scalars"},
                {"name": "MMAX", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "MLD", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "MLR", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "MLV", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "MCL", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NCMAX", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NTMAX", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "fortran_footer_1", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "fortran_header_2", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "ntot", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "npairs", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "nttot", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "ia", "n": 85, "type": "i", "save_in": "scalars"},
                {"name": "b", "n": 168, "type": "f", "save_in": "scalars"},
                {"name": "c", "n": 530, "type": "f", "save_in": "scalars"},
        ]
        self._value_map = copy.deepcopy(self._init_value_map)
        self._byte_map = pd.DataFrame(self._value_map)
        self._init_byte_map()

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

    def _init_byte_map(self):
        self._byte_map_data_types()
        self._byte_map.loc[:, "bytes_tot"] = pd.to_numeric(self._byte_map["bytes_data_type"] * self._byte_map["n"], downcast="signed")
        self._byte_map.loc[:, "bytes_start"] = 0
        self._byte_map.loc[:, "bytes_end"] = self._byte_map["bytes_tot"].cumsum() 
        self._byte_map.loc[:, "bytes_start"] = self._byte_map["bytes_end"] - self._byte_map["bytes_tot"]
        
    def _byte_map_data_types(self):
        self._byte_map.loc[self._byte_map["type"] == "i", "bytes_data_type"] = 4
        self._byte_map.loc[self._byte_map["type"] == "f", "bytes_data_type"] = 4
        self._byte_map.loc[self._byte_map["type"] == "d", "bytes_data_type"] = 8
        self._byte_map["bytes_data_type"] = pd.to_numeric(self._byte_map["bytes_data_type"], downcast="signed")

    def _update_byte_map_with_scalars(self):
        ntot = self.scalars["ntot"]
        npairs = self.scalars["npairs"]
        self._value_map += [
                {"name": "d", "n": 381 + self.scalars["MLR"] + self.scalars["MLD"] + self.scalars["MLV"], "type": "f", "save_in": "scalars"},
                {"name": "e", "n": 24, "type": "f", "save_in": "scalars"},
                {"name": "g", "n": 132, "type": "f", "save_in": "scalars"},
                {"name": "l", "n": 99, "type": "f", "save_in": "scalars"},
                {"name": "m", "n": 40, "type": "f", "save_in": "scalars"},
                {"name": "o", "n": 20 * self.scalars["MCL"] + 16, "type": "f", "save_in": "scalars"},
                {"name": "p", "n": 32 * self.scalars["NTMAX"], "type": "f", "save_in": "scalars"},
                {"name": "q", "n": 31 * self.scalars["MMAX"], "type": "f", "save_in": "scalars"},
                {"name": "s", "n": 44 * self.scalars["MMAX"], "type": "f", "save_in": "scalars"},
                {"name": "fortran_footer_2", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "fortran_header_3", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "X123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "X0123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "V0123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "V123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "F123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "FDOT123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "M", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "RS", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "FI123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "D1_123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "D2_123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "D3_123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "FR123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "D1R_123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "D2R_123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "D3R_123", "n": 3 * ntot, "type": "d", "save_in": "data"},
                {"name": "STEP", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "T0", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "STEPR", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "T0R", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "TIMENW", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "RADIUS", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "TEV", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "TEV0", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "BODY0", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "EPOCH", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "spin", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "xstar", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "zlmsty", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "FIDOT123", "n": 3*ntot, "type": "d", "save_in": "data"},
                {"name": "D0_123", "n": 3*ntot, "type": "d", "save_in": "data"},
                {"name": "FRDOT123", "n": 3*ntot, "type": "d", "save_in": "data"},
                {"name": "D0R_123", "n": 3*ntot, "type": "d", "save_in": "data"},
                {"name": "KSTAR", "n": ntot, "type": "i", "save_in": "data"},
                {"name": "IMINR", "n": ntot, "type": "i", "save_in": "data"},
                {"name": "NAME", "n": ntot, "type": "i", "save_in": "data"},
                {"name": "fortran_footer_3", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "fortran_header_4", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "ASPN", "n": ntot, "type": "d", "save_in": "data"},
                {"name": "fortran_footer_4", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "fortran_header_5", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "U_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "U0_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "UDOT_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "FU_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "FUDOT_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "FUDOT2_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "FUDOT3_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "H", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "HDOT", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "HDOT2", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "HDOT3", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "HDOT4", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "DTAU", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "TDOT2", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "TDOT3", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "R", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "R0", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "GAMMA", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "SF_1234567", "n": 7*npairs, "type": "d", "save_in": "pairs"},
                {"name": "H0", "n": npairs, "type": "d", "save_in": "pairs"},
                {"name": "FP0_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "FD0_1234", "n": 4*npairs, "type": "d", "save_in": "pairs"},
                {"name": "KBLIST", "n": 10*npairs, "type": "i", "save_in": "pairs"},
                {"name": "KSLOW", "n": npairs, "type": "i", "save_in": "pairs"},
                {"name": "TBLIST", "n": 1, "type": "d", "save_in": "scalars"},
                {"name": "fortran_footer_5", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "fortran_header_6", "n": 1, "type": "i", "save_in": "ignore"},
                ]
        self._byte_map = pd.DataFrame(self._value_map)
        self._init_byte_map()
        self._update_byte_map_with_list()

    def _update_byte_map_with_list(self):
        header_row_idx = self._byte_map.loc[self._byte_map["name"] == "fortran_header_6"].index.values[0]
        header_row = self._byte_map.iloc[header_row_idx].to_dict()

        total_bytes = struct.unpack("i", self._byte_data[header_row["bytes_start"]:header_row["bytes_end"]])[0]
        n_list_entries = int(total_bytes/4)
        self._value_map += [
                {"name": "LIST_DATA", "n": n_list_entries, "type": "i", "save_in": "list"},
                {"name": "fortran_footer_6", "n": 1, "type": "i", "save_in": "ignore"},
                {"name": "fortran_header_7", "n": 1, "type": "i", "save_in": "ignore"},
                ]
        self._byte_map = pd.DataFrame(self._value_map)
        self._init_byte_map()
        
        header_row_idx = self._byte_map.loc[self._byte_map["name"] == "fortran_header_7"].index.values[0]
        header_row = self._byte_map.iloc[header_row_idx].to_dict()
        nxtlimit = struct.unpack("i", self._byte_data[header_row["bytes_end"]:header_row["bytes_end"] + 4])[0]
        nghosts = struct.unpack("i", self._byte_data[header_row["bytes_end"] + 4:header_row["bytes_end"] + 8])[0]
        nlstdelay0 = struct.unpack("i", self._byte_data[header_row["bytes_end"] + 4*(70+nxtlimit+nghosts):header_row["bytes_end"] + 4*(71+nxtlimit+nghosts)])[0]
        self._value_map += [
                {"name": "NXTLIMIT", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NGHOSTS", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NXTLST", "n": nxtlimit+nghosts, "type": "i", "save_in": "ignore"},
                {"name": "NXTLEN", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NDTK", "n": 64, "type": "i", "save_in": "ignore"},
                {"name": "NDTMIN", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NDTMAX", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NXTLEVEL", "n": 1, "type": "i", "save_in": "scalars"},
                {"name": "NLSTDELAY", "n": nlstdelay0 + 1, "type": "i", "save_in": "ignore"},
                {"name": "fortran_footer_7", "n": 1, "type": "i", "save_in": "ignore"},
                ]
        self._byte_map = pd.DataFrame(self._value_map)
        self._init_byte_map()

    def _load_scalars_init(self):
        """
        loads a few scalars at first, as we later need further scalars!
        """
        if self._byte_map.shape[0] == 0:
            raise ValueError("Did you load some data?")

        for i, r in self._byte_map[self._byte_map["type"] == "i"].iterrows():
            val = None
            if r["n"] == 1:
                val = struct.unpack("i", self._byte_data[r["bytes_start"]:r["bytes_end"]])[0]
            else:
                val = np.array(struct.unpack("i"*r["n"], self._byte_data[r["bytes_start"]:r["bytes_end"]]))
            self._comm_scalars[r["name"]] = val
    
    def _load_scalars(self):
        """
        loads all scalars after unpacking the first few!
        """
        if self._byte_map.shape[0] < 27:
            raise ValueError("Did you load some data?")

        for i, r in self._byte_map[:27].iterrows():
            val = None
            if r["n"] == 1:
                val = struct.unpack(r["type"], self._byte_data[r["bytes_start"]:r["bytes_end"]])[0]
            else:
                val = np.array(struct.unpack(r["type"]*r["n"], self._byte_data[r["bytes_start"]:r["bytes_end"]]))
            self._comm_scalars[r["name"]] = val

    def _load_cluster_data(self):
        if self._byte_map.shape[0] < 28:
            raise ValueError("Did you load some data?")

        data = {}

        for i, r in self._byte_map[self._byte_map["save_in"] == "data"].iterrows():
            if "fortran_header" in r["name"] or "fortran_footer" in r["name"]:
                continue
            val = None
            if r["n"] == 1:
                val = struct.unpack(r["type"], self._byte_data[r["bytes_start"]:r["bytes_end"]])[0]
            else:
                val = np.array(struct.unpack(r["type"]*r["n"], self._byte_data[r["bytes_start"]:r["bytes_end"]]))

            if r["name"][-3:] == "123":
                base_name = r["name"][:-3]
                for j in range(1, 4):
                    exec(f"data['{base_name}{j}'] = val[{j-1}::3]")
            else:
                data[r["name"]] = val

        self._comm_data = pd.DataFrame(data)

        return self.data

    def get_bytes(self, name: int, col: str = None):
        """
        get the relevant bytes for column for `name`

        Either use `j` as param *or* `name`!

        :param col: Column name of the value of interest
        :type col: str
        :param name: Nbody name (integer) of particle of interest
        :type name: int
        """
        if col is None:
            ret = {}
            for i,r in self._byte_map.iterrows():
                if r["save_in"] == "ignore" or r["save_in"] == "scalars":
                    continue
                b = self.get_bytes(name=name, col=r["name"])
                print(i, r["name"], b, b[1]-b[0], int((b[1]-b[0])/r["bytes_data_type"]))
                ret[r["name"]] = b
            return ret
                
        j = self.data.loc[self.data["NAME"] == name].index.values[0]
        
        start = None
        end = None

        # Case e.g. col="V1" needs to be mapped to V123 and then extracted
        if col[-1] in ["1", "2", "3"] and col[:-1] + "123" in self._byte_map["name"].values:
            vec_id = int(col[-1]) - 1
            col_name = col[:-1] + "123" 
            byte_map_row = self._byte_map[self._byte_map["name"] == col_name]

            start = byte_map_row["bytes_start"].values[0]
            start += byte_map_row["bytes_data_type"].values[0] * j * 3
            start += byte_map_row["bytes_data_type"].values[0] * vec_id
            end = start + byte_map_row["bytes_data_type"].values[0]
        # Case col="V123" return all bytes!
        elif col[-3:] == "123" and col in self._byte_map["name"].values:
            byte_map_row = self._byte_map[self._byte_map["name"] == col]
            start = byte_map_row["bytes_start"].values[0]
            start += byte_map_row["bytes_data_type"].values[0] * j * 3
            end = start + 3*byte_map_row["bytes_data_type"].values[0]
        # Case col="U_1234" (pair data)
        elif col[-4:] == "1234" and col in self._byte_map["name"].values:
            byte_map_row = self._byte_map[self._byte_map["name"] == col]
            start = byte_map_row["bytes_start"].values[0]
            start += byte_map_row["bytes_data_type"].values[0] * j * 4
            end = start + 4*byte_map_row["bytes_data_type"].values[0]
        # Case col="SF_1234567" (pair data)
        elif col[-7:] == "1234567" and col in self._byte_map["name"].values:
            byte_map_row = self._byte_map[self._byte_map["name"] == col].index.values[0]
            byte_map_row = self._byte_map.iloc[byte_map_row].to_dict()
            start = byte_map_row["bytes_start"]
            start += byte_map_row["bytes_data_type"] * j * 7
            end = start + 7*byte_map_row["bytes_data_type"]
        elif col == "LIST_DATA":
            ret = []
            byte_map_row = self._byte_map[self._byte_map["name"] == col].index.values[0]
            byte_map_row = self._byte_map.iloc[byte_map_row].to_dict()
            particle_idx = 0
            skipper = 0
            for i in range(0,byte_map_row["n"]):
                if skipper > 0:
                    skipper -= 1
                    continue
                n = struct.unpack("i", self._byte_data[byte_map_row["bytes_start"] + (i*4): byte_map_row["bytes_start"] + ((i+1)*4)])[0]
                if n == 0:
                    print(particle_idx,i,n)
                    particle_idx += 1
                else:
                    print(particle_idx,i,n, end=" ")
                    for j in range(1, n+1):
                        cur_name = struct.unpack("i", self._byte_data[byte_map_row["bytes_start"] + ((i+j)*4): byte_map_row["bytes_start"] + ((i+j+1)*4)])[0]
                        print(cur_name, end = " ")
                    print()
                    skipper = n
                    particle_idx += 1

        # simple case
        elif col in self._byte_map["name"].values:
            byte_map_row = self._byte_map[self._byte_map["name"] == col]
            start = byte_map_row["bytes_start"].values[0]
            start += byte_map_row["bytes_data_type"].values[0] * j
            end = start + byte_map_row["bytes_data_type"].values[0]
        else:
            raise KeyError(f"Don't know column of name {col}")
        return (start, end)

    def give_kick(self,
                  v: str = "V1",
                  name: int = None,
                  n_mean: float = 5.0,
                  n_std: int = 10.0):
        """
        will give a particle a velocity kick on given `v` (`V1` by default)

        by default `V1` will be added `n_mean * mean_v + n_std * std_v`

        :param v: which V (`V1`, `V2` or `V3`) to apply the kick to
        :type v: str
        :param name: Nbody name (integer) of particle of interest
        :type name: int
        :param n_mean: factor for mean to add to the velocity
        :type n_mean: float
        :param n_std: factor for standard deviation to add to the velocity
        :type n_std: float

        """

        relevant_bytes = self.get_bytes(name=name, col=v)
        relevant_bytes_v0 = self.get_bytes(col=v[0] + "0" + v[1], name=name)
        
        self._byte_data = ( self._byte_data[:relevant_bytes[0]]
                            + struct.pack("d", self.data[v].mean() * n_mean + self.data[v].std() * n_std)
                            + self._byte_data[relevant_bytes[1]:]
                           )
        self._byte_data = ( self._byte_data[:relevant_bytes_v0[0]]
                            + struct.pack("d", self.data[v].mean() * n_mean + self.data[v].std() * n_std)
                            + self._byte_data[relevant_bytes_v0[1]:]
                           )

    def _load_file(self, file: str, time: float = None):
        if time is not None:
            self.time = time
        else:
            try:
                self.time = float(file[file.rfind("/")+1:].replace("comm.2_", ""))
            except:
                self.time = None
 
        # reset data 
        self.time = None
        self._byte_data = None
        self._byte_map = None
        self._value_map = copy.deepcopy(self._init_value_map)
        self._byte_map = pd.DataFrame(self._value_map)
        self._init_byte_map()
        self._comm_data = None
        self._comm_scalars = None

        self._comm_scalars = {}

        data = None
        with open(file, "rb") as comm_file:
            data = comm_file.read()
        self._byte_data = data

        self._load_scalars_init()
        self._update_byte_map_with_scalars()
        self._load_scalars()
        self._load_cluster_data()
                
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

    def save(self, filename: str):
        """
        Save comm file. Currently only useful after applying a kick!

        :param filename: name (and path) of the output file
        """
        with open(filename, "wb") as f:
            f.write(self._byte_data)
