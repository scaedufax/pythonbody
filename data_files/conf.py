import pandas as pd
import numpy as np
import pathlib
import glob
import struct

from ..nbdf import nbdf


class conf():
    """
    Base class for handling conf files
    
    :param data_path: path to (conf) data, usually just the nbody project run.
    :type data_path: str or None

    Usage
    -----

    .. code-block:: python
        
        >>> from pythonbody.data_files import conf
        >>> c = conf("/path/to/nbody/run")
        >>> c.load(0)
        >>> c.scalars
        {'NTOT': 1000,  'MODEL': 1,  'NRUN': 1,  'NK': 20,  'TIME': 0.0,  'NPAIRS': 0.0,  'RBAR': 1.0,  'ZMBAR': 604.2362670898438,  'RTIDE': 13.855620384216309,  'TIDAL4': 0.022379951551556587,  'RDENS1': -0.02988378144800663,  'RDENS2': 0.017876293510198593,  'RDENS3': 0.08017675578594208,  'TIME/TCR': 0.0,  'TSCALE': 0.6076550483703613,  'VSTAR': 1.6120661497116089,  'RC': 0.2905365228652954,  'NC': 81.0,  'VC': 0.7904078960418701,  'RHOM': 2.732787847518921,  'CMAX': 7.402853012084961,  'RSCALE': 0.7310086488723755,  'RSMIN': 100.0,  'DMIN1': 100.0}
        >>> c.data
                    M       RHO         XNS        X1        X2  ...        V1        V2        V3       POT  NAME
        0    0.086948  0.000000    0.629718 -0.073043  0.484168  ...  0.154465 -0.826052 -0.629742 -0.947794     1 
        1    0.044023  0.112857   12.510635 -0.373770  0.219240  ...  0.433374  0.202484  0.236858 -1.031206     2 
        2    0.032648  0.000000    0.000000 -2.536047 -0.354711  ...  0.002954 -0.209583  0.420556 -0.371486     3 
        3    0.022738  0.647835  202.326477 -0.244546  0.179264  ...  0.443664  1.051664  0.061355 -1.402774     4
        4    0.019321  3.739928  153.056396 -0.076585 -0.395928  ...  0.256862  0.116328 -0.392418 -1.308345     5 
        ..        ...       ...         ...       ...       ...  ...       ...       ...       ...       ...   ...
        995  0.000134  0.064970    0.761322  0.473128  0.365016  ... -0.475343  0.367096 -0.185623 -1.097924   996 
        996  0.000134  0.085231    0.700280  0.394393  0.888119  ...  0.026954 -0.162636 -0.209924 -0.893778   997 
        997  0.000134  0.025101    0.237065 -0.335770 -0.674808  ... -0.527224  0.290886  0.010729 -0.888138   998 
        998  0.000134  1.250348  383.481293 -0.108430  0.059100  ...  0.517059  0.226135 -0.205559 -1.623267   999 
        999  0.000133  0.000000    0.000000 -0.104434 -0.940669  ...  0.480972  0.098430  0.040469 -0.732077  1000

    """
    AUTO_LOAD = False
    def __init__(self, data_path):
        self._files = pd.DataFrame(columns=["time", "file"])
        if data_path is not None and not pathlib.Path(data_path).is_dir():
            raise IOError(f"Couldn't find {data_path}. Does it exist?")
        self.data_path = data_path

        self.time = None
        
        if self.data_path is not None:
            files = sorted(glob.glob(self.data_path + "/conf.3_*"))
            time = [float(file[file.rfind("/")+1:].replace("conf.3_", "")) for file in files]
            self._files["file"] = files
            self._files["time"] = time
            self._files.index = time
            self._files = self.files.sort_index()
        
        self._scalar_map = {1: "NTOT",
                            2: "MODEL",
                            3: "NRUN",
                            4: "NK",
                            7: "TIME",
                            8: "NPAIRS",
                            9: "RBAR",
                            10: "ZMBAR",
                            11: "RTIDE",
                            12: "TIDAL4",
                            13: "RDENS1",
                            14: "RDENS2",
                            15: "RDENS3",
                            16: "TIME/TCR",
                            17: "TSCALE",
                            18: "VSTAR",
                            19: "RC",
                            20: "NC",
                            21: "VC",
                            22: "RHOM",
                            23: "CMAX",
                            24: "RSCALE",
                            25: "RSMIN",
                            26: "DMIN1"}

        self._conf_data_offset = 27

        self._conf_data = None
        self._conf_scalars = None

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded cluster data,
        otherwise returns snap list data
        """
        try:
            return self._conf_data[value]
        except:
            pass
        try:
            return self._conf_scalars[value]
        except:
            pass
        try:
            return self._files[value]
        except:
            pass

        raise ValueError(f"Couldn't find data for {value}")

    def __repr__(self):
        if self._conf_data is not None:
            return self._conf_data.__repr__()
        return self._files.__repr__()
    
    def _repr_html_(self):
        if self._conf_data is not None:
            return self._conf_data._repr_html_()
        return self._files._repr_html_()

    @property
    def data(self):
        return self._conf_data
    
    @property
    def scalars(self):
        return self._conf_scalars

    @property
    def files(self):
        return self._files

    def load(self, time: float):
        """
        load conf file at a given time step

        :param time: time in Nbody units to load
        """
        if time not in self._files.index:
            raise KeyError(f"Couldn't find file according to time {time} see conf.files for files")
        if not pathlib.Path(self._files.loc[time,"file"]).is_file():
            raise IOError(f"Couldn't find {self._files.loc[time,'file']}. Does it exist?")

        self.time = time

        self._conf_scalars = {}

        data = None
        with open(self._files.loc[time,"file"], "rb") as conf_file:
            data = conf_file.read()

        for scalar in self._scalar_map.keys():
            if scalar < 6:
                val = struct.unpack("i", data[scalar*4:(scalar+1)*4])[0]
            else:
                val = struct.unpack("f", data[scalar*4:(scalar+1)*4])[0]
            self._conf_scalars[self._scalar_map[scalar]] = val

        ntot = self._conf_scalars["NTOT"]
        offset = self._conf_data_offset
        mass = np.array(struct.unpack("f"*ntot, data[(offset + 0*ntot)*4: (offset + 1*ntot)*4]))
        rho  = np.array(struct.unpack("f"*ntot, data[(offset + 1*ntot)*4: (offset + 2*ntot)*4]))
        XNS  = np.array(struct.unpack("f"*ntot, data[(offset + 2*ntot)*4: (offset + 3*ntot)*4]))
        X123 = np.array(struct.unpack("f"*ntot*3, data[(offset + 3*ntot)*4: (offset + 6*ntot)*4]))
        V123 = np.array(struct.unpack("f"*ntot*3, data[(offset + 6*ntot)*4: (offset + 9*ntot)*4]))
        X1 = np.array([i for i in X123[0::3]])
        X2 = np.array([i for i in X123[1::3]])
        X3 = np.array([i for i in X123[2::3]])
        V1 = np.array([i for i in V123[0::3]])
        V2 = np.array([i for i in V123[1::3]])
        V3 = np.array([i for i in V123[2::3]])
        POT  = np.array(struct.unpack("f"*ntot, data[(offset + 9*ntot)*4: (offset + 10*ntot)*4]))
        NAME = np.array(struct.unpack("i"*ntot, data[(offset + 10*ntot)*4: (offset + 11*ntot)*4]))

        self._conf_data = pd.DataFrame({
            "M": mass,
            "RHO": rho,
            "XNS": XNS,
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "V1": V1,
            "V2": V2,
            "V3": V3,
            "POT": POT,
            "NAME": NAME
            })
