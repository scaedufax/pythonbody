import pandas as pd
import numpy as np
import glob
import logging
import h5py
import sys
from tqdm import tqdm
import pathlib
import warnings
from packaging import version

from ..ffi import ffi
from ..nbdf import nbdf
from .binaries import Binaries
from .. import defaults
from .. import settings
if settings.DEBUG_TIMING:
    import datetime as dt


class NotAvailableException(Exception):
    """
    Exception if cluster hasn't been loaded yet
    """


class snap():
    """
    Base class for handling snap (HDF5) files
    
    :param data_path: path to (snap) data -> nbody project run.
    :type data_path: str or None

    .. note::
    
        By default only looks at step 0 of each snap file. See 
        `analyze_files() <#pythonbody.snap.analyze_files>`_ for reading all steps in snap files

    Usage
    -----
    
    setting up
    
    .. code-block:: python

        >>> from pythonbody import snap

        >>> n100k = snap(data_path="/path/to/nbody/run")
        [...]
        >>> n100k.snap_data
                  time                           file step
        0.0        0.0  /path/to/nbody/run/snap.40...    0
        1.0        1.0  /path/to/nbody/run/snap.40...    0
        2.0        2.0  /path/to/nbody/run/snap.40...    0
        3.0        3.0  /path/to/nbody/run/snap.40...    0
        4.0        4.0  /path/to/nbody/run/snap.40...    0
        ...        ...                            ...  ...
        2940.0  2940.0  /path/to/nbody/run/snap.40...    0
        2941.0  2941.0  /path/to/nbody/run/snap.40...    0
        2942.0  2942.0  /path/to/nbody/run/snap.40...    0
        2943.0  2943.0  /path/to/nbody/run/snap.40...    0
        2944.0  2944.0  /path/to/nbody/run/snap.40...    0

        [2945 rows x 3 columns]

    
    loading some data
    
    .. code-block:: python

        >>> n100k.load_cluster(0)
        [...]
        >>> n100k.cluster_data
                      M         X1        X2        X3         V1  ...  RC*  MC*  K*   NAME  Type
        0      0.128024   2.385841 -4.204420  2.512420   3.158618  ...  0.0  0.0   0   7288     0
        1      1.175865   2.385809 -4.204371  2.512364   0.118995  ...  0.0  0.0   1   7287     0
        2      0.659138  -1.386117  0.442578 -0.864308  -3.723401  ...  0.0  0.0   0   2944     0
        3      0.270822  -1.386078  0.442543 -0.864313   0.623611  ...  0.0  0.0   0   2943     0
        4      0.245347   1.012453  0.333743 -0.990999  18.138487  ...  0.0  0.0   0   5860     0
        ...         ...        ...       ...       ...        ...  ...  ...  ...  ..    ...   ...
        98357  0.463962   7.346551 -2.370663  1.295420  -0.873880  ...  0.0  0.0   0  40565     0
        98358  0.173271   5.372622 -4.724232  1.189188  -2.692101  ...  0.0  0.0   0  77253     0
        98359  0.610795 -21.684790  5.551836 -5.083984   0.649133  ...  0.0  0.0   0  32193     0
        98360  0.174317 -27.157290  3.586303 -9.307677  -0.874081  ...  0.0  0.0   0  77022     0
        98361  0.194638   5.186951 -3.153325  5.327578   1.225115  ...  0.0  0.0   0  72237     0


    Methods and Properties
    ----------------------
    """
    def __init__(self, data_path):
        self._snap_data = pd.DataFrame(columns=["time", "file", "step"])
        if not pathlib.Path(data_path).is_dir():
            raise IOError(f"Couldn't find {data_path}. Does it exist?")
        self.data_path = data_path

        self.files = None
        self.time = None
        if self.data_path is not None:
            self.files = sorted(glob.glob(self.data_path + "/snap*"))
            self._load_files()

        self._cluster_data = None
        self._binaries_data = None
        self.binaries_mask = None
        self.singles_mask = None
        self.time_evolution_data = None
        self.cluster_data_mask = None
        self.binaries_data_mask = None
        self.scalar_data = {}
        self.RTIDE = None

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded cluster data,
        otherwise returns snap list data
        """
        try:
            return self.cluster_data[value]
        except:
            pass
        try:
            return self.binaries_data[value]
        except:
            pass
        try:
            return self.scalar_data[value]
        except:
            pass

        raise ValueError(f"Couldn't find data for {value}")

    def __repr__(self):
        if self.cluster_data is not None:
            return self.cluster_data.__repr__()
        return self.snap_data.__repr__()
    
    def _repr_html_(self):
        if self.cluster_data is not None:
            return self.cluster_data._repr_html_()
        return self.snap_data._repr_html_()

    @property
    def cluster_data(self):
        """
        :return: Full data of cluster read from snap file
        :rtype: nbdf
        """
        if self._cluster_data is None:
            raise NotAvailableException("Couldn't get cluster data, have you "
                                        " used the load() function?")
        return self._cluster_data

    @cluster_data.setter
    def cluster_data(self, value):
        self._cluster_data = value

    @property
    def binaries_data(self):
        """
        :return: Full data of binaries read from snap file
        :rtype: nbdf
        """
        if self._binaries_data is None:
            raise NotAvailableException("Couldn't get cluster data, have you "
                                        " used the load() function?")
        return self._binaries_data
    
    @binaries_data.setter
    def binaries_data(self, value):
        self._binaries_data = value
    
    @property
    def snap_data(self):
        """
        :return: known data about snap files
        :rtype: nbdf
        """
        if self._snap_data is None:
            raise NotAvailableException("Couldn't get snap data? That's weird "
                                        "This should really not be happending")
        return self._snap_data

    @snap_data.setter
    def snap_data(self, value):
        self._snap_data = value

    @property
    def reduced(self):
        if self.snap_data.shape[0] == 0:
            self._load_files()
        return self.snap_data[self.snap_data["time"] == self.snap_data["time"].values.astype(int)]

    @property
    def binaries(self):
        """
        :return: cluster data filtered by binaries only
        :rtype: nbdf
        """
        if self.binaries_mask is None:
            raise NotAvailableException("Couldn't get binaries, have you used "
                                        "load() function?")
        return self.cluster_data[self.binaries_mask]

    @property
    def singles(self):
        """
        :return: cluster data filtered by singles only
        :rtype: nbdf
        """
        if self.singles_mask is None:
            raise NotAvailableException("Couldn't get single, have you used "
                                        "load() function?")
        return self.cluster_data[self.singles_mask]

    @property
    def time_evolution(self):
        """
        :return: time evolution data
        :rtype: nbdf
        """
        if self.time_evolution_data is None:
            self.calculate_time_evolution()
        return self.time_evolution_data

    @property
    def potential_escapers(self, G=4.30091e-3):
        if "R" not in self.cluster_data.columns:
            self.cluster_data.calc_R()
        if "Eb_spec" not in self.cluster_data.columns:
            self.cluster_data.calc_Eb_spec()

        if self.RTIDE is not None:
            return self.cluster_data[(self.cluster_data["Eb_spec"] < 0) & (self.cluster_data["Eb_spec"] > (-1.5 * G * self.cluster_data["M"].sum() / float(self.RTIDE)))]

        if self.scalar_data["RTIDE"] == 0:
            return pd.DataFrame(columns=self.cluster_data.columns)

        return self.cluster_data[self.singles_mask & (self.cluster_data["Eb_spec"] < 0) & (self.cluster_data["Eb_spec"] > (-1.5 * G * self.cluster_data["M"].sum() / float(self.scalar_data["RTIDE"])))]

    """@property
    def binding_enegery(self, G=4.30091e-3):
        return -1.5 * G * float(self.cluster_data["M"].sum()) / float(self.cluster_data["R"].max())"""

    @property
    def loc(self, *args, **kwargs):
        return self.cluster_data.loc(*args, **kwargs)

    @property
    def iloc(self, *args, **kwargs):
        return self.cluster_data.iloc(*args, **kwargs)

    @property
    def scalars(self, *args, **kwargs):
        return self.scalar_data

    def calculate_time_evolution(self,
                                 RLAGRS: list[float] = None,
                                 stepsize: int = 1,
                                 min_nbtime: float = None,
                                 max_nbtime: float = None,
                                 *args,
                                 **kwargs):
        """
        calculate time evolution of some data

        :param RLAGRS: list of Lagrangian radii spheres to look at
        :type RLAGRS: list[float]
        :param stepsize: Step size when going through
            `snap_data <#pythonbody.snap.snap_data>`_
            This uses the stepsize on the index of
            `snap_data <#pythonbody.snap.snap_data>`_,
            this is not the time step!
        :type stepsize: int
        :param min_nbtime: offset to start at NB time unit when calculating
            time evolution
        :type min_nbtime: float
        :param max_nbtime: stop calulating time evolution at NB time unit
        :type max_nbtime: float
        :param \*args: will be passed to
            `load_cluster() <#pythonbody.snap.load_cluster>`_
        :param \*\*kwargs: will be passed to
            `load_cluster() <#pythonbody.snap.load_cluster>`_
        """
        if RLAGRS is None:
            RLAGRS = defaults.RLAGRS
        self.time_evolution_data = {
                "RLAGR": nbdf(),
                "RLAGR_BH": nbdf(),
                "E": nbdf(),
                "N": nbdf(),
                "M": nbdf(),
                "DEBUG": nbdf(),
                }
        if max_nbtime is None:
            max_nbtime = self.snap_data.index.shape[0]
        else:
            max_nbtime = self.snap_data.index[self.snap_data.index <= max_nbtime].shape[0]
        if min_nbtime is None:
            min_nbtime = 0
        else:
            min_nbtime = self.snap_data.index[self.snap_data.index < min_nbtime].shape[0]

        for idx in tqdm(self.snap_data.index[min_nbtime:max_nbtime:stepsize]):
            nbtime = None
            try:
                nbtime = self.load_cluster(idx, **kwargs,
                                           return_nbtime=True,
                                           cluster_cols=defaults.snap.time_evolution_cluster_cols,
                                           binary_cols=defaults.snap.time_evolution_binary_cols,
                                           scalar_ids=defaults.snap.time_evolution_scalars,
                                           )
            except Exception as e:
                print(f"Snap file: {self.snap_data.loc[idx,'file']}, error:\n{str(e)}")
                warnings.warn(f"Error with hdf5 file \"{self.snap_data.loc[idx,'file']}\". Exception:\n{str(e)}", Warning)
                continue

            if settings.DEBUG_TIMING:
                time_debug_time_evolution_calc = time_debug_calc = time_debug_calc_R = dt.datetime.now()
            self.cluster_data.calc_R()
            if settings.DEBUG_TIMING:
                print(f"Calculating R took {dt.datetime.now() - time_debug_calc_R}")
                time_debug_calc_M_over_MT = dt.datetime.now()
            self.cluster_data.calc_M_over_MT()
            if settings.DEBUG_TIMING:
                print(f"Calculating M/MT took {dt.datetime.now() - time_debug_calc_M_over_MT}")
                time_debug_calc_Eb = dt.datetime.now()
            self.binaries_data.calc_relEb_spec()
            self.binaries_data.calc_cmEb_spec()
            self.cluster_data.calc_Eb()
            if settings.DEBUG_TIMING:
                stop = dt.datetime.now()
                print(f"Calculating Eb took {stop - time_debug_calc_Eb}")
                print(f"All calc_* functions took {stop - time_debug_calc}")
                time_debug_RLAGR = dt.datetime.now()
            
            for rlagr in RLAGRS:
                self.time_evolution_data["RLAGR_BH"].loc[nbtime,str(rlagr)] = float(self.cluster_data[(self.cluster_data["K*"] == 14) & self.singles_mask & (self.cluster_data["M/MT"] < rlagr)]["R"].max())
                self.time_evolution_data["RLAGR"].loc[nbtime,str(rlagr)] = float(self.cluster_data[self.cluster_data["M/MT"] < rlagr]["R"].max())
            
            if settings.DEBUG_TIMING:
                print(f"Calculating RLAGR took {dt.datetime.now() - time_debug_RLAGR}")
                time_debug_N = dt.datetime.now()

            self.time_evolution_data["N"].loc[nbtime,"SINGLE_BH"] = self.cluster_data[(self.cluster_data["K*"] == 14) & self.singles_mask].shape[0]
            self.time_evolution_data["N"].loc[nbtime,"BH-BH"] = self.binaries_data[(self.binaries_data["K*1"] == 14) & (self.binaries_data["K*2"] == 14)].shape[0]
            self.time_evolution_data["N"].loc[nbtime,"BH-Any"] = self.binaries_data[(self.binaries_data["K*1"] == 14) | (self.binaries_data["K*2"] == 14)].shape[0]
            self.time_evolution_data["N"].loc[nbtime,"POTENTIAL_ESCAPERS"] = self.potential_escapers.shape[0]
            self.time_evolution_data["N"].loc[nbtime,"POTENTIAL_ESCAPERS_REL"] = self.potential_escapers.shape[0]/self.cluster_data.shape[0]
            self.time_evolution_data["N"].loc[nbtime,"SINGLES"] = self.singles.shape[0]
            self.time_evolution_data["N"].loc[nbtime,"BINARIES"] = self.binaries.shape[0]
            self.time_evolution_data["N"].loc[nbtime,"TOT"] = self.cluster_data.shape[0]
            if settings.DEBUG_TIMING:
                print(f"Calculating N data took {dt.datetime.now() - time_debug_N}")
                time_debug_M = dt.datetime.now()
            self.time_evolution_data["M"].loc[nbtime,"SINGLE_BH"] = self.cluster_data[(self.cluster_data["K*"] == 14) & self.singles_mask]["M"].sum()
            self.time_evolution_data["M"].loc[nbtime,"BH-BH"] = self.binaries_data[(self.binaries_data["K*1"] == 14) & (self.binaries_data["K*2"] == 14)][["M1", "M2"]].sum().sum()
            self.time_evolution_data["M"].loc[nbtime,"BH-Any"] = self.binaries_data[(self.binaries_data["K*1"] == 14) | (self.binaries_data["K*2"] == 14)][["M1", "M2"]].sum().sum()
            self.time_evolution_data["M"].loc[nbtime,"POTENTIAL_ESCAPERS"] = self.potential_escapers["M"].sum()
            self.time_evolution_data["M"].loc[nbtime,"SINGLES"] = self.singles["M"].sum()
            self.time_evolution_data["M"].loc[nbtime,"BINARIES"] = self.binaries["M"].sum()
            self.time_evolution_data["M"].loc[nbtime,"TOT"] = self.cluster_data["M"].sum()
            if settings.DEBUG_TIMING:
                print(f"Calculating N data took {dt.datetime.now() - time_debug_M}")
                time_debug_E = dt.datetime.now()

            self.time_evolution_data["E"].loc[nbtime,"Eb_spec_tot"] = self.cluster_data["Eb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"SINGLES_Eb_spec_tot"] = self.singles["Eb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"BINARIES_Eb_spec_tot"] = self.binaries["Eb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"Any-Any_Eb_tot"] = self.binaries_data["cmEb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"BH-Any_Eb_tot"] = self.binaries_data[(self.binaries_data["K*1"] == 14) | (self.binaries_data["K*2"] == 14)]["cmEb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"BH-BH_Eb_tot"] = self.binaries_data[(self.binaries_data["K*1"] == 14) & (self.binaries_data["K*2"] == 14)]["cmEb_spec"].sum()
            
            # clean up zero values as nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["Any-Any_Eb_tot"] == 0,"BH-Any_Eb_tot"] = np.nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["BH-Any_Eb_tot"] == 0,"BH-Any_Eb_tot"] = np.nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["BH-BH_Eb_tot"] == 0,"BH-BH_Eb_tot"] = np.nan

            if settings.DEBUG_TIMING:
                print(f"Calculating E data took {dt.datetime.now() - time_debug_E}")
                print(f"Calculating time evolution data for NB time {idx} took {dt.datetime.now() - time_debug_time_evolution_calc}")

            self.time_evolution_data["DEBUG"].loc[nbtime,"RTIDE"] = self.RTIDE
            self.time_evolution_data["DEBUG"].loc[nbtime,"RBAR"] = self.scalar_data["RBAR"]
            self.time_evolution_data["DEBUG"].loc[nbtime,"POT_NAN_OR_NULL"] = np.sum(pd.isna(self.cluster_data["POT"]) | self.cluster_data["POT"] == 0)

    def load_cluster(self,
                     time: float,
                     return_nbtime: bool = False,
                     cluster_cols: list[str] = None,
                     binary_cols: list[str] = None,
                     scalar_ids: list[int] = None,
                     cluster_data_filter: str = None,
                     binaries_data_filter: str = None):
        """
        Load cluster at a given time step

        :param time: load cluster at time from snap files
        :type time: float
        :param return_nbtime: return time in nbody units which was loaded
        :type return_nbtime: bool
        :param cluster_cols: specify columns to load in general
        :type cluster_cols: list[str]
        :param binary_cols: specify columns to load for binaries
        :type binary_cols: list[str]
        :param scalar_ids: specify the ids to load from scalars in snap files.
        :type scalar_ids: list[in]
        :param cluster_data_filter: filter cluster_data, the passed string will
            be called using ``eval``. The internals need to be known and
            understood to properly use this.

        :type cluster_data: str
        :param binaries_data_filter: filter binaries_data, the passed string
            will be called using ``eval``. The internals need to be known and
            understood to properly use this.

        :type binaries_data_filter: str
        """
        if self.snap_data.shape == (0, 3):
            self._load_files()

        if settings.DEBUG_TIMING:
            time_debug_load_cluster = dt.datetime.now()

        self.time = time

        default_cluster_cols = cluster_cols if cluster_cols is not None else defaults.snap.cluster_col_map.keys()
        default_binary_cols = binary_cols if binary_cols is not None else defaults.snap.binary_col_map.keys()
        default_scalar_ids = scalar_ids if scalar_ids is not None else defaults.snap.scalar_map.keys()
        
        if settings.DEBUG_TIMING:
            time_debug_hdf5_file = dt.datetime.now()
        f = h5py.File(self.snap_data.loc[time]["file"],"r")
        if settings.DEBUG_TIMING:
            print(f"Loading hdf5 file {time} took {dt.datetime.now() - time_debug_hdf5_file}")
        nbtime = f["Step#" + self.snap_data.loc[time]["step"]]["000 Scalars"][0]
        self.cluster_data = nbdf(columns=default_cluster_cols)
        for col in default_cluster_cols:
            if defaults.snap.cluster_col_map[col] in f["Step#" + self.snap_data.loc[time]["step"]].keys():
                self.cluster_data[col] = f["Step#" + self.snap_data.loc[time]["step"]][defaults.snap.cluster_col_map[col]][:]

        self.binaries_data = Binaries(columns=default_binary_cols)
        for col in default_binary_cols:
            if defaults.snap.binary_col_map[col] in f["Step#" + self.snap_data.loc[time]["step"]].keys():
                self.binaries_data[col] = f["Step#" + self.snap_data.loc[time]["step"]][defaults.snap.binary_col_map[col]][:]

        #self.singles_data = singles(self.cluster_data, self.binary_data)
        try:
            self.singles_mask = ~self.cluster_data["NAME"].isin(self.binaries_data["NAME1"]) & ~self.cluster_data["NAME"].isin(self.binaries_data["NAME2"])
        except:
            self.singles_mask = np.repeat(True, self.cluster_data.shape[0])
        self.binaries_mask = ~ self.singles_mask

        for scalar in default_scalar_ids:
            self.scalar_data[defaults.snap.scalar_map[scalar]] = f["Step#" + self.snap_data.loc[time]["step"]]["000 Scalars"][scalar]

        self.RTIDE = self.scalar_data["RTIDE"] = (self.cluster_data["M"].sum()/self.scalar_data["ZMBAR"]/self.scalar_data["TIDAL1"])**(1/3)
        
        if settings.DEBUG_TIMING:
            print(f"Loading cluster data at time {time} took {dt.datetime.now() - time_debug_load_cluster}")
        
        # apply filters
        if cluster_data_filter is not None:
            self.cluster_data.calc_spherical_coords()
            self.cluster_data = self.cluster_data[eval(cluster_data_filter)]
        if binaries_data_filter is not None:
            self.binaries_data.calc_spherical_coords()
            self.binaries_data = self.binaries_data[eval(binaries_data_filter)]

        if return_nbtime:
            return float(nbtime)

        return self.cluster_data

    def filter(self, value):
        if value == "BH":
            return self.cluster_data[self.cluster_data["K*"] == 14]
        elif value == "SINGLE_BH":
            return self.cluster_data[(self.singles["K*"] == 14) & self.singles_mask]
        elif value == "BH-Any":
            return self.binaries_data[(self.binaries_data["K*1"] == 14) | (self.binaries_data["K*2"] == 14)]
        elif value == "BH-BH":
            return self.binaries_data[(self.binaries_data["K*1"] == 14) & (self.binaries_data["K*2"] == 14)]
        else:
            raise KeyError(f"Couldn't filter by value \"{value}\"")

    def _load_files(self):
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0
        for file in self.files:
            # if else clause should do the same, but kepler complains about setting new row with loc
            # newer python versions complain about appending... so either way...
            # TODO: change python ver to pandas ver
            if sys.version_info.minor >= 10:
                self.snap_data.loc[float(file[file.rfind("/")+1:].replace("snap.40_","").replace(".h5part",""))] = [float(file[file.rfind("/") + 1:].replace("snap.40_","").replace(".h5part","")),file, "0"]
            else:
                self.snap_data.__init__(self.snap_data.append({"time": float(file[file.rfind("/")+1:].replace("snap.40_","").replace(".h5part","")),
                                              "file": file,
                                              "step": "0"}, ignore_index=True))
                self.snap_data.index = self.snap_data["time"].values
        self.snap_data.sort_index(inplace=True)

    def analyze_files(self):
        """
        analyze snap files to read all steps from each snap file.

        .. note::
            
            depending on the size of the project run, reading all the snap
            files might take some time!
        """
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0

        self.snap_data.__init__(columns=["time","file","step"])

        for file in tqdm(self.files):
            f = h5py.File(file,"r")

            for step in f.keys():

                if version.parse(pd.__version__) >= version.parse("1.4.0") :
                    self.snap_data.loc[f[step]['000 Scalars'][0]] = [f[step]['000 Scalars'][0],file, step.replace("Step#","")]
                else:
                    self.snap_data.__init__(self.snap_data.append({"time": f[step]['000 Scalars'][0],
                                                  "file": file,
                                                  "step": step.replace("Step#","")}, ignore_index=True))
            self.snap_data.index = self.snap_data["time"].values
            f.close() 
