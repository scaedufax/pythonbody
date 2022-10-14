import pandas as pd
import numpy as np
import glob
import logging
import h5py
import sys
from tqdm import tqdm
import pathlib
import warnings

from pythonbody.ffi import ffi
from pythonbody.nbdf import nbdf
from pythonbody.snap.binaries import Binaries
from pythonbody import defaults
from .. import settings
if settings.DEBUG_TIMING:
    import datetime as dt
#from .. import defaults
#from pythonbody.snap.singles import singles


class snap():
    def __init__(self, data_path=None):
        self.snap_data = pd.DataFrame(columns=["time", "file", "step"])
        if not pathlib.Path(data_path).is_dir():
            raise IOError(f"Couldn't find {data_path}. Does it exist?")
        self.data_path = data_path

        self.files = None
        self.time = None
        if self.data_path is not None:
            self.files = sorted(glob.glob(self.data_path + "/snap*"))
            self._load_files()

        self.cluster_data = None
        self.binaries_data = None
        self.binaries_mask = None
        self.singles_mask = None
        self.time_evolution_data = None
        self.scalar_data = {}
        self.RTIDE = None

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded cluster data,
        otherwise returns snap list data
        """
        """if type(value) != list:
            value = [value]"""

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

        """missing_list = []
        if self.cluster_data is not None:
            for val in value:
                if val not in self.cluster_data.columns:
                    missing_list += [val]

            if len(missing_list) == 0:
                return self.cluster_data[value]
            elif len(missing_list) > 0 and np.sum([f"calc_{val}".replace("/", "_over_") not in dir(self) for val in missing_list]) == 0:
                for missing in missing_list:
                    if missing not in self.cluster_data.columns:
                        eval(f"self.calc_{missing}()".replace("/", "_over_"))
                return self.cluster_data[value]
            else:
                return super().__getitem__(value)
        else:
            return super().__getitem__(value)"""

    def __repr__(self):
        if self.cluster_data is not None:
            return self.cluster_data.__repr__()
        return self.snap_data.__repr__()
    
    def _repr_html_(self):
        if self.cluster_data is not None:
            return self.cluster_data._repr_html_()
        return self.snap_data._repr_html_()

    @property
    def reduced(self):
        if self.snap_data.shape[0] == 0:
            self._load_files()
        return self.snap_data[self.snap_data["time"] == self.snap_data["time"].values.astype(int)]

    @property
    def binaries(self, t=0):
        if self.binaries_mask is None:
            self.load_cluster(t)
        return self.cluster_data[self.binaries_mask]

    @property
    def singles(self, t=0):
        if self.singles_mask is None:
            self.load_cluster(t)
        return self.cluster_data[self.singles_mask]

    @property
    def time_evolution(self):
        if self.time_evolution_data is None:
            self.calculate_time_evolution()
        return self.time_evolution_data

    @property
    def potential_escapers(self, t=0, G=4.30091e-3):
        if self.cluster_data is None:
            self.load_cluster(t)
        if "R" not in self.cluster_data.columns:
            self.calc_R()
        if "Eb_spec" not in self.cluster_data.columns:
            self.calc_Eb_spec()

        if self.RTIDE is not None:
            return self.cluster_data[(self.cluster_data["Eb_spec"] < 0) & (self.cluster_data["Eb_spec"] > (-1.5 * G * self.cluster_data["M"].sum() / float(self.RTIDE)))]

        if self.scalar_data["RTIDE"] == 0:
            return pd.DataFrame(columns=self.cluster_data.columns)

        return self.cluster_data[self.singles_mask & (self.cluster_data["Eb_spec"] < 0) & (self.cluster_data["Eb_spec"] > (-1.5 * G * self.cluster_data["M"].sum() / float(self.scalar_data["RTIDE"])))]

    @property
    def binding_enegery(self, t=0, G=4.30091e-3):
        if self.cluster_data is None:
            self.load_cluster(t)
        return -1.5 * G * float(self.cluster_data["M"].sum()) / float(self.cluster_data["R"].max())

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
                                 RLAGRS=None,
                                 stepsize=1,
                                 min_nbtime=None,
                                 max_nbtime=None):
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
                nbtime = self.load_cluster(idx, return_nbtime=True,
                                           cluster_cols=defaults.snap.time_evolution_cluster_cols,
                                           binary_cols=defaults.snap.time_evolution_binary_cols,
                                           )
            except Exception as e:
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

            self.time_evolution_data["E"].loc[nbtime,"Any-Any_Eb_tot"] = self.binaries_data["cmEb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"Singles_Eb_tot"] = self.cluster_data["Eb"].sum()
            self.time_evolution_data["E"].loc[nbtime,"Any-Any_Eb_mean"] = self.binaries_data["cmEb_spec"].mean()
            self.time_evolution_data["E"].loc[nbtime,"BH-Any_Eb_tot"] = self.binaries_data[(self.binaries_data["K*1"] == 14) | (self.binaries_data["K*2"] == 14)]["cmEb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"BH-Any_Eb_mean"] = self.binaries_data[(self.binaries_data["K*1"] == 14) | (self.binaries_data["K*2"] == 14)]["cmEb_spec"].mean()
            self.time_evolution_data["E"].loc[nbtime,"BH-BH_Eb_tot"] = self.binaries_data[(self.binaries_data["K*1"] == 14) & (self.binaries_data["K*2"] == 14)]["cmEb_spec"].sum()
            self.time_evolution_data["E"].loc[nbtime,"BH-BH_Eb_mean"] = self.binaries_data[(self.binaries_data["K*1"] == 14) & (self.binaries_data["K*2"] == 14)]["cmEb_spec"].mean()
            
            # clean up zero values as nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["Any-Any_Eb_tot"] == 0,"BH-Any_Eb_tot"] = np.nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["Any-Any_Eb_mean"] == 0,"BH-Any_Eb_mean"] = np.nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["BH-Any_Eb_tot"] == 0,"BH-Any_Eb_tot"] = np.nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["BH-Any_Eb_mean"] == 0,"BH-Any_Eb_mean"] = np.nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["BH-BH_Eb_tot"] == 0,"BH-BH_Eb_tot"] = np.nan
            self.time_evolution_data["E"].loc[self.time_evolution_data["E"]["BH-BH_Eb_mean"] == 0,"BH-BH_Eb_mean"] = np.nan

            if settings.DEBUG_TIMING:
                print(f"Calculating E data took {dt.datetime.now() - time_debug_E}")
                print(f"Calculating time evolution data for NB time {idx} took {dt.datetime.now() - time_debug_time_evolution_calc}")

            self.time_evolution_data["DEBUG"].loc[nbtime,"RTIDE"] = self.RTIDE
            self.time_evolution_data["DEBUG"].loc[nbtime,"RBAR"] = self.scalar_data["RBAR"]
            self.time_evolution_data["DEBUG"].loc[nbtime,"POT_NAN_OR_NULL"] = np.sum(pd.isna(self.cluster_data["POT"]) | self.cluster_data["POT"] == 0)



    def load_cluster(self, time, return_nbtime=False, cluster_cols=None, binary_cols=None):
        if self.snap_data.shape == (0, 3):
            self._load_files()

        if settings.DEBUG_TIMING:
            time_debug_load_cluster = dt.datetime.now()

        self.time = time

        default_cluster_cols = cluster_cols if cluster_cols is not None else defaults.snap.cluster_col_map.keys()
        default_binary_cols = binary_cols if binary_cols is not None else defaults.snap.binary_col_map.keys()
        
        if settings.DEBUG_TIMING:
            time_debug_hdf5_file = dt.datetime.now()
        f = h5py.File(self.snap_data.loc[time]["file"],"r")
        if settings.DEBUG_TIMING:
            print(f"Loading hdf5 file {time} took {dt.datetime.now() - time_debug_hdf5_file}")
        nbtime = f["Step#" + self.snap_data.loc[time]["step"]]["000 Scalars"][0]
        self.cluster_data = nbdf(columns=[key for key in defaults.snap.cluster_col_map.keys() if defaults.snap.cluster_col_map[key] in f["Step#" + self.snap_data.loc[time]["step"]].keys()])
        for col in default_cluster_cols:
            if defaults.snap.cluster_col_map[col] in f["Step#" + self.snap_data.loc[time]["step"]].keys():
                self.cluster_data[col] = f["Step#" + self.snap_data.loc[time]["step"]][defaults.snap.cluster_col_map[col]][:]

        self.binaries_data = Binaries(columns=[key for key in defaults.snap.binary_col_map.keys() if defaults.snap.binary_col_map[key] in f["Step#" + self.snap_data.loc[time]["step"]].keys()])
        for col in default_binary_cols:
            if defaults.snap.binary_col_map[col] in f["Step#" + self.snap_data.loc[time]["step"]].keys():
                self.binaries_data[col] = f["Step#" + self.snap_data.loc[time]["step"]][defaults.snap.binary_col_map[col]][:]

        #self.singles_data = singles(self.cluster_data, self.binary_data)
        try:
            self.singles_mask = ~self.cluster_data["NAME"].isin(self.binaries_data["NAME1"]) & ~self.cluster_data["NAME"].isin(self.binaries_data["NAME2"])
        except:
            self.singles_mask = np.repeat(True, self.cluster_data.shape[0])
        self.binaries_mask = ~ self.singles_mask

        for scalar in defaults.snap.scalar_map.keys():
            self.scalar_data[defaults.snap.scalar_map[scalar]] = f["Step#" + self.snap_data.loc[time]["step"]]["000 Scalars"][scalar]

        self.RTIDE = self.scalar_data["RTIDE"] = (self.cluster_data["M"].sum()/self.scalar_data["ZMBAR"]/self.scalar_data["TIDAL1"])**(1/3)
        
        if settings.DEBUG_TIMING:
            print(f"Loading cluster data at time {time} took {dt.datetime.now() - time_debug_load_cluster}")
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

    def _analyze_files(self):
        if self.files is None:
            logging.error("Couldn't find any snap files to load")
            return 0

        self.snap_data.__init__(columns=["time","file","step"])

        for file in tqdm(self.files):
            f = h5py.File(file,"r")

            for step in f.keys():

                # if else clause should do the same, but kepler complains about setting new row with loc
                # newer python versions complain about appending... so either way...
                # TODO: change python ver to pandas ver
                if sys.version_info.minor >= 10:
                    self.snap_data.loc[f[step]['000 Scalars'][0]] = [f[step]['000 Scalars'][0],file, step.replace("Step#","")]
                else:
                    self.snap_data.__init__(self.snap_data.append({"time": f[step]['000 Scalars'][0],
                                                  "file": file,
                                                  "step": step.replace("Step#","")}, ignore_index=True))
                    self.index = self.snap_data["time"].values
            f.close() 
