import logging
import re
import glob
from tqdm import tqdm
import pandas as pd
import sys

from .snap import snap
from .cluster import cluster
from . import data_files


#SUPPORTED_DATA_FILES = [i for i in dir(data_files) if ((i[:2] != "__") and (i != "pythonbody") and ((i != "data_file") and (i != "pbdf")))]
SUPPORTED_DATA_FILES = [i for i in dir(data_files) if ((i[:2] != "__") and eval(f"data_files.{i}.AUTO_LOAD"))]
#SUPPORTED_COLS = {i: eval(f"data_files.{i}").COLS for i in dir(data_files) if ((i[:2] != "__") and (i != "pythonbody") and ((i != "data_file") and (i != "pbdf")))}
SUPPORTED_DATA_COLS = [eval(f"data_files.{i}").COLS for i in dir(data_files) if ((i[:2] != "__") and eval(f"data_files.{i}.AUTO_LOAD"))]
CALCABLE_DATA = {key: [i for i in eval(f"dir(data_files.{key})") if "calc_" in i] for key in SUPPORTED_DATA_FILES}


class UnknownDataLoad(Exception):
    pass


class nbody:
    """
    Class for handling and nbody run or project

    :param data_path: Path where output file of Nbody can be found
    :type data_path: str or None
    :param nb_stdout_files: Name(s) of the Nbody standard output files
    :type nb_stdout_files: list[str] or str or None

    Usage
    _____

    Basically a `dict` type. When given a ``data_path`` and ``nb_stdout_files``
    you can use the ``load()`` function to automatically load everything.

    .. code-block:: python

        >>> from pythonbody import nbody
        >>> n100k = nbody(data_path="/path/to/nbody/run",
                          nb_stdout_files = ["N100k_1.out", "N100k_2.out"])

        >>> n100k.load() # try to load all the data

    Afterwards the data can be accessed either by standard dict way or using
    the properties

    .. code-block:: python
        
        >>> n100k.keys()
        dict_keys(['globals', 'lagr', 'stdout'])
        
        >>> n100k["lagr"].keys()
        dict_keys(['RLAGR', 'RLAGR_S', 'RLAGR_B', '<M>', 'N_SHELL', '<V_x>', '<V_y>', '<V_z>', '<V>', '<V_r>', '<V_t>', 'SIG2', 'SIGR2', 'SIGT2', 'VROT'])

        >>> n100k.lagr.keys() # same as above
        dict_keys(['RLAGR', 'RLAGR_S', 'RLAGR_B', '<M>', 'N_SHELL', '<V_x>', '<V_y>', '<V_z>', '<V>', '<V_r>', '<V_t>', 'SIG2', 'SIGR2', 'SIGT2', 'VROT'])
    
        >>> n100k["lagr"]["RLAGR"] # same as n100k.lagr["RLAGR"]
                0.001     0.003     0.005      0.01  ...      0.95       0.99         1.       <RC
        TIME                                            ...
        0.0     0.055542  0.075024  0.087036  0.104731  ...  2.649137   3.645590   5.157713  0.219672
        1.0     0.031653  0.063853  0.077020  0.093948  ...  2.666836   3.659202   5.149258  0.193862
        2.0     0.039620  0.060507  0.077393  0.091861  ...  2.681057   3.670711   7.472866  0.192845
        3.0     0.034356  0.036832  0.059603  0.089674  ...  2.690331   3.667219  11.214958  0.114667
        4.0     0.047654  0.051617  0.069807  0.097573  ...  2.683758   3.696618  14.946478  0.151964
        ...          ...       ...       ...       ...  ...       ...        ...        ...       ...
        2941.0  0.121543  0.154369  0.173205  0.234624  ...  8.197157  15.914945  51.670314  0.491288
        2942.0  0.098412  0.156419  0.195484  0.247737  ...  8.195195  15.871122  51.222233  0.519969
        2943.0  0.075497  0.144453  0.170336  0.187283  ...  8.193181  15.870110  51.328479  0.439498
        2944.0  0.068826  0.090216  0.104810  0.164519  ...  8.204566  15.934179  51.527380  0.359200
        2945.0  0.073164  0.110651  0.149710  0.181364  ...  8.216966  15.939291  51.687435  0.392462

        [2954 rows x 19 columns]        

    Methods and Properties
    ----------------------
    """

    def __init__(self, data_path: str = None, nb_stdout_files: list = None):
        """
        nbody class initializer (with data)

        :param data_path: Path where output file of Nbody can be found
        :type data_path: str or None
        :param nb_stdout_files: Name(s) of the Nbody standard output files
        :type nb_stdout_files: list[str] or str or None       
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
         
        self.data_path = data_path
        # make sure self.data_path ends with "/"
        if (self.data_path and
            self.data_path[len(self.data_path) -1 ] != "/"):            
            self.data_path += "/"

        self.snap = snap(self.data_path)
        self.cluster = cluster(self.data_path, s=self.snap)
        self.conf = data_files.conf(self.data_path)
         
        self._data = {}
        
        self._files = {}
        self._files["nb_stdout_files"] = nb_stdout_files if type(nb_stdout_files) == list else [nb_stdout_files]
         
    def __setitem__(self, key, item):
        self._data[key] = item

    def __getitem__(self, key):
        if key not in self._data.keys():
            if key in SUPPORTED_DATA_FILES:
                self.load(key)
            else:
                for k in self._data.keys():
                    if key in self._data[k]:
                        return self._data[k][key]
                for k in SUPPORTED_COLS.keys():
                    if key in SUPPORTED_COLS[k]:
                        self.load(k)
                        return self._data[k][key]

        return self._data[key]

    def __repr__(self):
        return str(self._data.keys())

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def keys(self):
        """
        return available keys (examined data files) in data.
        """
        return self._data.keys()
 
    def load(self, *what, max_nb_time: float = None):
        """
        Loads data into the self._data from files associated with nbody.

        Available `what` values are `esc`, `globals` `lagr`, `stdout`.
        If there is nothing passed, automatically all available data_files
        will be loaded.
        
        :param what: which data type to load, available values are ``esc``,
            ``globals`` ``lagr``, ``stdout``. 

            If there is nothing passed, 
            automatically all available data_files will be loaded.
        :type what: list[str] or str or 'all' or None
        :param max_nb_time: Optional do not read anything above max_nb_time. Currently
            only applies to stdout data_file, as everything else is quite
            fast anyway.
        :type max_nb_time: float or None
        """
        if len(what) == 0 or what is None:
            return self.load(*SUPPORTED_DATA_FILES, max_nb_time=max_nb_time)

        for load_type in what:
            
            # Handle weird quirks due to global being a keyword in python
            if load_type == "global":
                load_type = "globals"
                self._data["global"] = eval(f"data_files.{load_type}.load('{self.data_path}')")

            elif load_type not in SUPPORTED_DATA_FILES:
                for key in SUPPORTED_COLS.keys():
                    if load_type in SUPPORTED_COLS[key]:
                        data_path = self.data_path
                        if key == "stdout":
                            data_path = [self.data_path + "/" + i for i in self._files['nb_stdout_files']]
                            self._data[key] = eval(f"data_files.{key}.load({data_path}, max_nb_time={max_nb_time})")
                            break
                        else:
                            self._data[key] = eval(f"data_files.{key}.load('{data_path}')")
                            break
                else:
                    raise UnknownDataLoad(f"Couldn't load data for {load_type}")
            else:
                data_path = self.data_path
                if load_type == "stdout":
                    data_path = [self.data_path + "/" + i for i in self._files['nb_stdout_files']]
                    self._data[load_type] = eval(f"data_files.{load_type}.load({data_path}, max_nb_time={max_nb_time})")
                else:
                    try:
                        self._data[load_type] = eval(f"data_files.{load_type}.load('{data_path}')")
                    except Exception as e:
                        print(f"Unknown error with {load_type}: {type(e)}, {str(e)}")

    @property
    def esc(self):
        """
        See :doc:`data_files/esc`.

        :returns: data from esc.11
        :rtype: pd.DataFrame or None
        """
        if "esc" in self.keys():
            return self._data["esc"]
        else:
            return None
    
    @property
    def globals(self):
        """
        See :doc:`data_files/globals`.

        :returns: data from global.30
        :rtype: pd.DataFrame or None
        """
        if "globals" in self.keys():
            return self._data["globals"]
        else:
            return None
    
    @property
    def lagr(self):
        """
        See :doc:`data_files/lagr`.

        :returns: data from lagr.7
        :rtype: pd.DataFrame or None
        """
        if "lagr" in self.keys():
            return self._data["lagr"]
        else:
            return None

    @property
    def stdout(self):
        """
        See :doc:`data_files/stdout`.

        :returns: data from stdout
        :rtype: pd.DataFrame or None
        """
        if "stdout" in self.keys():
            return self._data["stdout"]
        else:
            return None



    def calculate_energy_evolution(self):
        """
        Calculates potential and kinetic energy for every nbody time step.
        Careful: depending on hardware and runtime, this can take a while!
        """
        if self.snap.shape == (0.3):
            self.snap._load_files()
        self._data["E"] = pd.DataFrame() 
        #self._data["E"] = pd.DataFrame() 

        for i in tqdm(self.snap.reduced.index):
            self.cluster.load(i)

            # TODO: change to pandas version
            if sys.version_info.minor >= 10:
                self._data["E"].loc[i, ["EPOT", "EKIN"]] = [self.cluster.EPOT,
                                                            self.cluster.EKIN]
            else:
                self._data["E"].__init__(self._data["E"].append(pd.DataFrame(
                                                     [[self.cluster.EPOT,
                                                       self.cluster.EKIN]],
                                                     index=[i]
                                                )))
        self._data["E"].sort_index(inplace=True)
 
    def show_mem_usage(self):
        mem_usage = []
        for key in self._data.keys():
            mem_usage += [(key, "%.03f M" % (sys.getsizeof(self._data[key])/1024/1024))]
        mem_usage = sorted(mem_usage, key=lambda item: item[1], reverse=True)
        print("Memory usage:")
        print("Total: %.03f M" % sum([float(i[1].replace(" M", " ")) for i in mem_usage]))
        for data in mem_usage:
            print("%s: %s" % data)
                        
            
    def _analyze_files(self):
        """
        Scans the nbody data dirctory for usable files. Called upon at the end of __init__()
        """
        files = sorted(glob.glob(self.data_path + "*"))
        self.logger.debug("Found data files %s", files)
        
        self.logger.info("Analyzing data files in %s", self.data_path)
        
        for file in tqdm(files):
            for file_type in SUPPORTED_DATA_FILES.keys():
                if re.search(SUPPORTED_DATA_FILES[file_type], file):
                    self._files[file_type] += [file]
                    break            
