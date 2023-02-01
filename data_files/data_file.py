import pandas as pd
import ast
from pythonbody import data_files

AUTO_LOAD = False

COLS = None

class DataFile(pd.DataFrame):
    """
    extends pandas.DataFrame

    will calculate a missing column if calc_{NAME} is available in the
    referenced file type.
    """
    AUTO_LOAD = False
    def __init__(self, name: str = None, *args):
        super().__init__(*args)
        self.__name__ = name

    def __getitem__(self,key):
        try:
            return super().__getitem__(key)
        except KeyError as ke:
            missing_list = None
            if "[" in str(ke) and "]" in str(ke):
                missing_list = ast.literal_eval(str(ke).replace(" not in index", "").replace("\"",""))
            else:
                missing_list = [str(ke).replace(" not in index", "").replace("\"","").replace("\'","")]
            
            for missing in missing_list:
                if f"calc_{missing}" in eval(f"dir(data_files.{self.__name__})"):
                    self[missing] = eval(f"data_files.{self.__name__}.calc_{missing}(self)")
                else: raise KeyError(f"{missing} not in index")

            return self[key]

#def read_csv(self, *args, **kwargs):
#    return self.__init__(pd.read_csv(*args, **kwargs))

