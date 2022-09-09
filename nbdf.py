import pandas as pd
import numpy as np

class nbdf(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, value):
        """
        checks if passed value(s) are in currently loaded dataframe, otherwise returns snap list data
        """
        if type(value) != list:
            value = [value]

        missing_list = []
        for val in value:
            if val not in self.columns:
                missing_list += [val]
            
            if len(missing_list) == 0:
                return super().__getitem__(value)
            elif len(missing_list) > 0 and np.sum([f"calc_{val}".replace("/","_over_") not in dir(self) for val in missing_list]) == 0:
                for missing in missing_list:
                    if missing not in self.columns:
                        eval(f"self.calc_{missing}()".replace("/","_over_"))
                return super().__getitem__(value)
            else:
                #return super().__getitem__(value)
                raise KeyError(f"Couldn't get key(s) {missing_list}")
        else:
            return super().__getitem__(value)

    def calc(self, *args):
        if len(args) != 0:
            for arg in args:
                if f"calc_{arg}" in dir(self):
                    eval(f"self.calc_{arg}()".replace("/","_over_"))
                else:
                    raise KeyError(f"Couldn't calculate {arg}")
        else:
            methods = dir(self)
            for method in methods:
                if "calc_" in method:
                    eval(f"self.calc_{method}()".replace("/","_over_"))
