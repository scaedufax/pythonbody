import numpy as np
import re

from . import nbdf

def calc_shell_data(data: nbdf,
                    n_shells: int = 100,
                    by: str = "M",
                    cumulative: bool = False):

    if by not in data.columns:
        try:
            eval(f"data.calc_{by}()")
        except:
            raise KeyError(f"Couldn't find or calculate Key '{by}' in the data!")

    if not re.search(".*/.*T", by) and f"{by}/{by}T" not in data.columns:
        try:
            eval(f"data.calc_{by}_over_{by}T()")
        except:
            raise KeyError(f"Couldn't find or calculate Key '{by}/{by}T' in the data!")
    
    by = f"{by}/{by}T"
    data = data.sort_values(by=by)

    by_min = data[by].min()
    by_max = data[by].max()
    by_step = (by_max - by_min)/n_shells


    df = nbdf(columns=["N"] + list(data.columns))
    
    empty_count = 0
    for i in range(0, n_shells):
        if not cumulative:
            N = np.sum((data[by] > by_min + i*by_step) & (data[by] <= by_min + (i+1)*by_step))
            if N == 0:
                empty_count += 1
                continue
            df.loc[(by_min + (i+1)*by_step)/by_max] = data.loc[(data[by] > by_min + i*by_step) & (data[by] <= by_min + (i+1)*by_step)].mean()
            df.loc[(by_min + (i+1)*by_step)/by_max,"N"] = N
        else:
            N = np.sum(data[by] <= by_min + (i+1)*by_step)
            if N == 0:
                empty_count += 1
                continue
            df.loc[(by_min + (i+1)*by_step)/by_max] = data.loc[(data[by] <= by_min + (i+1)*by_step)].mean()
            df.loc[(by_min + (i+1)*by_step)/by_max,"N"] = N

    return df

"""class shells(pd.DataFrame):
    def __init__(self, N: int, data: pd.Dataframe, by="M"):
        super().__init__(columns=data.columns)
        self.data = data

"""        
