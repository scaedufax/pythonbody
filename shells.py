import pandas as pd
import numpy as np
import logging

def calc_shell_data(data: pd.DataFrame,n_shells=100, by="M"):
    if by not in data.columns:
        raise KeyError(f"Couldn't find Key '{by}' in the data!")

    by_min = data[by].min()
    by_max = data[by].max()
    by_step = (by_max - by_min)/n_shells

    df = pd.DataFrame(columns=["N"] + list(data.columns))
    
    empty_count = 0
    for i in range(0,n_shells):
        N = np.sum((data[by] > by_min + i*by_step) & (data[by] <= by_min + (i+1)*by_step))
        if N == 0:
            empty_count += 1
            continue
        df.loc[(by_min + (i+1)*by_step)/by_max] = data.loc[(data[by] > by_min + i*by_step) & (data[by] <= by_min + (i+1)*by_step)].mean()
        df.loc[(by_min + (i+1)*by_step)/by_max,"N"] = N
    return df



"""class shells(pd.DataFrame):
    def __init__(self, N: int, data: pd.Dataframe, by="M"):
        super().__init__(columns=data.columns)
        self.data = data

"""        
