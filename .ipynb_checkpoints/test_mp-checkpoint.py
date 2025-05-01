import numpy as np
import pandas as pd
from multiprocessing import Pool
import os

def f(x):
    pid = os.getpid()
    print(f"process ID: {pid}")
    df = pd.DataFrame({'AAA': x, 'BBB': x})
    df['CCC'] = pid
    return df

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    a = np.array([1, 2, 3])

    p = Pool(2)
    sq = p.map(f, [a, a, a, a, a, a, a, a])
    p.close()
    df = pd.DataFrame(sq[0])
    for i in range(len(sq)-1):
        df = pd.concat([df, pd.DataFrame(sq[i+1])])
    df.reset_index(drop=True, inplace=True)
    print(df)