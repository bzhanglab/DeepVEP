import numpy as np
import pandas as pd
from scipy.stats import iqr

def combine_rts(x, method="mean", remove_outlier=True):
    n_1 = len(x)
    if remove_outlier == True:
        #print("Remove outlier: %s" % str(remove_outlier))
        r1 = np.percentile(x, 25) - 1.5 * iqr(x)
        r2 = np.percentile(x, 75) + 1.5 * iqr(x)
        x = x[(x >= r1) & (x <= r2)]
        n_2 = len(x)
        #print("remove %d" % (n_1 - n_2))
    if method == "mean":
        #print("Combine method: %s" % str(method))
        res = np.mean(x)
    else:
        #print("Combine method: %s" % str(method))
        res = np.median(x)
        #res = np.percentile(x, 50)

    return (res)

def add_ptm_column(f,ptm):
    a = pd.read_table(f,sep="\t",header=0)
    a['ptm'] = ptm
    return a