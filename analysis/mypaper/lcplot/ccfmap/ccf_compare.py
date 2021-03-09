import numpy as np
import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import tqdm

# import seaborn as sns; sns.set()

# time resolution is 0.05 s,
# conversions are below.
# 1000 = 50 s
# 200  = 10 s
# 100  =  5 s
# 20   =  1 s
PERIOD = 1000
SLIDE = 500
MAXLAGS = 200


ESTIMATE = 1


def cal_ccfseries(df_lc1, df_lc2, period, slide, maxlags=20, save=True):
    iteration = int(np.floor((len(df_lc1) - period) / slide))
    t_list = []
    for i in tqdm.tqdm(range(iteration)):
        # Time
        t = (df_lc1["t"][i*slide] + df_lc1["t"][i*slide+period]) / 2
        t_list.append(t)
        # Flux
        flx1 = stats.zscore(df_lc1["flx"][i*slide : i*slide+period])
        flx2 = stats.zscore(df_lc2["flx"][i*slide : i*slide+period])
        # Calculate
        dt = 0.05
        lags, r, z, zz = plt.xcorr(flx2, flx1, normed=1, maxlags=maxlags)
        plt.close()
        lags = lags * dt
        r_vec = np.array(r) if i == 0 else np.vstack([r_vec,r])
    if save:
        np.savetxt("lags.dat", np.array(lags), delimiter=" ")
        np.savetxt("tlist.dat", np.array(t_list), delimiter=" ")
        np.savetxt("rseries.dat", r_vec, delimiter=" ")
    return t_list, lags, r_vec



if __name__ == "__main__":

    df_lc1_cb = pd.read_csv("lc/lc1_reccb_opsconst.dat", sep=" ", names=["t", "flx", "err"])
    df_lc2_cb = pd.read_csv("lc/lc2_reccb_opsconst.dat", sep=" ", names=["t", "flx", "err"])
    df_lc1_xp = pd.read_csv("lc/lc1_recxps_opsconst.dat", sep=" ", names=["t", "flx", "err"])
    df_lc2_xp = pd.read_csv("lc/lc2_recxps_opsconst.dat", sep=" ", names=["t", "flx", "err"])
    df_lc1_op = pd.read_csv("lc/lc1_recops_opsconst.dat", sep=" ", names=["t", "flx", "err"])
    df_lc2_op = pd.read_csv("lc/lc2_recops_opsconst.dat", sep=" ", names=["t", "flx", "err"])

    lcs = [[df_lc1_cb, df_lc2_cb],
           [df_lc1_xp, df_lc2_xp],
           [df_lc1_op, df_lc2_op]]

    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(3):
    
        t_list, lags, r_vec = cal_ccfseries(lcs[i][0], lcs[i][1], PERIOD, SLIDE, maxlags=MAXLAGS)

        # Plot heat map
        z = r_vec
        x = t_list
        dx = np.round(x[1] - x[0], 5)
        y = list(lags)
        dy = np.round(y[1] - y[0], 5)
        X, Y = np.mgrid[x[0]:x[-1]+dx:len(x)*1j, y[0]:y[-1]+dy:len(y)*1j]
        # Plot
        zmax = np.max(np.abs(z))
        im = ax[i].pcolor(X, Y, z, cmap="RdBu_r", norm=Normalize(vmin=-zmax, vmax=zmax))
        # ax[i].colorbar(im)
        ax[i].hlines(0, min(x), max(x), lw=0.5, colors="grey")
        # Layout
        ax[i].set_xlabel("t")
        ax[i].set_ylabel("lag")
        ax[i].set_ylim([-1, 1])
    fig.savefig('a.png')
    plt.show()
