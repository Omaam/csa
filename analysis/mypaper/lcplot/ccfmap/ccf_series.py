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
SLIDE = 100
MAXLAGS = 200


ESTIMATE = 1


def cal_ccfseries(df_lc1, df_lc2, period, slide, maxlags=20, save=True):
    iteration = int(np.floor((len(df_lc1) - period) / slide))
    t_list = []
    print("start estimating")
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
    print("finish estimating")
    return t_list, lags, r_vec



if __name__ == "__main__":

    df_lc1 = pd.read_csv("lc/lc1_recops_opsconst.dat", sep=" ", names=["t", "flx", "err"])
    df_lc2 = pd.read_csv("lc/lc2_recops_opsconst.dat", sep=" ", names=["t", "flx", "err"])
    
    if ESTIMATE:
        cal_ccfseries(df_lc1, df_lc2, PERIOD, SLIDE, maxlags=MAXLAGS)

    # Plot heat map
    fig = plt.figure(figsize=(8, 5))
    r_vec = np.loadtxt("rseries.dat", delimiter=" ")
    t_list = np.loadtxt("tlist.dat", delimiter=" ")
    lags = np.loadtxt("lags.dat", delimiter=" ")
    # np.savetxt("r.dat", r_vec.T, delimiter=" ")
    z = r_vec
    x = t_list
    dx = np.round(x[1] - x[0], 5)
    y = list(lags)
    dy = np.round(y[1] - y[0], 5)
    X, Y = np.mgrid[x[0]:x[-1]+dx:len(x)*1j, y[0]:y[-1]+dy:len(y)*1j]
    print(z.shape)
    print(X.shape)
    # Plot
    zmax = np.max(np.abs(z))
    im = plt.pcolor(X, Y, z, cmap="RdBu_r", norm=Normalize(vmin=-zmax, vmax=zmax))
    fig.colorbar(im)
    plt.hlines(0, min(x), max(x), lw=0.5, colors="grey")
    # Layout
    plt.xlabel("t")
    plt.ylabel("lag")
    plt.tight_layout()
    #
    # plt.savefig("heatmap.pdf")
    plt.savefig("heatmap.png", transparent=True, dpi=300)
    #
    plt.ylim([-5,5])
    # plt.savefig("heatmap-5to5.pdf")
    plt.savefig("heatmap-5to5.png", transparent=True, dpi=300)
    #
    plt.ylim([-1,1])
    # plt.savefig("heatmap-1to1.pdf")
    plt.savefig("heatmap-1to1.png", transparent=True, dpi=300)
    #
    plt.ylim([-0.5,0.5])
    # plt.savefig("heatmap-05to05.pdf")
    plt.savefig("heatmap-05to05.png", transparent=True, dpi=300)
    plt.show()
    plt.close()
