import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('whitegrid')

def ccf(y1, y2, dt, lagrange=[-10,10]):
    len_y1 = len(y1)
    lags_tmp = np.linspace(1, len_y1, len_y1)
    lags = np.array((lags_tmp - np.median(lags_tmp)) * dt)
    y1_stand = (y1 - np.average(y1)) / np.std(y1)
    y2_stand = (y2 - np.average(y2)) / np.std(y2)
    c = np.correlate(y2_stand, y1_stand, mode='same') / len_y1
    df = pd.DataFrame({ 'lag': lags,
                       'sp': c})
    low_lag = lagrange[0]
    hig_lag = lagrange[1]
    df_query = df.query('@low_lag <= lag < @hig_lag')
    return df_query['lag'].values, df_query['sp'].values

def divide_lc(lcdata, len_period, len_slide):
    itera = int((len(lcdata) - len_period) / len_slide)
    divided_lc = []
    for i in range(itera):
        lcdata_out = lcdata[i*len_slide : i * len_slide + len_period, :]
        divided_lc.append(lcdata_out)
    return np.array(divided_lc)

def fit(x, y, deg=4):
    A = np.polyfit(x, y, deg)
    X = []
    for k in reversed(range(0, deg+1)):
        X.append(x**k)
    X = np.array(X)
    y_hat = np.dot(A, X)
    return y_hat
    
TLIM = [0, 1200]
SLIDE = 1
if __name__ == '__main__':

    # load xps
    lcdata1_xpc = np.loadtxt('./XY/data1_xps_000.dat')
    lcdata2_xpc = np.loadtxt('./XY/data2_xps_000.dat')
    lcdatas1_xpc_slide = divide_lc(lcdata1_xpc, 1000, SLIDE)
    lcdatas2_xpc_slide = divide_lc(lcdata2_xpc, 1000, SLIDE)

    # load ops
    lcdata1_opc = np.loadtxt('./XY/data1_ops_000.dat')
    lcdata2_opc = np.loadtxt('./XY/data2_ops_000.dat')
    lcdatas1_opc_slide = divide_lc(lcdata1_opc, 1000, SLIDE)
    lcdatas2_opc_slide = divide_lc(lcdata2_opc, 1000, SLIDE)

    std1_xpc_data = np.array(list(map(np.std, lcdatas1_xpc_slide[:,:,1])))
    std2_xpc_data = np.array(list(map(np.std, lcdatas2_xpc_slide[:,:,1])))
    std1_opc_data = np.array(list(map(np.std, lcdatas1_opc_slide[:,:,1])))
    std2_opc_data = np.array(list(map(np.std, lcdatas2_opc_slide[:,:,1])))
    # np.savetxt('std_x_xpc_th07ci90.dat', std1_xpc_data)

    t_data = np.array(list(map(np.mean, lcdatas1_xpc_slide[:,:,0])))

    # figure
    plt.rcParams["font.size"] = 10
    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8,5),
                           gridspec_kw={'height_ratios': [3, 1, 3, 1]})
    ax[0].plot(lcdata1_xpc[:,0], lcdata1_xpc[:,1],
               label='XPS', color='tab:orange', alpha=.7)
    ax[0].set_ylabel('Standardized flux')
    ax[1].plot(t_data, std2_xpc_data, label='XPS', color='tab:orange', alpha=.7)
    ax[1].set_ylabel('SD')
    ax[2].plot(lcdata1_opc[:,0], lcdata1_opc[:,1],
               label='OPS', color='tab:blue', alpha=.7)
    ax[2].set_ylabel('Standardized flux')
    ax[3].plot(t_data, std2_opc_data, label='OPS', color='tab:blue', alpha=.7)
    ax[3].set_ylabel('SD')
    ax[3].set_xlabel(r'$t$ (s)')
    # ax[1,1].set_xticklabels([])
    # ax[1,0].set_xticklabels([])

    plt.subplots_adjust(left=0.09, right=0.97, bottom=0.09, top=0.95)
    plt.savefig('fig/lcNstd.png', transparent=True)
    plt.show()
