import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from csa.csa import stcs, istcs, cv
from csa.cvresult import show_cvdata, lambda_fromcvdata
from ccfbootstrap import ccfbootstrap


# analysis option
CV = 0
STCS = 0
ISTCS = 1

CCF = 1

FIGSHOW = 1

def _get_savename(basename: str, path_to_dir, digit=3, ext='dat'):
    files = os.listdir(path_to_dir)
    for i in range(10**(digit)):
        suffix = str(i).rjust(digit, '0')
        search_file = basename + '_' + suffix + '.' + ext
        if (search_file in files) is False:
            path_to_out = os.path.join(path_to_dir, search_file)
            return path_to_out
            break

def main():

    # constant
    tperseg = 1000
    toverlap = 500
    basewidth_triang = 2*(tperseg - toverlap)

    # load data
    data1 = np.loadtxt('../xdata.dat')
    data2 = np.loadtxt('../odata.dat')
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    freqinfo = [0, 0.5, 2000]

    # cross-validation
    if CV:
        cv_sta = np.random.randint(toverlap, np.min([n1, n2]) - tperseg)
        cv_end = cv_sta + tperseg
        print(f'time range of cv: [{cv_sta}, {cv_end}]')
        cvdata = cv(data1[cv_sta:cv_end], data2[cv_sta:cv_end], freqinfo,
                    [1e-2, 1e3, 20], droprate=.5)
        np.savetxt('cvdata.dat', cvdata)

        # plot cvdata
        show_cvdata(cvdata)
        plt.savefig('cvcurve.png')
        if FIGSHOW:
            plt.show()
    cvdata = np.loadtxt('./cvdata.dat')
    show_cvdata(cvdata)
    plt.savefig('cvcurve.png')
    plt.close()
    lam_min = lambda_fromcvdata(cvdata, mode='min')
    print(f'lam_min = {lam_min:.3f}')


    # bootstrap
    if STCS:
        for k in range(10):
            freqs, t, X = stcs(data1, data2, freqinfo, lam_min,
                               tperseg, toverlap, droprate=0.5)
            filename = _get_savename('X', 'Xs')
            np.savetxt(filename, X)

    if ISTCS:
        filesX = os.listdir('Xs')
        print(f'number of X: {len(filesX)}')
        for fileX in filesX:
            path_to_fileX = os.path.join('Xs', fileX)
            X = np.loadtxt(path_to_fileX)

            # istcs
            data1_rec, data2_rec = istcs(X, data1, data2, freqinfo,
                                   tperseg, toverlap,
                                   basewidth=basewidth_triang)
            fname_data1 = _get_savename('data1', 'Ys')
            np.savetxt(fname_data1, data1_rec)
            fname_data2 = _get_savename('data2', 'Ys')
            np.savetxt(fname_data2, data2_rec)

    # ccf
    if CCF:
        files_in_Ys = os.listdir('Ys')
        files_data1 = sorted([f for f in files_in_Ys if ('data1' in f)])
        files_data2 = sorted([f for f in files_in_Ys if ('data2' in f)])
        files_data1 = list(map(lambda f, suf: suf + f,
                               files_data1,
                               np.repeat('Ys/', len(files_data1))))
        files_data2 = list(map(lambda f, suf: suf + f,
                               files_data2,
                               np.repeat('Ys/', len(files_data2))))

        # collect data1
        Y1 = []
        for file_data1 in files_data1:
            data1 = np.loadtxt(file_data1)
            t1 = data1[:,0]
            Y1.append(data1[:,1])
        Y1 = np.array(Y1)

        # collect data2
        Y2 = []
        for file_data2 in files_data2:
            data2 = np.loadtxt(file_data2)
            t2 = data2[:,0]
            Y2.append(data2[:,1])
        Y2 = np.array(Y2)

        for y1 in Y1:
            plt.plot(t1, y1, alpha=.3)
        plt.show()

        lags, c_low, c_med, c_hig = ccfbootstrap(Y1, Y2, maxlags=200)
        plt.fill_between(lags, c_low, c_hig, alpha=.5)
        plt.plot(lags, c_med)
        plt.show()

if __name__ == '__main__':
    main()
