from concurrent.futures import ProcessPoolExecutor, as_completed
import os
ncpu = os.cpu_count()
MAX_WORKERS = ncpu
print(f'number of cpu: {ncpu}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


def set_bins(low, hig, binsize):
    low = float(round(low, 10))
    hig = float(round(hig, 10))
    # Get digit
    dig_low = str(low).split(".")[1]
    dig_hig = str(hig).split(".")[1]
    nd_max = len(dig_low) if len(dig_low) >= len(dig_hig) else len(dig_hig) 
    # Float into integer
    low_trans = low * np.power(10, nd_max)
    hig_trans = hig * np.power(10, nd_max)
    binsize_trans = binsize * np.power(10, nd_max)
    # Make bins
    bins = np.arange(low_trans, hig_trans + binsize_trans, binsize_trans)
    # Change digit into original
    bins = bins / np.power(10, nd_max)
    return bins

def _make_model(periods, lagbins):
        lags = np.random.rand(len(periods))*periods - periods/2
        lagdistmodel = pd.cut(lags, bins=lagbins).value_counts().values
        return lagdistmodel

class LagSignifTest:

    def __init__(self, df_sum, lagrange=[-0.5, 0.5], lag_binwidth=.1):
        '''
        basic flow is below
            1. make random sample
            2. compare sample with model
        '''
        # summary
        self.lag = df_sum['lag'].values
        self.norm12 = df_sum['norm12'].values
        self.norm1 = df_sum['norm1'].values
        self.norm2 = df_sum['norm2'].values
        self.period = df_sum['period'].values
        self.freq = df_sum['freq'].values

        # lagrange
        self.lagrange = lagrange
        self.lag_binwidth = lag_binwidth


    def make_model(self, iteration=1000):
        '''make rondom model
        '''
        lagbins = set_bins(self.lagrange[0], self.lagrange[1], self.lag_binwidth)
        lagdistmodels = np.zeros((iteration, lagbins.shape[0]-1))
        with ProcessPoolExecutor(MAX_WORKERS) as executor:
            futures = [executor.submit(_make_model,
                                      periods=self.period, lagbins=lagbins)
                       for i in range(iteration)]
            for i, future in enumerate(futures):
                lagdistmodels[i] = future.result()

        self.lagbins = lagbins
        self.iteration = iteration
        self.n_model_tile = lagdistmodels



    def get_signifrange(self, ci=.68, retbins=False, verbose=False):
        ''' get significance lag range
        '''
        lagbins = set_bins(self.lagrange[0], self.lagrange[1], self.lag_binwidth)
        laglabel_list = np.array(list(zip(lagbins[:-1], lagbins[1:])))
        n_sample_list = pd.cut(self.lag, bins=lagbins).value_counts().values
        
        n_atci_list = []
        for i in range(len(self.n_model_tile[0,:])):
            n_atci = np.percentile(self.n_model_tile[:,i], 100*ci)
            n_atci_list.append(n_atci)
        n_atci_list = np.array(n_atci_list)
        lag_ci_list = laglabel_list[n_sample_list >= n_atci_list]
        if verbose:
            print('lag   : {}'.format(lagbins))
            print('sample: {}'.format(n_sample_list))
            print('ci {}%: {}'.format(int(ci*100), n_atci_list))
        if retbins:
            return lag_ci_list, lagbins
        else:
            return lag_ci_list

    def get_civalue(self, ci):
        ''' get cignificance value
        '''
        n_atci_list = []
        for i in range(len(self.n_model_tile[0,:])):
            n_atci = np.percentile(self.n_model_tile[:,i], 100*ci)
            n_atci_list.append(n_atci)
        n_atci_list = np.array(n_atci_list)
        return self.lagbins, n_atci_list


if __name__ == "__main__":
    
    df_sum = pd.read_csv(
        'example/out/sum.dat', sep=' ',
        names=['lag', 'norm12', 'norm1', 'norm2', 'period', 'freq'])

    tester = LagSignifTest(df_sum)
    tester.make_model(iteration=1000)
    signif_ci90 = tester.get_signifrange(ci=.9)
    print('signif range: {}'.format(signif_ci90))
    bins, ci90_list = tester.get_civalue(0.90)
    print('bins : {}'.format(bins))
    print('ci 90: {}'.format(ci90_list))
