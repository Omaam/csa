import math
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import scipy.stats as stats
import seaborn as sns; sns.set()
sns.set_style("whitegrid")

from .signiftest import LagSignifTest

__all__ = ['read_summary']

def read_summary(path_to_summary, path_to_x, freq_info):

    df_sum = pd.read_csv(
        path_to_summary, sep=' ',
        names=["lag", "norm12", "norm1", "norm2", "period", "freq"])
    x_vec = np.loadtxt(path_to_x)[:, 1]

    return Summary(df_sum, x_vec, freq_info)

def limit_phase(phase_diff):
    if phase_diff < -np.pi: # phase_lag > pi
        phase_diff_out = 2*np.pi + phase_diff
    elif phase_diff >= np.pi:
        phase_diff_out = -2*np.pi + phase_diff
    else:
        phase_diff_out = phase_diff
    return phase_diff_out

def make_summary(x, freq_info):
    # x
    x_data = x.reshape(4, int(freq_info[2])).T
    df_x = pd.DataFrame(x_data, columns=['a', 'b', 'c', 'd'])
    # freq
    freq_lo = freq_info[0]
    freq_up = freq_info[1]
    n_freq = int(freq_info[2])
    delta_freq = (freq_up - freq_lo) / n_freq
    df_x['freq'] = freq_lo + delta_freq * (df_x.index.values + 0.5)
    df_x['period'] = 1 / df_x.freq.values
    # norm
    df_x['norm1'] = np.sqrt(df_x.a.values**2 + df_x.b.values**2)
    df_x['norm2'] = np.sqrt(df_x.c.values**2 + df_x.d.values**2)
    df_x['norm12'] = np.sqrt(df_x.norm1.values**2 + df_x.norm2.values**2)
    # lag
    df_x['alpha1'] = np.arctan2(df_x.b[df_x.a != 0], df_x.a[df_x.a != 0])
    df_x['alpha2'] = np.arctan2(df_x.d[df_x.c != 0], df_x.c[df_x.c != 0])
    df_x.fillna(0, inplace=True)
    delta_alpha = list(map(limit_phase, df_x.alpha2 - df_x.alpha1))
    df_x['lag'] = delta_alpha / (2 * np.pi * df_x.freq.values)

    df_sum = df_x[['lag', 'norm12', 'norm1', 'norm2', 'period', 'freq']]\
                 [df_x.norm12 > 0]

    return df_sum

def get_index_fromfreq(freq, freq_info):
    freq_lo = freq_info[0]
    freq_up = freq_info[1]
    n_freq = int(freq_info[2])
    delta_freq = (freq_up - freq_lo) / n_freq
    ind = (freq - freq_lo) / delta_freq -0.5
    print(ind)
    return ind

def get_round_of_2nd_digit(d, mode):
    d_abs = np.abs(d)
    log_d_abs = np.log10(d_abs)
    diff_dig_from_m1 = np.log10(0.1) - np.floor(log_d_abs)
    d_tmp = d * 10**(diff_dig_from_m1 + 1)
    d_tmp2 = np.floor(d_tmp) if mode == 'low' else np.ceil(d_tmp)
    d_out = d_tmp2 / 10**(diff_dig_from_m1 + 1)
    return d_out


def set_bins(low, hig, binsize):

    def get_deci_nonzero(a):
        a = float(round(a,10))
        deci_str = str(a).split(".")[1]
        for i, num in enumerate(list(deci_str)):
            if num != '0':
                break
        return i+1

    low = float(round(low, 10))
    hig = float(round(hig, 10))
    # Get digit
    min_abs = abs(low) if abs(low) <= abs(hig) else abs(hig)
    if min_abs > 1:
        nd_max = 0
    else:
        dig_low = get_deci_nonzero(low)
        dig_hig = get_deci_nonzero(hig)
        nd_max = dig_low if dig_low >= dig_hig else dig_hig
    # Float into integer
    low_tmp = np.floor(low * 10 ** nd_max)
    hig_tmp =  np.ceil(hig * 10 ** nd_max)
    binsize_tmp = binsize * 10 ** nd_max
    # Make bins
    bins = np.arange(low_tmp, hig_tmp + binsize_tmp, binsize_tmp)
    # Change digit into original
    bins = bins / 10 **nd_max
    return bins


class Summary:

    def __init__(self, df_sum, x_vec, freq_info):

        # summary
        self.df_sum = df_sum

        # freq
        self.freq_info = freq_info
        n_freq = int(freq_info[2])
        self.freq_list = self._make_freqlist(freq_info[0],
                                             freq_info[1],
                                             freq_info[2])
        self.freq_lo, self.freq_hi, self.n_freq = self.freq_info
        self.delta_freq = (self.freq_hi - self.freq_lo) / self.n_freq

        # x
        self.x = x_vec
        self.x1 = x_vec[:2 * n_freq]
        self.x2 = x_vec[ 2 * n_freq:]


    def __repr__(self):
        return repr(self.df_sum)

    def __add__(self, sum_add):

        # serch duplicated record
        id_add = []
        for i, freq_add in enumerate(sum_add.df_sum['freq'].values):
            for j, freq_ori in enumerate(self.df_sum['freq'].values):
                if freq_add == freq_ori:
                    id_add.append(i)
                    break
        id_rm = np.ones(len(sum_add.df_sum), dtype=bool)
        id_rm[id_add] = False

        # add
        df_sum_add = sum_add.df_sum[id_rm]
        df_sum_comb = pd.concat(
            [self.df_sum, df_sum_add]).sort_values('freq')
        return Summary(df_sum_comb.reset_index(drop=True),
                       self.x, self.freq_info)

    def __sub__(self, sum_sub):
        # serch duplicated record
        id_sub = ~self.df_sum.freq.isin(sum_sub.df_sum.freq)
        return Summary(self.df_sum[id_sub].reset_index(drop=True),\
               self.x, self.freq_info)

    def anti(self):
        df_sum_out = self.df_sum.copy()
        df_sum_out.loc[:,'lag'] = list(map(
            lambda lag, period: lag - period/2 if lag >= 0 else lag + period/2,
            df_sum_out['lag'],
            df_sum_out['period']))
        return Summary(df_sum_out, self.x, self.freq_info)

    def unique(self):
        df_sum_original = make_summary(self.x, self.freq_info)
        freq_list= self.df_sum.loc[:,'freq'].values
        df_sum_out = df_sum_original\
            [df_sum_original.freq.isin(freq_list)].reset_index(drop=True)
        return Summary(df_sum_out, self.x, self.freq_info)

    def threshold(self, threshold):
        df_sum_th = self.df_sum.copy()
        df_sum_th['ratio'] = list(map(
                                lambda norm1, norm2: norm1/norm2
                                    if norm1 <= norm2 else norm2/norm1,
                                df_sum_th['norm1'].values,
                                df_sum_th['norm2'].values
                                ))
        df_sum_th = df_sum_th.query('ratio > @threshold').copy()\
                        .reset_index(drop=True)
        df_sum_th.drop('ratio', axis=1, inplace=True)
        return Summary(df_sum_th, self.x, self.freq_info)

    def extract(self, ext_ranges):
        # serch
        id_list = []
        for j,lag in enumerate(self.df_sum['lag'].values):
            for ext_range in ext_ranges:
                if ext_range[0] <= lag < ext_range[1]:
                    id_list.append(j)
                    break
        # choice
        df_sum_out = self.df_sum.iloc[id_list,:].reset_index(drop=True)
        return Summary(df_sum_out, self.x, self.freq_info)

    def cut(self, cut_ranges):

        # serch
        id_add = []
        for j,lag in enumerate(self.df_sum['lag'].values):
            for cut_range in cut_ranges:
                if cut_range[0] <= lag < cut_range[1]:
                    id_add.append(j)
                    break
        # choice
        id_rm = np.ones(len(self.df_sum), dtype=bool)
        id_rm[id_add] = False
        df_sum_out = self.df_sum.iloc[id_rm,:].reset_index(drop=True)
        return Summary(df_sum_out, self.x, self.freq_info)

    def make_periodicsum(self, lagrange):
        columns = self.df_sum.columns.values
        ndarray_sum = self.df_sum.values
        for para in list(ndarray_sum):
            (lag, norm12, norm1, norm2, period, freq) = para
            # plus
            i = 0
            lag_peri = lag
            while lagrange[0] < lag + (i*period) <= lagrange[1]:
                lag_peri = lag + (i*period)
                new_col = np.array([lag_peri, norm12, norm1,
                                    norm2, period, freq])
                ndarray_sum = np.vstack([ndarray_sum, new_col])
                i += 1
            # munus
            i = -1
            lag_peri = lag - period
            #while lag_peri > lagrange[0]:
            while lagrange[0] < lag + (i*period) <= lagrange[1]:
                lag_peri = lag + (i*period)
                new_col = np.array([lag_peri, norm12, norm1,
                                    norm2, period, freq])
                ndarray_sum = np.vstack([ndarray_sum, new_col])
                i += -1
        df_sum_out = pd.DataFrame(ndarray_sum, columns=columns)
        return Summary(df_sum_out, self.x, self.freq_info)


    def pred(self, t1, t2):
        # extract x from sum
        x1_out = self._make_x(self.x1, self.df_sum, self.freq_list)
        x2_out = self._make_x(self.x2, self.df_sum, self.freq_list)

        # make y
        A1 = self._make_matrix_dft(t1, self.freq_list)
        A2 = self._make_matrix_dft(t2, self.freq_list)
        x1 = self._make_x(x1_out, self.df_sum, self.freq_list)
        x2 = self._make_x(x2_out, self.df_sum, self.freq_list)
        y1 = np.dot(A1, x1)
        y2 = np.dot(A2, x2)

        return y1, y2

    def psd(self):

        # extract x from sum
        x1_out = self._make_x(self.x1, self.df_sum, self.freq_list)
        x2_out = self._make_x(self.x2, self.df_sum, self.freq_list)

        n_freq = self.freq_info[2]
        power1_vec = list(map(
            lambda a, b: np.sqrt(a**2+b**2),
            x1_out[n_freq:],
            x1_out[:n_freq]
        ))
        power2_vec = list(map(
            lambda a, b: np.sqrt(a**2+b**2),
            x2_out[n_freq:],
            x2_out[:n_freq]
        ))

        return np.array(power1_vec), np.array(power2_vec)

    def resfunc(self, t):
        # extract x from sum
        x1_out = self._make_x(self.x1, self.df_sum, self.freq_list)
        x2_out = self._make_x(self.x2, self.df_sum, self.freq_list)
        x_out = np.hstack([x1_out, x2_out])

        # make x of resfunc
        n_freq = self.freq_info[2]
        a = x_out[0 * n_freq : 1 * n_freq]
        b = x_out[1 * n_freq : 2 * n_freq]
        c = x_out[2 * n_freq : 3 * n_freq]
        d = x_out[3 * n_freq : 4 * n_freq]
        delta_f = self.freq_list[1] - self.freq_list[0]
        x_res_real = np.array(list(map(
            lambda a,b,c,d:(a*c+b*d)/(b**2+a**2)/(2*delta_f)\
                if a != b else 0,
            a, b, c, d)))
        x_res_imag = np.array(list(map(
            lambda a,b,c,d:(a*d-b*c)/(b**2+a**2)/(2*delta_f)\
                if a != b else 0,
            a, b, c, d)))
        x_res = np.hstack([x_res_real, x_res_imag])

        # make y
        A = self._make_matrix_dft(t, self.freq_list)
        y = np.dot(A, x_res)
        return y

    def hist(self, lagbins, density=True):
        df_sum_pow = self.df_sum.copy()
        hist_value = pd.cut(df_sum_pow.lag.values, bins=lagbins)\
            .value_counts().sort_index().values
        if density:
            hist_value = hist_value / len(self.df_sum)
        bins_center = (lagbins[:-1] + lagbins[1:]) / 2
        return bins_center, hist_value

    def powerhist(self, lagbins, density=True):
        df_sum_pow = self.df_sum.copy()
        df_sum_pow['id_bins'] = pd.cut(df_sum_pow.lag.values,
                                       bins=lagbins)
        powerhist_value = df_sum_pow.groupby('id_bins')['norm12']\
                            .sum().values
        if density:
            powerhist_value = powerhist_value / df_sum_pow.norm12.sum()
        bins_center = (lagbins[:-1] + lagbins[1:]) / 2
        return bins_center, powerhist_value


    def plot_cbplot(self, threshold=None, lagrange=None, ax=None):

        import matplotlib.cm as cm
        if ax == None:
            fig, ax = plt.subplots()
        # color babble plot
        size = list(map(
            lambda norm1, norm2: norm1/norm2 if norm1 <= norm2
                                             else norm2/norm1,
            self.df_sum.norm1.values,
            self.df_sum.norm2.values))
        size_log = np.log10(1 + np.array(size)) * 750
        im = ax.scatter(self.df_sum['lag'].values,
                        self.df_sum['norm12'].values,
                        c=self.df_sum['freq'].values, cmap=cm.jet,
                        alpha=0.8, s=size_log,
                        norm=Normalize(self.freq_info[0],
                        self.freq_info[1]))
        ax.set_yscale('log')
        ax.set_ylim(np.min(self.df_sum.norm12))
        ax.set_ylabel('Power')
        ax.set_xlabel('Lag')
        if lagrange:
            ax.set_xlim(lagrange)

        # colorbat setting
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="7%", pad="2%")
        plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_text("Frequency")
        cax.xaxis.set_label_position("top")

        return ax

    def plot_omplot(self, lagrange=None, lagbinwidth=None,
                    lag_mode='standard', ci_list=None, ci_label=None):
        ''' plot omplot which is combined with color babble plot and
            histgram and poer histgram

        ci_list: (lag, ci_value)
        ci_label: ci name e.g. 'ci 95%'
        '''
        fig = plt.figure(figsize=(8, 2*2+2))
        height = 0.8/2

        # periodic sum
        if lag_mode == 'periodic':
            summ_peri = self.make_periodicsum(lagrange=lagrange)
            self.df_sum = summ_peri.df_sum

        # get hist value
        ax0 = fig.add_axes([0.10, 0.1+0*height, 0.85, height-0.2])
        if lagrange == None:
            lag_min = self.df_sum.lag.min()
            lag_max = self.df_sum.lag.max()
        else:
            lag_min = lagrange[0]
            lag_max = lagrange[1]
        if lagbinwidth == None:
            lagbinwidth = (lag_max - lag_min) / 20
        lagbins = set_bins(lag_min, lag_max, lagbinwidth)

        # hist and power hist
        bins_center, hist_value = self.hist(lagbins)
        bins_center, phist_value = self.powerhist(lagbins)
        ax0.bar(bins_center, hist_value,  color="r", width=lagbinwidth,
                align="center", alpha=0.5, edgecolor="k", label="Number")
        ax0.bar(bins_center, phist_value, color="b", width=lagbinwidth,
                align="center", alpha=0.5, edgecolor="k", label="Power")
        if ci_list:
            ax0.plot(ci_list[0], ci_list[1], label=ci_label,
                     color='r', alpha=0.7)
        ax0.set_ylabel('Density')
        ax0.set_xlabel('Lag')
        ax0.legend(loc='best')

        # color babble plot
        ax1 = fig.add_axes([0.10, 0.1+1*height-0.2, 0.85, height+0.2],
                           sharex=ax0)
        ax1 = self.plot_cbplot(ax=ax1, lagrange=lagrange)
        ax1.set_xlabel(None)
        plt.setp(ax1.get_xticklabels(), visible=False)

        return fig

    def _make_freqlist(self, freq_lo, freq_hi, n_freq):
        delta_freq = (freq_hi - freq_lo) / n_freq
        freq_list = np.round(
            np.linspace(
                freq_lo + delta_freq,
                freq_hi,
                n_freq) - delta_freq / 2, 10)
        return freq_list

    def _make_matrix_dft(self, t_list, freq_list):
        '''
        make matrix from time and freq
        '''
        delta_freq = freq_list[1] - freq_list[0]
        matrix_phase = 2 * np.pi * np.dot(
            np.array([t_list]).T, np.array([freq_list]))
        matrix_cos = 2 * delta_freq * np.cos(matrix_phase)
        matrix_sin = 2 * delta_freq * np.sin(matrix_phase)
        matrix = np.hstack([matrix_cos, matrix_sin])
        return matrix


    def _make_x(self, x_vec, df_sum, freq_list):
        """make x vector from df_sum
        Return
        x_out: arraylike
            x vector wchich is calcurated by component whare demand lag
        """
        # x
        x_vec = np.array(x_vec)
        #
        flg_list = np.zeros(len(freq_list))
        for sum_freq in df_sum['freq'].values:
            rep_sum_freq = np.repeat(np.round(sum_freq,10),
                                     len(freq_list))
            flg_list_tmp = np.array(list(map(
                lambda x_freq, sum_freq: 1 if x_freq == sum_freq else 0,
                freq_list,
                rep_sum_freq)))
            flg_list += flg_list_tmp
        flg_list = np.tile(flg_list, 2)
        x_out = x_vec * flg_list

        return x_out

class SummaryNew(Summary):

    def __init__(self, x_vec, freq_info):

        # summary
        self.df_sum = make_summary(x_vec, freq_info)

        # freq
        self.freq_info = freq_info
        n_freq = int(freq_info[2])
        self.freq_list = self._make_freqlist(freq_info[0],
                                             freq_info[1],
                                             freq_info[2])
        self.freq_lo, self.freq_hi, self.n_freq = self.freq_info
        self.delta_freq = (self.freq_hi - self.freq_lo) / self.n_freq

        # x
        self.x = x_vec
        self.x1 = x_vec[:2 * n_freq]
        self.x2 = x_vec[ 2 * n_freq:]

if __name__ == "__main__":

    # summ = read_summary('example/out/sum.dat',
    #                     'example/out/x.dat', (0,10,2000))
    summ = read_summary('example2/out/sum.dat', 'example2/out/x.dat',
                        [0,0.5,200])
    freq_list = summ.freq_list

    # summ = summ.ext_from_sum([[0.1,0.3]])

    summ1 = summ.extract([[4,9]])
    summ2 = summ.extract([[6,9]])
    print(summ1)
    print(summ2)
    print(summ1 - summ2)

    # get value
    t_list = np.linspace(0, 100, 1000)
    y1, y2 = summ.pred(t_list, t_list)
    p1, p2 = summ.psd()

    # resfunc
    # t_res = np.linspace(-5,5,100)
    # y_res = summ.resfunc(t_res)
    # plt.plot(t_res, y_res)
    # plt.show()

    # # psd
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(freq_list, p1)
    # ax[1].plot(freq_list, p2)
    # plt.show()

    # pred
    data_lc1 = np.loadtxt('example2/data/lc1.dat')
    data_lc2 = np.loadtxt('example2/data/lc2.dat')
    t1 = data_lc1[:,0]
    f1 = data_lc1[:,1]
    t2 = data_lc2[:,0]
    f2 = data_lc2[:,1]
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(t1, f1, alpha=0.7, label='Obs')
    ax[0].plot(t_list, y1, alpha=0.7, label='Model')
    ax[0].set_ylabel('X-ray')
    ax[0].set_xlabel('Time')
    ax[0].legend(loc='best')
    ax[1].plot(t2, f2, alpha=0.7)
    ax[1].plot(t_list, y2, alpha=0.7)
    ax[1].set_ylabel('Optical')
    ax[1].set_xlabel('Time')
    plt.subplots_adjust(hspace=0)
    plt.show()

    # omplot
    summ.plot_omplot(lagbinwidth=1)
    plt.show()
