import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore

if __name__ == '__main__':

    plt.rcParams["font.size"] = 18
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in


    # spectrogram
    fig, ax = plt.subplots(2,2, figsize=(8,5))
    lcname_list = [['./gx339_o_original.dat',
                    'FIR2Hz/gx339_o_fir_original.dat'],
                   ['./gx339_x_original.dat',
                    'FIR2Hz/gx339_x_fir_original.dat']]
    texts = ['(a)', '(b)', '(c)', '(d)']

    for i, lcnames in enumerate(lcname_list):
        lcdata = np.loadtxt(lcnames[0])
        wave = zscore(lcdata[:,1])
        f, t, Zxx = signal.spectrogram(wave, fs=20, nperseg=1000, noverlap=900)
        print(np.log10(Zxx.min()))
        print(np.log10(Zxx.max()))
        ax[i,0].pcolormesh(t, f, np.log10(Zxx), cmap='inferno', vmin=-4)
        ax[i,0].set_ylim(1e-2, 1e1)
        ax[i,0].set_yscale('symlog', linthreshy=0.1)
        ax[i,0].text(1100, 4, texts[i*2+0], color='w', size=15)

        lcdata = np.loadtxt(lcnames[1])
        wave = zscore(lcdata[:,1])
        f, t, Zxx = signal.spectrogram(wave, fs=20, nperseg=1000, noverlap=990)
        print(np.log10(Zxx.min()))
        print(np.log10(Zxx.max()))
        ax[i,1].pcolormesh(t, f, np.log10(Zxx), cmap='inferno', vmin=-4)
        ax[i,1].set_ylim(1e-2, 1e1)
        ax[i,1].set_yscale('symlog', linthreshy=0.1)
        ax[i,1].text(1100, 4, texts[i*2+1], color='w', size=15)
    # fig.tight_layout()

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel(r'$t\;{\rm (s)}$')
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.13, top=0.95)
    # fig.tight_layout()

    # save
    print('save fig')
    fig.savefig('spectrogram_FIR7Hz.png', transparent=True, dpi=300)
    # fig.savefig('spectrogram.pdf', transparent=True, dpi=100)
    plt.show()
