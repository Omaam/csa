import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

if __name__ == '__main__':

    fig, ax = plt.subplots(4,2, figsize=(10,10))
    lcname_list = [
                   ['./data/gx339_x.dat', './data/gx339_o.dat'],
                   ['./data/gx339_o_ma05s.dat', './data/gx339_x_ma05s.dat'],
                   ['./data/gx339_o_ma1s.dat', './data/gx339_x_ma1s.dat'],
                   ['./data/gx339_o_ma2s.dat', './data/gx339_x_ma2s.dat'],
                  ]

    for i, lcnames in enumerate(lcname_list):

        lcdata = np.loadtxt(lcnames[0])
        wave = lcdata[:,1]
        f, t, Zxx = signal.stft(wave, fs=20, nperseg=1000, noverlap=990)
        ax[i,0].pcolormesh(t, f, np.log10(1+np.abs(Zxx)), cmap='inferno')
        # ax[i,0].pcolormesh(t, f, np.abs(Zxx), cmap='inferno')
        ax[i,0].set_yscale('symlog', linthreshy=0.1)
        ax[i,0].set_title('X-ray')

        lcdata = np.loadtxt(lcnames[1])
        wave = lcdata[:,1]
        f, t, Zxx = signal.stft(wave, fs=20, nperseg=1000, noverlap=990)
        ax[i,1].pcolormesh(t, f, np.log10(1+np.abs(Zxx)), cmap='inferno')
        # ax[i,1].pcolormesh(t, f, np.abs(Zxx), cmap='inferno')
        ax[i,1].set_yscale('symlog', linthreshy=0.1)
        ax[i,1].set_title('Optical')

    plt.tight_layout()

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    print('save fig')
    fig.savefig('spectrogram_compare.png', transparent=True, dpi=300)
    plt.show()
