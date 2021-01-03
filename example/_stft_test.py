import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def main():
    # setting
    infile1 = 'xdata.dat'
    infile2 = 'odata.dat'
    freqinfo = [0, 0.5, 10000]
    lambdainfo = [1e-1, 1e2, 20]
    nsegment = 1000
    noverlap = 500

    # read data
    xdata = np.loadtxt(infile1)
    odata = np.loadtxt(infile2)
    time = xdata[:,0]
    x = xdata[:,1]
    o = odata[:,1]

    f, t, Zxx_x = signal.stft(xdata[:,1], 1,
                              nperseg=nsegment, noverlap=noverlap)
    f, t, Zxx_o = signal.stft(odata[:,1], 1,
                              nperseg=nsegment, noverlap=noverlap)
    print(f't: {t.shape}')
    print(f't: {t}')
    print(f'f: {f.shape}')
    print(f'Zxx_x: {Zxx_x.shape}')
    print(f'Zxx_o: {Zxx_o.shape}')

    # plot
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].pcolormesh(t, f, np.log10(np.abs(Zxx_o)), shading='gouraud')
    ax[0].set_ylabel('Frequency [Hz]')
    ax[0].set_title('STFT Magnitude')

    ax[1].pcolormesh(t, f, np.log10(np.abs(Zxx_x)), shading='gouraud')
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    plt.show()

    _, xrec = signal.istft(Zxx_x, 1)
    plt.figure()
    plt.plot(_, xrec, time, x, time, o)
    plt.xlim(0, 500)
    plt.xlabel('Time [sec]')
    plt.ylabel('Signal')
    plt.legend(['Filtered via STFT', 'True Carrier'])
    plt.show()

if __name__ == '__main__':
    main()
