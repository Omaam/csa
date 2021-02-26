import os
import re
import platform
import glob
import logging

import numpy as np
import matplotlib.pyplot as plt

from csa.cvresult import show_cvdata


def main():
    # get lam value from previous cv data
    cvfiles = glob.glob('cv/cvdata*')
    lams = []
    for cvfile in cvfiles:
        cvdata = np.loadtxt(cvfile)
        show_cvdata(cvdata)
        plt.show()


if __name__ == '__main__':
    main()
