import numpy as np
import pandas as pd

def fftime(start, end):
    theta = np.arccos(np.sqrt(end/start))
    t = (1/ALPHA) * np.sqrt(start**3 * (theta+np.sin(theta)*np.cos(theta)) / (2 * G * M))
    return t

G = 6.67e-11 # m^3 kg^-1 s-2
Ms = 2e30 # kg
M = 6 * Ms
ALPHA = 0.01
if __name__ == '__main__':
    t = 5
    r = (2*G*M)**(1/3) * t**(1/3)
    print(r)
    # t = fftime(1e7, 1e4)
    # print('free fall time = {}'.format(t))
    
