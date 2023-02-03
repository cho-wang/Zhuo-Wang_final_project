import scipy.io as sio
import numpy as np
import scipy.io as sio

attenuationDb = sio.loadmat('C:\pythonFinalProject/attenuationDb.mat')

def mat_to_den (attenuationDb,matIm):
    comp = attenuationDb['comp'][0:53,:]
    dens = attenuationDb['density'][0:53]
    Z = [1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26, 53]
    A = [1.01, 12.01, 14.01, 16.00, 22.99, 24.305, 30.97, 32.066, 35.45, 39.098, 40.08, 55.845, 126.90]
    eDen = comp * (np.divide(Z.T,A.T))
    waterE = 0.1119 * 1 / 1.01 + 0.8881 * 8 / 16
    relDen = dens*(np.divide(eDen,waterE))
    eden = matIm
    mden = matIm
    for k in range(1,53,1):
        eden[matIm == k] = relDen[k]
        mden[matIm == k] = dens[k]

    mden[matIm == 54] = 4.506
    eden[matIm == 54] = 3.7326

    return (eden,mden)