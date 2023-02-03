import numpy as np
import scipy.io as sio

attenuationDb = sio.loadmat('C:\pythonFinalProject/attenuationDb.mat')

def mat_to_den(attenuationDb = None,matIm = None): 
    comp = attenuationDb.comp(np.arange(1,53+1),:)
    dens = attenuationDb.density(np.arange(1,53+1))
    Z = np.array([1,6,7,8,11,12,15,16,17,19,20,26,53])
    A = np.array([1.01,12.01,14.01,16.0,22.99,24.305,30.97,32.066,35.45,39.098,40.08,55.845,126.9])
    eDen = comp * (np.transpose(Z) / np.transpose(A))
    waterE = 0.1119 * 1 / 1.01 + 0.8881 * 8 / 16
    relDen = np.multiply(dens,eDen) / waterE
    eden = matIm
    mden = matIm
    for k in np.arange(1,53+1).reshape(-1):
        eden[matIm == k] = relDen(k)
        mden[matIm == k] = dens(k)
    
    mden[matIm == 54] = 4.506
    eden[matIm == 54] = 3.7326

    return eden,mden
