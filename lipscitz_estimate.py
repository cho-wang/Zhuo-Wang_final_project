import numpy as np

def lipscitz_estimate(specData,I0,s,y,flat,At):
    specProb = np.devide(specData['spectrum'],np.sum(specData['spectrum']))
    tmpA = 0;

    for k in range(1,len(specData['spectrum'])):
        tmpA = tmpA + specProb(k) * specData['knee'](1, 1, k)**2

    fac = I0 * (np.devide(1-y * s,((I0 + s)**2)))
    p2A = At.conj().transpose() * (flat*tmpA*fac)
    out = max(p2A)

    return out
