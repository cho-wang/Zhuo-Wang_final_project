import numpy as np
import astra

# This function calculates the gradient, objective function unless using
# OS, and the scatter if calculated on the fly.

def polyquant_grad(specData, A, vol_geom, proj_geom, proj_id, At, I0, rho, y, scatFun, subSet, w):
    projSet = [[0 for j in range(2)] for i in range(len(specData['hinge']) - 1)]
    mask = [[] for i in range(len(specData['hinge']) - 1)]
    projSet[0][1] = 0
    for k in range(len(specData['hinge']) - 1):
        mask[k] = np.where((rho > specData['hinge'][k]) & (rho < specData['hinge'][k + 1]), 1, 0)
        projSet[k][0] = A(np.dot(mask[k], rho), proj_id)   # check if you need to use np.dot or no.multuiply?
        # projSet[k][0] = astra.create_sino(np.dot(mask[k], rho), proj_id)
        if k > 1:
            projSet[k][1] = A(mask[k], proj_id)
            # projSet[k][1] = astra.create_sino(mask[k], proj_id)

    specProb = specData['spectrum'] / np.sum(specData['spectrum'])

    mainFac = np.zeros(np.shape(y))
    hingeFac = [[] for i in range(len(specData['hinge']) - 1)]
    for k in range(len(specData['hinge']) - 1):
        hingeFac[k] = np.zeros(np.shape(y))

    if len(specData['hinge']) > 2: # to bodge error for one linear fit
        s = scatFun(I0, projSet[0][0], projSet[1][0], projSet[1][1], rho, subSet, specData['knee'])
    else:
        s = scatFun(I0, projSet[0][0], projSet[0][0], projSet[0][1], rho, subSet, specData['knee'])

    for k in range(len(specData['spectrum'])):
        linSum = np.zeros(np.shape(y))
        for l in range(len(specData['hinge']) - 1):
            linSum = linSum + np.multiply(specData['knee'][0][l][k], projSet[l][0]) + np.multiply(
                specData['knee'][1][l][k], projSet[l][1])
        tmp = np.multiply(specProb[k], np.exp(-linSum))
        mainFac = mainFac + tmp
        for l in range(len(specData['hinge']) - 1):
            hingeFac[l] = hingeFac[l] + np.multiply(tmp, specData['knee'][0][l][k])

    mainFac = np.multiply(I0, mainFac)

    deriFac = np.multiply(w, (y / (mainFac + s)) - 1)
    out = np.zeros((rho.shape, rho.shape))

    for l in np.arange(1, len(specData['hinge']) - 1 + 1).reshape(-1):
        sino_g = np.multiply(np.multiply(I0, hingeFac[l])
                             
        sinog_id = astra.data2d.create('-sino', proj_geom)
                             
        out = out + np.multiply(mask[l], At(sinog_id, vol_geom))

    strOut = {}
    strOut['grad'] = out
    strOut['objFac'] = mainFac
    strOut['s'] = s

    return strOut
