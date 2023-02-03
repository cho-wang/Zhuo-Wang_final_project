import numpy as np
    
def polyquant_grad(specData = None,A = None,At = None,I0 = None,rho = None,y = None,ind = None,scatFun = None,subSet = None,w = None): 
    # This function calculates the gradient, objective function unless using
# OS, and the scatter if calculated on the fly.
    projSet = cell(len(specData.hinge) - 1,2)
    mask = cell(len(specData.hinge) - 1,1)
    projSet[1,2] = 0
    for k in np.arange(1,len(specData.hinge) - 1+1).reshape(-1):
        mask[k] = double(rho > np.logical_and(specData.hinge(k),rho) < specData.hinge(k + 1))
        projSet[k,1] = A(np.multiply(mask[k],rho),ind)
        if k > 1:
            projSet[k,2] = A(mask[k],ind)
    
    specProb = specData.spectrum / sum(specData.spectrum)
    mainFac = np.zeros((y.shape,y.shape))
    hingeFac = cell(len(specData.hinge) - 1)
    for k in np.arange(1,len(specData.hinge) - 1+1).reshape(-1):
        hingeFac[k] = np.zeros((y.shape,y.shape))
    
    if len(specData.hinge) > 2:
        s = scatFun(I0,projSet[1,1],projSet[2,1],projSet[2,2],rho,subSet,specData.knee)
    else:
        s = scatFun(I0,projSet[1,1],projSet[1,1],projSet[1,2],rho,subSet,specData.knee)
    
    for k in np.arange(1,len(specData.spectrum)+1).reshape(-1):
        linSum = np.zeros((y.shape,y.shape))
        for l in np.arange(1,len(specData.hinge) - 1+1).reshape(-1):
            linSum = linSum + specData.knee(1,l,k) * projSet[l,1] + specData.knee(2,l,k) * projSet[l,2]
        tmp = np.multiply(specProb(k),np.exp(- linSum))
        mainFac = mainFac + tmp
        for l in np.arange(1,len(specData.hinge) - 1+1).reshape(-1):
            hingeFac[l] = hingeFac[l] + tmp * specData.knee(1,l,k)
    
    mainFac = np.multiply(I0,mainFac)
    deriFac = w(y / (mainFac + s) - 1)
    out = np.zeros((rho.shape,rho.shape))
    for l in np.arange(1,len(specData.hinge) - 1+1).reshape(-1):
        out = out + np.multiply(mask[l],At(np.multiply(np.multiply(I0,hingeFac[l]),deriFac),ind))
    
    strOut.grad = out
    strOut.objFac = mainFac
    strOut.s = s
    return strOut
    
    return strOut