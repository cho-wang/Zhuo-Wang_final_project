import numpy as np
import numpy.matlib
    
def offset_weight(proj = None,cg = None): 
    # Offset weighting for half-fan case from [G. Wang, Med Phys. 2002]
    out = proj
    us = (np.arange((cg.ns / 2 - 0.5),(- cg.ns / 2 + 0.5)+- 1,- 1)) * cg.ds - cg.offset_s * cg.ds
    overlap = np.amax(us)
    overLoc = sum(np.abs(us) <= overlap)
    replaceLoc = np.arange(1,overLoc+1)
    denom = 2 * np.arctan(overlap / cg.dsd)
    num = np.pi * np.arctan(us(replaceLoc) / cg.dsd)
    #weightArray = 1-cos(linspace(0,pi/2,overLoc)).^2;
    weightArray = 1 - 0.5 * (np.sin(num / denom) + 1)
    weightMat = np.matlib.repmat(np.transpose(weightArray),1,cg.nt)
    replaceLoc = np.arange(1,weightMat.shape[1-1]+1)
    for k in np.arange(1,proj.shape[3-1]+1).reshape(-1):
        out[end() - replaceLoc + 1,:,k] = np.multiply(proj(end() - replaceLoc + 1,:,k),weightMat)
    
    return out
    
    return out