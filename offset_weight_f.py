import numpy as np
# Offset weighting for half-fan case from [G. Wang, Med Phys. 2002]

def offset_weight(proj, cg):
    out = proj.copy()
    us = np.arange((cg["ns"] / 2 - 0.5), (-cg["ns"] / 2+0.5), -1) * cg["ds"] - cg["offset_s"] * cg["ds"]
    overlap = max(us)
    overLoc = sum(abs(us) <= overlap)
    replaceLoc = range(1, overLoc + 1)
    denom = 2 * np.arctan(overlap / cg["dsd"])
    num = np.pi * np.arctan(us[replaceLoc - 1] / cg["dsd"])
    #weight_array = 1 - np.cos(np.linspace(0, np.pi/2, overloc))**2
    weightArray = 1 - 0.5 * (np.sin(num / denom) + 1)
    weightMat = np.tile(weightArray, (1, cg["nt"]))
    replaceLoc = range(1, weightMat.shape[0] + 1)

    for k in range(proj.shape[2]):
        out[-replaceLoc, :, k] = proj[-replaceLoc, :, k] * weightMat

    return out