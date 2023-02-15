import numpy as np
from numpy.linalg import norm
import initialise_mode
import polyquant_grad_f
import matplotlib.pyplot as plt
import offset_weight_f
import lipscitz_estimate_f

def ismatrix(arr):
    return np.ndim(arr) == 2

def polyquant(mode, specData, y, I0, Af, xTrue):

# %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# % Performs direct quantitative reconstruction from polyergetic data.
# %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# % Parameters
# % ----------
# % mode          -- structure containing the settings and functions:
# % (all these settings have default values: see initialise_mode)
# %   mode.tau          -- stepsize scaling factor (< 2 is conservative).
# %   mode.maxIter      -- number of iterations.
# %   mode.nest         -- use FISTA-like Nesterov acceleration.
# %   mode.nSplit       -- number of ordered subset divisions (1 is full).
# %   mode.verbose      -- output settings: 0 = silent; 1 = text; 2 = figure.
# %   mode.contrast     -- display contrast for output live updat figure.
# %   mode.regFun       -- handle to regularisation function.
# %   mode.proxFun      -- handle to proximity operator for regularisation.
# %   mode.scatFun      -- scatter estimation function (see poly_sks.m).
# %   mode.useConst     -- offset objective function to better range.
# %   mode.bitRev       -- use subset shuffling (bit-reversal ordering).
# %   mode.offset       -- use Wang offset detector weighting for half-fan.
# %   mode.L            -- supplying Lipschitz estimate will save time.
# % specData      -- structure containing spectral information:
# %   specData.energy   -- the energies (MeV) in the subsampled spectrum.
# %   specData.spectrum -- the subsampled source spectrum.
# %   specData.response -- the detector response function.
# %   specData.hinge    -- the location of the piecewise linear fit
# %                        transitions, for 3 linear sections.
# %   specData.knee     -- contains the equations for the piecewise linear
# %                        fits between relative electron density and each
# %                        energy in specData.energy. This was fitted against
# %                        the biological materials in the ICRP 89 and for
# %                        titanium (density = 4.506 g/cm3).
# % y             -- the raw X-ray CT measurements.
# % I0            -- the incident flux profile.
# % Af            -- the CT system operator generated from Fessler's toolbox.
# % xTrue         -- ground truth image (can be 0 if unknown).
# %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# % Created:      07/03/2018
# % Last edit:    02/06/2019
# % Jonathan Hugh Mason
# %
# %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# % References: (please cite if making use of this code or its methods)
# % Jonathan H Mason et al 2017 Phys. Med. Biol. 62 8739
# % Jonathan H Mason et al 2018 Phys. Med. Biol. 63 225001
# %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ## Initialisation
    mode = initialise_mode(mode);

    if len(y.shape) == 2:
        x0 = np.ones(np.shape(Af["arg"]["mask"]))
    else:
        x0 = np.ones(Af["arg"]["ig"]["dim"])

    Ab = Gblock(Af, mode["nSplit"]); #?????

    def A(x, ind):
        return np.dot(Ab[ind], x)

    def At(p, ind):
        return np.dot(Ab[ind].T, p)

    if mode["offset"]:
        def w(z):
            return offset_weight_f(z, mode["cg"])
    else:
        def w(z):
            return z

    if hasattr(mode, 'numLinFit'):
        if mode["numLinFit"] < len(specData["hinge"]):
            specData["hinge"] = specData["hinge"][:mode["numLinFit"]] + [float('inf')]

    if hasattr(specData, 'response'):
        specData["spectrum"] *= specData["response"]

    if 'L' not in mode:  # estimate Lipschitz if unknown
        mode['L'] = lipscitz_estimate_f(specData, I0, mode['scat'], y, Ab(x0), Af)

    alpha = mode['nSplit'] * mode['tau'] / mode['L']  # the step-size

    if mode['useConst']:
        const = y - y * np.log(y + np.finfo(float).eps)
        const = np.sum(const.ravel())  # a constant offset for objective function
    else:
        const = 0

    x1 = x0
    timeTot = time.time() #????

    if mode.nest:
        t = 1

    out = {}
    out['rmse'][0] = np.sqrt(np.mean((x1 - xTrue) ** 2))

    if mode.verbose == 2:

        if xTrue.ndim == 3:
            plt.subplot(2, 3, 1)
            plt.imshow(np.rot90(xTrue[:, :, 20], k=-1), cmap=mode.contrast)
            plt.subplot(2, 3, 2)
            plt.imshow(np.rot90(xTrue[:, :, 30], k=-1), cmap=mode.contrast)
            plt.title('ground truth')
            plt.subplot(2, 3, 3)
            plt.imshow(np.rot90(xTrue[:, :, 40], k=-1), cmap=mode.contrast)
        else:
            plt.subplot(2, 1, 1)
            plt.imshow(xTrue, cmap=mode.contrast)
            plt.title('ground truth')
            plt.subplot(2, 1, 2)

        plt.show()

    def grAx(x1, is_, ys, ind, subSet):
        return polyquant_grad_f(specData, A, At, is_, x1, ys, ind, mode.scatFun, subSet, w)

    objFac = np.zeros_like(y)
    out['scat'] = np.zeros_like(y)

    # The main iterative loop

    if mode['verbose'] > 0:
    print('Starting Polyquant reconstruction:')

    for k in range(mode['maxIter']):
    ind = (k % mode['nSplit']) + 1
    if mode['bitRev']:
        ind = bit_rev(ind - 1, mode['nSplit']) + 1 #???

    subSet = np.arange(ind - 1, np.size(y, np.ndim(x0)), mode['nSplit'])

    if np.ndim(x0) == 3:
        is_ = I0[:, :, subSet]
        ys = y[:, :, subSet]
    else:
        is_ = I0[:, subSet]
        ys = y[:, subSet]

    gradAx = grAx(x1, is_, ys, ind, subSet)

    if np.ndim(x0) == 3:
        out['scat'][:, :, subSet] = gradAx.s
        objFac[:, :, subSet] = gradAx.objFac
    else:
        out['scat'][:, subSet] = gradAx.s
        objFac[:, subSet] = gradAx.objFac

    xNew = mode['proxFun'](x1 - alpha * gradAx.grad, alpha)

    if mode['nest']:
        t1 = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        x1 = xNew + (t - 1) / t1 * (xNew - x0)
        x0 = xNew
        t = t1
    else:
        x1 = xNew

    out['rmse'][k + 1] = norm(x1.flatten() - xTrue.flatten()) / norm(xTrue.flatten())
    out['obj'][k + 1] = np.sum(objFac.flatten() + out['scat'].flatten() - y.flatten() * np.log(
        objFac.flatten() + out['scat'].flatten() + np.finfo(float).eps)) - const + mode['regFun'](x1)

    if mode['verbose'] > 0:
        print('\rIter = %i;\t RMSE = %.4e;\t obj = %.4e;\t subset = %i    ' % (k, out.rmse[k + 1], out.obj[k + 1], ind),
              end='')

    if mode['verbose'] == 2:
        str_ = 'polyquant at iteration: %i' % k
        if np.ndim(x1) == 3:
            plt.subplot(2, 3, 4)
            plt.imshow(np.rot90(x1[:, :, 20], 3), mode.contrast)
            plt.subplot(2, 3, 5)
            plt.imshow(np.rot90(x1[:, :, 30], 3), mode.contrast)
            plt.title(str_)
            plt.subplot(2, 3, 6)
            plt.imshow(np.rot90(x1[:, :, 40], 3), mode.contrast)
        else:
            plt.imshow(x1, mode['contrast'])
            plt.title(str_)
        plt.draw()

        time = time.time() - timeTot
        out['time'] = time
        out['recon'] = xNew
        if mode['verbose'] > 0:
            print(f'\nFinished in {time:.2e} seconds')

    return out

#I guess.
def bit_rev(x):
    N = len(x)
    bits = np.binary_repr(np.arange(N), width=int(np.log2(N)))
    indices = [int(b[::-1], 2) for b in bits]
    return x[indices]




