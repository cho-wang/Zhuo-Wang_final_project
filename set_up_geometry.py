import astra
import numpy as np

vol_geom = astra.create_vol_geom(137, 299)
# proj_geom = astra.create_proj_geom('parallel', 85/512, 512, np.linspace(0,np.pi,360,False), 64.5, 55.8)
proj_geom = astra.create_proj_geom('parallel', 1, 512, np.linspace(0,np.pi,360,False))

import scipy.io
P = scipy.io.loadmat('fan_mat_d.mat')['fan_mat']

proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sinogram_id, Ax = astra.create_sino(P, proj_id)

I0 = 1e3

sinogram = astra.functions.add_noise_to_sino(Ax, I0, seed=None)  # I = I0 exp(-Ax) --> sinogram = -log(I0/I)

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

# create configuration
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['option'] = { 'FilterType': 'Ram-Lak' }

# possible values for FilterType:
# none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
# triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
# blackman-nuttall, flat-top, kaiser, parzen


# Create and run the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# Get the result
rec = astra.data2d.get(rec_id)

import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(P)

pylab.figure(2)
pylab.imshow(sinogram)

pylab.figure(3)
pylab.imshow(Ax)

pylab.figure(4)
pylab.imshow(rec)
pylab.show()

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

