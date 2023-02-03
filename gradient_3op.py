import numpy as np

def gradient_op(I, wx, wy, wz):
    nargin = 0
    if I != None:
        nargin = nargin+1
    if wx != None:
        nargin = nargin+1
    if wy != None:
        nargin = nargin+1

    dx = [I[2:-1,:,:]-I[1:-1 - 1,:,:]; np.zeros(1, np.size(I, 2), np.size(I, 3))]
    dy = [I[:,2:-1,:]-I[:,1:-1-1,:]; np.zeros(np.size(I, 1), 1, np.size(I, 3))]
    dz = cat(3, I(:,:, 2: -1,:)-I(:,:, 1: end - 1,:), ...zeros(size(I, 1), size(I, 2), 1, size(I, 4)));


    if nargin > 1:
        dx = dx * wx
        dy = dy * wy
        dz = dz * wz

return [dx,dy,dz]