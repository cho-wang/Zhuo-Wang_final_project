import numpy as np

def gradient_op(I, wx, wy):
    #Trying to translate nargin into python seems to work
    nargin = 0
    if I != None:
        nargin = nargin+1
    if wx != None:
        nargin = nargin+1
    if wy != None:
        nargin = nargin+1

    #Since the formula for gradient descent is not fully understood, the expressions of dx and dy need to be discussed
    dx = [I[1:-1,:,:]-I[0:-1-1,:,:]; np.zeros(0, np.size(I, 2), np.size(I, 3))]
    dy = [I[:,1:-1,:]-I[:,0:-1-1,:]; np.zeros(np.size(I, 1), 1, np.size(I, 3))]


    if (nargin > 1):
        dx = dx * wx
        dy = dy * wy

return (dx,dy)