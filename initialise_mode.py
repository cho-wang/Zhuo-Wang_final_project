import prox_nz

def initialise_mode(mode):
    # Make sure everything is in order
    if 'nest' not in mode:
        mode['nest'] = True
    if 'maxIter' not in mode:
        mode['maxIter'] = 100
    if 'bitRev' not in mode:
        mode['bitRev'] = True
    if 'offset' not in mode:
        mode['offset'] = False
    if 'verbose' not in mode:
        mode['verbose'] = 1
    if 'tau' not in mode:
        mode['tau'] = 1.99
    if 'nSplit' not in mode:
        mode['nSplit'] = 1
    if 'flip' not in mode:
        mode['flip'] = False
    if 'regFun' not in mode:
        mode['regFun'] = lambda z: 0
    if 'proxFun' not in mode:
        mode['proxFun'] = lambda z, t: prox_nz(z)
    if 'contrast' not in mode:
        mode['contrast'] = [0,2]
    if 'useConst' not in mode:
        mode['useConst'] = False
    if 'scatFun' not in mode:
        mode['scat'] = 0
        mode['scatFun'] = lambda z #?????
    elif not callable(mode['scatFun']):
        mode['scat'] = mode['scatFun']
        mode['scatFun'] = lambda z subSet, : mode['scat'][:,:,subSet] #?????
    else:
        mode['scat'] = 0
    return mode