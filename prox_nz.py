def prox_nz(in_, up=None):

    if up is not None:
        in_[in_ > up] = up

    in_[in_ < 0] = 0

    return in_