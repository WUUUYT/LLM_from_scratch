from math import cos, pi


def learning_rate_schedule(t, lr_max, lr_min, Tw, Tc):
    if t < Tw:
        return t / Tw * lr_max
    elif t <= Tc:
        return lr_min + 0.5 * (1 + cos((t - Tw) / (Tc - Tw) * pi)) * (lr_max - lr_min)
    else:
        return lr_min
