#####################################################
# Creator: Anubhab Ghosh 
# Jul 2024
#####################################################
import numpy as np
from os import sys, path
# __file__ should be defined in this case
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def identity_hfn(x):
    return x

def square_hfn(x):
    return np.square(x)

def cubic_hfn(x):
    return x**3

def poly_hfn(x, ord=3):
    h_x = np.zeros_like(x)
    for p in range(ord):
        h_x += x**p
    return h_x

def abs_hfn(x):
    return np.abs(x)

def dist_sq_hfn(x):
    return np.sum(np.square(x))

def get_measurement_fn(fn_name):

    MEASUREMENT_FN_LIST = {
        "identity":identity_hfn,
        "square": square_hfn,
        "cubic": cubic_hfn,
        "poly": poly_hfn,
        "abs": abs_hfn,
        "dist_sq": dist_sq_hfn
    }

    return MEASUREMENT_FN_LIST[fn_name.lower()]

        