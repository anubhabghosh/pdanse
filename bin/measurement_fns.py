#####################################################
# Creator: Anubhab Ghosh 
# Jul 2024
#####################################################
import torch
import numpy as np
from os import sys, path
# __file__ should be defined in this case
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def square_hfn(x):
    if type(x).__module__ == np.__name__:
        return np.square(x)
    elif type(x).__module__ == torch.__name__:
        return torch.square(x)

def cubic_hfn(x):
    return x**3

def poly_hfn(x, ord=3):
    if type(x).__module__ == np.__name__:
        h_x = np.zeros_like(x)
    elif type(x).__module__ == torch.__name__:
        h_x = torch.zeros_like(x)
    
    for p in range(ord):
        h_x += x**p
    return h_x

def abs_hfn(x):
    if type(x).__module__ == np.__name__:
        return np.abs(x)
    elif type(x).__module__ == torch.__name__:
        return torch.abs(x)

def dist_sq_hfn(x):
    if type(x).__module__ == np.__name__:
        return np.sum(np.square(x))
    elif type(x).__module__ == torch.__name__:
        return torch.sum(torch.square(x))

def get_measurement_fn(fn_name):

    MEASUREMENT_FN_LIST = {    
        "square": square_hfn,
        "cubic": cubic_hfn,
        "poly": poly_hfn,
        "abs": abs_hfn,
        "dist_sq": dist_sq_hfn
    }

    return MEASUREMENT_FN_LIST[fn_name.lower()]

        