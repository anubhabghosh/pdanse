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

def identity_hfn(x):
    return x

def splice32_hfn(x):
    return x[...,0:2] 

def splice31_hfn(x):
    return x[...,0:1]

def square_hfn(x):
    if type(x).__module__ == np.__name__:
        return np.square(x) #np.square(x+5) 
    elif type(x).__module__ == torch.__name__:
        return torch.square(x) #torch.square(x+5)
    
def scaled_square_hfn(x, d=1.0/20):
    if type(x).__module__ == np.__name__:
        return d * np.square(x) #np.square(x+5) 
    elif type(x).__module__ == torch.__name__:
        return d * torch.square(x) #torch.square(x+5)

def scaled_cubic_hfn(x, d=1.0/20):
    return d * x**3 

def sigmoid_hfn(x):
    if type(x).__module__ == np.__name__:
        return 1.0 / (1.0 + np.exp(-x))
    elif type(x).__module__ == torch.__name__:
        return 1.0 / (1.0 + torch.exp(-x))

def cubic_hfn(x):
    return (x)**3

def poly_hfn(x, ord=3):
    if type(x).__module__ == np.__name__:
        h_x = np.zeros_like(x)
    elif type(x).__module__ == torch.__name__:
        h_x = torch.zeros_like(x)
    
    for p in range(1,ord+1):
        h_x += x**p
    return h_x

def abs_hfn(x):
    if type(x).__module__ == np.__name__:
        return np.abs(x)
    elif type(x).__module__ == torch.__name__:
        return torch.abs(x)

def dist_sq_hfn(x):
    if type(x).__module__ == np.__name__:
        return np.sum(np.square(x)).reshape((-1,))
    elif type(x).__module__ == torch.__name__:
        return torch.sum(torch.square(x)).reshape((-1,))

def get_measurement_fn(fn_name):

    MEASUREMENT_FN_LIST = {  
        "identity": identity_hfn,  
        "square": square_hfn,
        "scaledsquare": square_hfn,
        "cubic": cubic_hfn,
        "poly": poly_hfn,
        "abs": abs_hfn,
        "distsq": dist_sq_hfn,
        "splice32": splice32_hfn,
        "splice31": splice31_hfn,
        "sigmoid":sigmoid_hfn
    }

    return MEASUREMENT_FN_LIST[fn_name.lower()]

        
