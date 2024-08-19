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

def relu_hfn(x):
    if type(x).__module__ == np.__name__:
        return np.maximum(0,x)
    elif type(x).__module__ == torch.__name__:
        return torch.relu(x)

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

def cart2sph3dmod_hfn(x):
    if type(x).__module__ == np.__name__:
        hx = np.zeros_like(x)
        hx[0] = np.sqrt(np.sum(np.square(x)))
        hx[1] = np.arctan2(x[1], x[0]+1e-10) #np.sign(x[1]) * np.arccos(x[0] / np.sqrt(np.sum(np.square(x)[:2]))) 
        hx[2] = x[2] #np.arccos(x[2] / (np.sqrt(np.sum(np.square(x)))))
        assert not np.isnan(hx).any(), "NaNs in measurement function, x={}, hx={}".format(x, hx)

    elif type(x).__module__ == torch.__name__:
        hx = torch.zeros_like(x)
        hx[...,0] = torch.sqrt(torch.sum(torch.square(x), -1))
        hx[...,1] = torch.atan2(x[...,1], x[...,0]+1e-10) # torch.sign(x[...,1]) * torch.acos(torch.div(x[...,0], torch.sqrt(torch.sum(torch.square(x)[...,:2]))))
        hx[...,2] = x[...,2] #torch.acos(torch.div(x[...,2], torch.sqrt(torch.sum(torch.square(x), -1))))
        assert not torch.isnan(hx).any(), "NaNs in measurement function, x={}, hx={}".format(x, hx)

    return hx

def get_measurement_fn(fn_name):

    MEASUREMENT_FN_LIST = {  
        "identity": identity_hfn,  
        "square": square_hfn,
        "relu":relu_hfn,
        "scaledsquare": scaled_square_hfn,
        "scaledcubic": scaled_cubic_hfn,
        "cubic": cubic_hfn,
        "poly": poly_hfn,
        "abs": abs_hfn,
        "cart2sph3dmod": cart2sph3dmod_hfn,
        "splice32": splice32_hfn,
        "splice31": splice31_hfn,
        "sigmoid":sigmoid_hfn
    }

    return MEASUREMENT_FN_LIST[fn_name.lower()]

