
#####################################################
# Creator: Anubhab Ghosh 
# Mar 2024
#####################################################
import torch
import sys
import os
from timeit import default_timer as timer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.k_net import KalmanNetNN
from config.parameters_opt import get_parameters
from torch.autograd import Variable
from utils.utils import push_model

def test_kalmannet_ssm(Y_test, f_fn, ssm_model_test, model_file_saved_knet, device='cpu'):

    knet_model = KalmanNetNN(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        n_layers=1,
        device=device
    )

    def fn(x):
        return f_fn(x)
        
    def hn(x):
        #return x
        return ssm_model_test.h_fn(x)
    
    knet_model.Build(f=fn, h=hn)
    knet_model.ssModel = ssm_model_test

    start_time_knet = timer()
    
    knet_model.load_state_dict(torch.load(model_file_saved_knet, map_location=device))
    knet_model = push_model(nets=knet_model, device=device)
    knet_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_filtered_knet = knet_model.compute_predictions(Y_test_batch)
    
    X_estimated_filtered_knet = torch.transpose(X_estimated_filtered_knet, 1, 2)
    time_elapsed_knet = timer() - start_time_knet 

    return X_estimated_filtered_knet, time_elapsed_knet

    