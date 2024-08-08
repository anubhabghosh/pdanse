#####################################################
# Creator: Anubhab Ghosh 
# Mar 2024
#####################################################

import sys
import os
from timeit import default_timer as timer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.pf import PFModel

def test_pf_ssm(X_test, Y_test, ssm_model_test, f_fn, h_fn, n_particles=500, Cw_test=None, U_test=None, device='cpu'):

    # Initializing the extended Kalman filter model in PyTorch
    pf_model = PFModel(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        f=f_fn,#ssm_model_test.A_fn, f_lorenz for KalmanNet paper, f_lorenz_danse for our work
        h=h_fn,
        Q=ssm_model_test.Ce, #For KalmanNet
        R=ssm_model_test.Cw, # For KalmanNet
        n_particles=n_particles,
        device=device
    )

    # Get the estimates using an extended Kalman filter model
    
    X_estimated_pf = None
    Pk_estimated_pf = None

    start_time_pf = timer()
    X_estimated_pf, Pk_estimated_pf, mse_arr_pf = pf_model.run_mb_filter(X_test, Y_test, Cw=Cw_test, U=U_test)
    time_elapsed_pf = timer() - start_time_pf

    return X_estimated_pf, Pk_estimated_pf, mse_arr_pf, time_elapsed_pf