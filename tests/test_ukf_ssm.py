#####################################################
# Creator: Anubhab Ghosh 
# Mar 2024
#####################################################
import sys
import os
from timeit import default_timer as timer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.ukf_aliter import UKF_Aliter

def test_ukf_ssm(X_test, Y_test, ssm_model_test, f_ukf_fn, h_ukf_fn, U_test=None, Cw_test=None, device='cpu'):

    # Initializing the extended Kalman filter model in PyTorch
    ukf_model = UKF_Aliter(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        f=f_ukf_fn,
        h=h_ukf_fn,
        Q=ssm_model_test.Ce, # For KalmanNet, else None
        R=ssm_model_test.Cw, # For KalmanNet, else None,
        kappa=0, # Usually kept 0
        alpha=2.0, # Usually small 1e-3
        delta_t=ssm_model_test.delta,
        beta=2,
        device=device
    )

    # Get the estimates using an extended Kalman filter model
    
    X_estimated_ukf = None
    Pk_estimated_ukf = None

    start_time_ukf = timer()
    X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf_lin, mse_arr_ukf = ukf_model.run_mb_filter(X_test, Y_test, Cw=Cw_test, U=U_test)
    time_elapsed_ukf = timer() - start_time_ukf

    return X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf, time_elapsed_ukf