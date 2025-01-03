#####################################################
# Creator: Anubhab Ghosh 
# Mar 2024
#####################################################
import numpy as np
import torch
import sys
import os
from timeit import default_timer as timer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.danse import DANSE
from src.semidanse import SemiDANSE
from src.danse_supervised import DANSE_Supervised
from src.pdanse import pDANSE
from config.parameters_opt import get_parameters
from torch.autograd import Variable
from utils.utils import push_model

def test_danse_ssm(Y_test, ssm_model_test, model_file_saved_danse, Cw_test=None, rnn_type='gru', device='cpu'):

    # Initialize the DANSE model parameters 
    ssm_dict, est_dict = get_parameters(n_states=ssm_model_test.n_states,
                                        n_obs=ssm_model_test.n_obs, 
                                        device=device)
    
    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        mu_w=ssm_model_test.mu_w,
        C_w=ssm_model_test.Cw,
        batch_size=1,
        H=ssm_model_test.H,
        mu_x0=np.zeros((ssm_model_test.n_states,)),
        C_x0=np.eye(ssm_model_test.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )
    
    print("DANSE Model file: {}".format(model_file_saved_danse))

    start_time_danse = timer()
    danse_model.load_state_dict(torch.load(model_file_saved_danse, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        Cw_test_batch = Variable(Cw_test, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch, 
                                                                                                                           Cw_test_batch)
    
    time_elapsed_danse = timer() - start_time_danse

    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered, time_elapsed_danse 

def test_danse_supervised_ssm(Y_test, ssm_model_test, h_fn_type, model_file_saved_danse_supervised, Cw_test=None, rnn_type='gru', device='cpu'):

    # Initialize the DANSE model parameters 
    ssm_dict, est_dict = get_parameters(n_states=ssm_model_test.n_states,
                                        n_obs=ssm_model_test.n_obs, 
                                        device=device)
    
    # Initialize the DANSE model in PyTorch
    danse_supervised_model = DANSE_Supervised(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        mu_w=ssm_model_test.mu_w,
        C_w=ssm_model_test.Cw,
        n_MC=est_dict['danse_supervised']['n_MC'],
        H=np.eye(ssm_model_test.n_obs, ssm_model_test.n_states),
        batch_size=1,
        h_fn_type=h_fn_type,
        mu_x0=np.zeros((ssm_model_test.n_states,)),
        C_x0=np.eye(ssm_model_test.n_states),
        rnn_type=rnn_type,
        kappa=est_dict['danse_supervised']['kappa'],
        rnn_params_dict=est_dict['danse_supervised']['rnn_params_dict'],
        device=device
    )
    
    print("DANSE Model file: {}".format(model_file_saved_danse_supervised))

    start_time_danse_supervised = timer()
    danse_supervised_model.load_state_dict(torch.load(model_file_saved_danse_supervised, map_location=device))
    danse_supervised_model = push_model(nets=danse_supervised_model, device=device)
    danse_supervised_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        Cw_test_batch = Variable(Cw_test, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred_supervised, Pk_estimated_pred_supervised, X_estimated_filtered_supervised, Pk_estimated_filtered_supervised = danse_supervised_model.compute_predictions(Y_test_batch, 
                                                                                                                           Cw_test_batch)
    
    time_elapsed_danse_supervised = timer() - start_time_danse_supervised

    return X_estimated_pred_supervised, Pk_estimated_pred_supervised, X_estimated_filtered_supervised, Pk_estimated_filtered_supervised, time_elapsed_danse_supervised

def test_danse_semisupervised_ssm(Y_test, ssm_model_test, model_file_saved_danse_semisupervised, Cw_test=None, rnn_type='gru', device='cpu'):

    # Initialize the DANSE model parameters 
    ssm_dict, est_dict = get_parameters(n_states=ssm_model_test.n_states,
                                        n_obs=ssm_model_test.n_obs, 
                                        device=device)
    
    # Initialize the DANSE model in PyTorch
    danse_semisupervised_model = SemiDANSE(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        mu_w=ssm_model_test.mu_w,
        C_w=ssm_model_test.Cw,
        batch_size=1,
        H=ssm_model_test.H,#jacobian(h_fn, torch.randn(lorenz_model.n_states,)).numpy(),
        mu_x0=np.zeros((ssm_model_test.n_states,)),
        C_x0=np.eye(ssm_model_test.n_states),
        rnn_type=rnn_type,
        kappa=est_dict['danse_semisupervised']['kappa'],
        rnn_params_dict=est_dict['danse_semisupervised']['rnn_params_dict'],
        device=device
    )
    
    print("pDANSE Model file: {}".format(model_file_saved_danse_semisupervised))

    start_time_danse_semisupervised = timer()
    danse_semisupervised_model.load_state_dict(torch.load(model_file_saved_danse_semisupervised, map_location=device))
    danse_semisupervised_model = push_model(nets=danse_semisupervised_model, device=device)
    danse_semisupervised_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        Cw_test_batch = Variable(Cw_test, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred_semisupervised, Pk_estimated_pred_semisupervised, X_estimated_filtered_semisupervised, Pk_estimated_filtered_semisupervised = danse_semisupervised_model.compute_predictions(Y_test_batch, 
                                                                                                                                                                                                Cw_test_batch)
    
    time_elapsed_danse_semisupervised = timer() - start_time_danse_semisupervised

    return X_estimated_pred_semisupervised, Pk_estimated_pred_semisupervised, X_estimated_filtered_semisupervised, Pk_estimated_filtered_semisupervised, time_elapsed_danse_semisupervised


def test_pdanse_ssm(Y_test, ssm_model_test, h_fn_type, model_file_saved_pdanse, Cw_test=None, rnn_type='gru', device='cpu'):

    # Initialize the DANSE model parameters 
    ssm_dict, est_dict = get_parameters(n_states=ssm_model_test.n_states,
                                        n_obs=ssm_model_test.n_obs, 
                                        device=device)
    
    # Initialize the DANSE model in PyTorch
    pdanse_model = pDANSE(
        n_states=ssm_model_test.n_states,
        n_obs=ssm_model_test.n_obs,
        mu_w=ssm_model_test.mu_w,
        C_w=ssm_model_test.Cw,
        n_MC=est_dict['pdanse']['n_MC'],
        H=np.eye(ssm_model_test.n_obs, ssm_model_test.n_states),
        batch_size=1,
        h_fn_type=h_fn_type,
        mu_x0=np.zeros((ssm_model_test.n_states,)),
        C_x0=np.eye(ssm_model_test.n_states),
        rnn_type=rnn_type,
        kappa=est_dict['pdanse']['kappa'],
        rnn_params_dict=est_dict['pdanse']['rnn_params_dict'],
        device=device
    )
    
    print("pDANSE plus Model file: {}".format(model_file_saved_pdanse))

    start_time_pdanse = timer()
    pdanse_model.load_state_dict(torch.load(model_file_saved_pdanse, map_location=device))
    pdanse_model = push_model(nets=pdanse_model, device=device)
    pdanse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        Cw_test_batch = Variable(Cw_test, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred_semisupervised, Pk_estimated_pred_semisupervised, X_estimated_filtered_semisupervised, Pk_estimated_filtered_semisupervised = pdanse_model.compute_predictions(Y_test_batch, 
                                                                                                                                                                                                Cw_test_batch)
    
    time_elapsed_semisupervised_plus = timer() - start_time_pdanse

    return X_estimated_pred_semisupervised, Pk_estimated_pred_semisupervised, X_estimated_filtered_semisupervised, Pk_estimated_filtered_semisupervised, time_elapsed_semisupervised_plus