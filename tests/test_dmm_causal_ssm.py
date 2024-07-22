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
from src.dmm_causal import DMM
from config.parameters_opt import get_parameters
from torch.autograd import Variable
from utils.utils import push_model,create_diag

def test_dmm_causal_ssm(Y_test, ssm_model_test, model_file_saved_dmm, Cw_test=None, rnn_type='gru', device='cpu'):

    # Initialize the DANSE model parameters 
    ssm_dict, est_dict = get_parameters(n_states=ssm_model_test.n_states,
                                        n_obs=ssm_model_test.n_obs, 
                                        device=device)
    
    # Initialize the DANSE model in PyTorch
    dmm_causal_model = DMM(
        obs_dim=ssm_model_test.n_obs,
        latent_dim=ssm_model_test.n_states,
        rnn_model_type=est_dict['dmm']['rnn_model_type'],
        rnn_params_dict=est_dict['dmm']['rnn_params_dict'],
        optimizer_params=est_dict['dmm']['optimizer_params'],
        use_mean_field_q=est_dict['dmm']['use_mean_field_q'],
        inference_mode=est_dict['dmm']['inference_mode'],
        combiner_dim=est_dict['dmm']['combiner_dim'],
        train_emission=est_dict['dmm']['train_emission'],
        H=ssm_model_test.H,
        C_w=ssm_model_test.Cw,
        emission_dim=est_dict['dmm']['emission_dim'],
        emission_use_binary_obs=est_dict['dmm']['emission_use_binary_obs'],
        emission_num_layers=est_dict['dmm']['emission_num_layers'],
        train_transition=est_dict['dmm']['train_transition'],
        transition_dim=est_dict['dmm']['transition_dim'],
        transition_num_layers=est_dict['dmm']['transition_num_layers'],
        train_initials=est_dict['dmm']['train_initials'],
        device=device
    )
    
    print("DMM Model file: {}".format(model_file_saved_dmm))

    start_time_dmm = timer()
    dmm_causal_model.load_state_dict(torch.load(model_file_saved_dmm, map_location=device))
    dmm_causal_model = push_model(nets=dmm_causal_model, device=device)
    dmm_causal_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        _, mu_X_latent_q_seq, var_X_latent_q_seq, _, _ = dmm_causal_model.inference(Y_test_batch)
        X_estimated_filtered = mu_X_latent_q_seq
        Pk_estimated_filtered = create_diag(var_X_latent_q_seq)
    
    time_elapsed_dmm = timer() - start_time_dmm

    return X_estimated_filtered, Pk_estimated_filtered, time_elapsed_dmm 