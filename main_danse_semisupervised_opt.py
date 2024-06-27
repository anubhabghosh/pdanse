#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
# Import necessary libraries
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import argparse
from parse import parse
import numpy as np
import json
from utils.utils import NDArrayEncoder
import scipy
#import matplotlib.pyplot as plt
import torch
import pickle as pkl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.utils import load_saved_dataset, Series_Dataset, \
    check_if_dir_or_file_exists, get_batch_size_sup, get_batch_size_unsup, \
    NDArrayEncoder, split_joint_dataset_S_US, create_dataloaders_from_dataset
# Import the parameters
from parameters_opt import get_parameters, get_H_DANSE
#from utils.plot_functions import plot_measurement_data, plot_measurement_data_axes, plot_state_trajectory, plot_state_trajectory_axes

# Import estimator model and functions
from src.danse_semisupervised import SemiDANSE, train_danse_semisupervised

def main():

    usage = "Train DANSE using trajectories of SSMs \n"\
        "python3.8 main_danse.py --mode [train/test] --model_type [gru/lstm/rnn] --dataset_mode [LinearSSM/LorenzSSM] \n"\
        "--datafile [fullpath to datafile] --splits [fullpath to splits file]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument("--rnn_model_type", help="Enter the desired model (rnn/lstm/gru)", type=str)
    parser.add_argument("--dataset_type", help="Enter the type of dataset (pfixed/vars/all)", type=str)
    parser.add_argument("--n_sup", help="Enter the no. of samples of training data to be used for supervision", type=int, default=5)
    parser.add_argument("--model_file_saved", help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str, default=None)
    parser.add_argument("--datafile", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--splits", help="Enter full path to splits file", type=str)
    
    args = parser.parse_args() 
    mode = args.mode
    model_type = args.rnn_model_type
    datafile = args.datafile
    dataset_type = args.dataset_type
    n_sup = args.n_sup
    datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))
    model_file_saved = args.model_file_saved
    splits_file = args.splits
    
    print("datafile: {}".format(datafile))
    print(datafile.split('/')[-1])
    # Dataset parameters obtained from the 'datafile' variable
    data_string, n_states, n_obs, _, T, N_samples, sigma_e2_dB, smnr_dB = parse("{}_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_smnr_{:f}dB.pkl", datafile.split('/')[-1])
    norm_indicator = data_string.split('_')[-1]
    
    kappa = n_sup / N_samples # Calculate the value of kappa

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))

    ssm_parameters_dict, est_parameters_dict = get_parameters(
                                            n_states=n_states,
                                            n_obs=n_obs,
                                            device=device
                                        )

    batch_size = est_parameters_dict["danse_semisupervised"]["batch_size"] # Set the batch size
    estimator_options = est_parameters_dict["danse_semisupervised"] # Get the options for the estimator

    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'generate_data.py / run_generate_data.sh' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present!")
        Z_XY = load_saved_dataset(filename=datafile)
    
    # Split the full dataset consisting of N_samples into a supervised dataset consisting of n_sup no. of samples and an unsupervised one consisting of
    # (N_samples - n_sup) samples 


    Z_XY_sup_dict, Z_XY_unsup_dict = split_joint_dataset_S_US(Z_XY, n_sup=n_sup, randomize=True)
    print(Z_XY_sup_dict['data'].shape, Z_XY_unsup_dict['data'].shape)

    ssm_model = Z_XY["ssm_model"]
    estimator_options['C_w'] = ssm_model.Cw # Get the covariance matrix of the measurement noise from the model information
    estimator_options['H'] = get_H_DANSE(type_=dataset_type, n_states=n_states, n_obs=n_obs) # Get the sensing matrix from the model info
    
    print(estimator_options['H'])

    train_loader_sup, val_loader_sup, test_loader_sup = create_dataloaders_from_dataset(datafile=datafile, Z_XY_dict=Z_XY_sup_dict, 
                                                                                        splits_file=splits_file, batch_size=batch_size, N=n_sup)
    train_loader_unsup, val_loader_unsup, test_loader_unsup = create_dataloaders_from_dataset(datafile=datafile, Z_XY_dict=Z_XY_unsup_dict, 
                                                                                        splits_file=splits_file, batch_size=batch_size, N=N_samples - n_sup)
    

    print("No. of training, validation and testing batches (Sup.) : {}, {}, {}".format(len(train_loader_sup), 
                                                                                len(val_loader_sup), 
                                                                                len(test_loader_sup)))
    print("Training, validation and testing batch sizes (Sup.) : {}, {}, {}".format(train_loader_sup.batch_size, 
                                                                            val_loader_sup.batch_size, 
                                                                            test_loader_sup.batch_size))

    
    print("No. of training, validation and testing batches (Unsup.) : {}, {}, {}".format(len(train_loader_unsup), 
                                                                                len(val_loader_unsup), 
                                                                                len(test_loader_unsup)))
    
    print("Training, validation and testing batch sizes (Unsup.) : {}, {}, {}".format(train_loader_unsup.batch_size, 
                                                                            val_loader_unsup.batch_size, 
                                                                            test_loader_unsup.batch_size))

    #ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    #print("Device Used:{}".format(device))
    
    logfile_path = "./log/"
    modelfile_path = "./models/"
    if norm_indicator.lower() == "normalized":
        dataset_type += "_" + norm_indicator.lower()
        
    #NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_danse_semisupervised_opt_{}_nsup_{}_m_{}_n_{}_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB".format(
                                                            dataset_type,
                                                            model_type,
                                                            n_sup, #estimator_options["kappa"],
                                                            n_states,
                                                            n_obs,
                                                            T,
                                                            N_samples,
                                                            sigma_e2_dB,
                                                            smnr_dB
                                                            )

    #print(params)
    tr_log_file_name = "training.log"
    te_log_file_name = "testing.log"

    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(os.path.join(logfile_path, main_exp_name),
                                                            file_name=tr_log_file_name)
    
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    
    flag_models_dir, _ = check_if_dir_or_file_exists(os.path.join(modelfile_path, main_exp_name),
                                                    file_name=None)
    
    print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))
    
    tr_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), tr_log_file_name)
    te_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), te_log_file_name)

    if flag_log_dir == False:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)
    
    if flag_models_dir == False:
        print("Creating {}".format(os.path.join(modelfile_path, main_exp_name)))
        os.makedirs(os.path.join(modelfile_path, main_exp_name), exist_ok=True)
    
    modelfile_path = os.path.join(modelfile_path, main_exp_name) # Modify the modelfile path to add full model file 

    if mode.lower() == "train":
        
        estimator_options['kappa'] = kappa
        model_danse = SemiDANSE(**estimator_options)
        tr_verbose = True  
        print("Chosen value of kappa: {}".format(model_danse.kappa))
        
        # Starting model training
        tr_losses, val_losses, _, _, _ = train_danse_semisupervised(
            model=model_danse,
            train_loader_unsup=train_loader_unsup,
            val_loader_unsup=val_loader_unsup,
            train_loader_sup=train_loader_sup,
            val_loader_sup=val_loader_sup,
            options=estimator_options,
            nepochs=model_danse.rnn.num_epochs,
            logfile_path=tr_logfile_name_with_path,
            modelfile_path=modelfile_path,
            save_chkpoints="some",
            device=device,
            tr_verbose=tr_verbose
        )
        #if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)
            
        losses_model = {}
        losses_model["tr_losses"] = tr_losses
        losses_model["val_losses"] = val_losses

        with open(os.path.join(os.path.join(logfile_path, main_exp_name), 
            'danse_semisupervised_{}_losses_eps{}.json'.format(estimator_options['rnn_type'], 
            estimator_options['rnn_params_dict'][model_type]['num_epochs'])), 'w') as f:
            f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

    return None

if __name__ == "__main__":
    main()
