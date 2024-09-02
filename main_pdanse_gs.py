#####################################################
# Creator: Anubhab Ghosh
# Feb 2023
#####################################################

# Import necessary libraries
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import argparse # noqa: E402
import copy # noqa: E402
import json # noqa: E402

# import matplotlib.pyplot as plt
import torch # noqa: E402
from parse import parse # noqa: E402

# Import the parameters
from config.parameters_opt import get_H_DANSE, get_parameters # noqa: E402

# from utils.plot_functions import plot_measurement_data, plot_measurement_data_axes, plot_state_trajectory, plot_state_trajectory_axes
# Import estimator model and functions
from src.pdanse import pDANSE, train_pdanse # noqa: E402
from utils.gs_utils import create_list_of_dicts # noqa: E402
from utils.utils import (  # noqa: E402
    NDArrayEncoder,
    check_if_dir_or_file_exists,
    create_dataloaders_from_dataset,
    load_saved_dataset,
    split_joint_dataset_S_US,
)


def main():
    parser = argparse.ArgumentParser(
        usage="Train DANSE using trajectories of SSMs \n"
        "python3.8 main_danse.py --mode [train/test] --model_type [gru/lstm/rnn] --dataset_mode [LinearSSM/LorenzSSM] \n"
        "--datafile [fullpath to datafile] --splits [fullpath to splits file]",
        description="Input a string indicating the mode of the script \n"
        "train - training and testing is done, test-only evlaution is carried out",
    )
    parser.add_argument(
        "--rnn_model_type", help="Enter the desired model (rnn/lstm/gru)", type=str
    )
    parser.add_argument(
        "--dataset_type", help="Enter the type of dataset (pfixed/vars/all)", type=str
    )
    parser.add_argument(
        "--n_sup",
        help="Enter the no. of samples of training data to be used for supervision",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model_file_saved",
        help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--datafile", help="Enter the full path to the dataset", type=str
    )
    parser.add_argument("--splits", help="Enter full path to splits file", type=str)

    args = parser.parse_args()
    model_type = args.rnn_model_type
    datafile = args.datafile
    dataset_type = args.dataset_type
    n_sup = args.n_sup
    #datafolder = "".join(
    #    datafile.split("/")[i] + "/" for i in range(len(datafile.split("/")) - 1)
    #)
    #model_file_saved = args.model_file_saved
    splits_file = args.splits

    print("datafile: {}".format(datafile))
    print(datafile.split("/")[-1])
    # Dataset parameters obtained from the 'datafile' variable
    (
        data_string,
        n_states,
        n_obs,
        _,
        measurement_fn_type,
        T,
        N_samples,
        sigma_e2_dB,
        smnr_dB,
    ) = parse(
        "{}_m_{:d}_n_{:d}_{}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_smnr_{:f}dB.pkl",
        datafile.split("/")[-1],
    )
    norm_indicator = data_string.split("_")[-1]

    kappa = n_sup / N_samples  # Calculate the value of kappa

    ngpu = 1  # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )
    print("Device Used:{}".format(device))

    ssm_parameters_dict, est_parameters_dict = get_parameters(
        n_states=n_states,
        n_obs=n_obs,
        device=device,
        measurment_fn_type=measurement_fn_type,
    )

    batch_size = est_parameters_dict["danse_semisupervised_plus"][
        "batch_size"
    ]  # Set the batch size
    estimator_options = est_parameters_dict[
        "danse_semisupervised_plus"
    ]  # Get the options for the estimator

    if not os.path.isfile(datafile):
        print(
            "Dataset is not present, run 'generate_data.py / run_generate_data.sh' to create the dataset"
        )
        # plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:
        print("Dataset already present!")
        Z_XY = load_saved_dataset(filename=datafile)

    # Split the full dataset consisting of N_samples into a supervised dataset consisting of n_sup no. of samples and an unsupervised one consisting of
    # (N_samples - n_sup) samples

    Z_XY_sup_dict, Z_XY_unsup_dict = split_joint_dataset_S_US(
        Z_XY, n_sup=n_sup, randomize=True
    )
    print(Z_XY_sup_dict["data"].shape, Z_XY_unsup_dict["data"].shape)

    ssm_model = Z_XY["ssm_model"]
    estimator_options["C_w"] = (
        ssm_model.Cw
    )  # Get the covariance matrix of the measurement noise from the model information
    estimator_options["H"] = get_H_DANSE(
        type_=dataset_type, n_states=n_states, n_obs=n_obs
    )  # Get the sensing matrix from the model info

    print(estimator_options["H"])

    train_loader_sup, val_loader_sup, test_loader_sup = create_dataloaders_from_dataset(
        datafile=datafile,
        Z_XY_dict=Z_XY_sup_dict,
        splits_file=splits_file,
        batch_size=batch_size,
        N=n_sup,
    )
    train_loader_unsup, val_loader_unsup, test_loader_unsup = (
        create_dataloaders_from_dataset(
            datafile=datafile,
            Z_XY_dict=Z_XY_unsup_dict,
            splits_file=splits_file,
            batch_size=batch_size,
            N=N_samples - n_sup,
        )
    )

    print(
        "No. of training, validation and testing batches (Sup.) : {}, {}, {}".format(
            len(train_loader_sup), len(val_loader_sup), len(test_loader_sup)
        )
    )
    print(
        "Training, validation and testing batch sizes (Sup.) : {}, {}, {}".format(
            train_loader_sup.batch_size,
            val_loader_sup.batch_size,
            test_loader_sup.batch_size,
        )
    )

    print(
        "No. of training, validation and testing batches (Unsup.) : {}, {}, {}".format(
            len(train_loader_unsup), len(val_loader_unsup), len(test_loader_unsup)
        )
    )

    print(
        "Training, validation and testing batch sizes (Unsup.) : {}, {}, {}".format(
            train_loader_unsup.batch_size,
            val_loader_unsup.batch_size,
            test_loader_unsup.batch_size,
        )
    )

    # ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    # print("Device Used:{}".format(device))

    logfile_path = "./log/"
    modelfile_path = "./models/"
    if norm_indicator.lower() == "normalized":
        dataset_type += "_" + norm_indicator.lower()

    # NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_{}_pdanse_opt_{}_nsup_{}_m_{}_n_{}_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB".format(
        dataset_type,
        measurement_fn_type,
        model_type,
        n_sup,  # estimator_options["kappa"],
        n_states,
        n_obs,
        T,
        N_samples,
        sigma_e2_dB,
        smnr_dB,
    )

    ngpu = 1  # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )
    print("Device Used:{}".format(device))

    # print(params)
    # Json file to store grid search results
    jsonfile_name = "gs_results_danse_{}_T_{}_N_{}.json".format(
        model_type, T, N_samples
    )
    gs_log_file_name = "gs_results_danse_{}_T_{}_N_{}.log".format(
        model_type, T, N_samples
    )

    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(
        os.path.join(logfile_path, main_exp_name), file_name=gs_log_file_name
    )

    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))

    # flag_models_dir, _ = check_if_dir_or_file_exists(os.path.join(modelfile_path, main_exp_name),
    #                                                file_name=None)

    # print("Is model-directory present:? - {}".format(flag_models_dir))
    # print("Is file present:? - {}".format(flag_file))

    tr_logfile_name_with_path = os.path.join(
        os.path.join(logfile_path, main_exp_name), gs_log_file_name
    )
    jsonfile_name_with_path = os.path.join(
        os.path.join(logfile_path, main_exp_name), jsonfile_name
    )

    if flag_log_dir is False:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)

    # Parameters to be tuned
    if model_type == "gru":
        gs_params = {
            "n_hidden": [40, 60, 80, 100],
            "n_layers": [2],
            "num_epochs": [5000, 7000],
            "lr":[5e-4, 1e-3],
            "min_delta":[2e-3],
            "n_hidden_dense": [32, 64],
        }
    elif model_type == "lstm":
        gs_params = {
            "n_hidden": [30, 40, 50, 60],
            "n_layers": [1, 2],
            "num_epochs": [2000],
            # "lr":[1e-2, 1e-3],
            # "min_delta":[5e-2, 1e-2],
            "n_hidden_dense": [32, 64],
        }

    # Creates the list of param combinations (options) based on the provided 'model_type'
    gs_list_of_options = create_list_of_dicts(
        options=estimator_options, model_type=model_type, param_dict=gs_params
    )

    print(
        "Grid Search to be carried over following {} configs:\n".format(
            len(gs_list_of_options)
        )
    )
    val_errors_list = []

    gs_stats = {}
    for i, gs_option in enumerate(gs_list_of_options):
        # Load the model with the corresponding options
        gs_option["kappa"] = kappa
        model_semidanse_plus = pDANSE(**gs_option)
        tr_verbose = True
        print("*"*100)
        print("Config number: {}".format(i+1))
        print("Chosen value of kappa: {}".format(model_semidanse_plus.kappa))

        tr_verbose = True
        save_chkpoints = None

        # Starting model training
        (
            tr_losses,
            val_losses,
            best_val_loss,
            tr_loss_for_best_val_loss,
            _,
        ) = train_pdanse(
            model=model_semidanse_plus,
            train_loader_unsup=train_loader_unsup,
            val_loader_unsup=val_loader_unsup,
            train_loader_sup=train_loader_sup,
            val_loader_sup=val_loader_sup,
            options=gs_option,
            nepochs=model_semidanse_plus.rnn.num_epochs,
            logfile_path=tr_logfile_name_with_path,
            modelfile_path=modelfile_path,
            save_chkpoints=save_chkpoints,
            device=device,
            tr_verbose=tr_verbose,
        )
        # if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)

        gs_stats["Config_no"] = i + 1
        gs_stats["tr_losses"] = tr_losses
        gs_stats["val_losses"] = val_losses
        gs_stats["tr_loss_end"] = tr_losses[-1]
        gs_stats["val_loss_end"] = val_losses[-1]
        gs_stats["tr_loss_best"] = tr_loss_for_best_val_loss
        gs_stats["val_loss_best"] = best_val_loss
        gs_stats["rnn_params_dict"] = copy.deepcopy(
            gs_option["rnn_params_dict"][model_type]
        )
        gs_stats["rnn_params_dict"]["device"] = "cuda"

        print(gs_stats)
        val_errors_list.append(copy.deepcopy(gs_stats))

    with open(jsonfile_name_with_path, "w") as f:
        print(val_errors_list)
        f.write(json.dumps(val_errors_list, indent=2, cls=NDArrayEncoder))

    return None


if __name__ == "__main__":
    main()
