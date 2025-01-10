#####################################################
# Creator: Anubhab Ghosh
# Feb 2023
#####################################################
import numpy as np
import glob
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from parse import parse
from timeit import default_timer as timer
import itertools
import json
import tikzplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from bin.measurement_fns import get_measurement_fn
from utils.plot_functions import (
    plot_3d_state_trajectory,
    plot_3d_measurement_trajectory,
    plot_state_trajectory_axes_all,
    plot_state_trajectory_w_lims,
    plot_meas_trajectory_w_lims,
)
from utils.utils import (
    dB_to_lin,
    nmse_loss,
    mse_loss_dB,
    load_saved_dataset,
    save_dataset,
    nmse_loss_std,
    mse_loss_dB_std,
    NDArrayEncoder,
)

# from parameters import get_parameters, A_fn, h_fn, f_lorenz_danse, f_lorenz_danse_ukf, delta_t, J_test
from config.parameters_opt import (
    get_parameters,
    f_lorenz,
    f_lorenz_ukf,
    f_chen,
    f_chen_ukf,
    f_rossler,
    f_rossler_ukf,
    f_nonlinear1d,
    f_nonlinear1d_ukf,
    cart2sph3dmod_ekf,
    J_TEST,
    DELTA_T_LORENZ63,
    DELTA_T_CHEN,
    DELTA_T_ROSSLER,
    DECIMATION_FACTOR_LORENZ63,
    DECIMATION_FACTOR_CHEN,
    DECIMATION_FACTOR_ROSSLER,
    get_H_DANSE,
)
from bin.generate_data import LorenzSSM, RosslerSSM, Nonlinear1DSSM
from test_ekf_ssm import test_ekf_ssm
from test_ukf_ssm import test_ukf_ssm
from test_pf_ssm import test_pf_ssm
from test_ukf_ssm_one_step import test_ukf_ssm_one_step
from test_danse_ssm import test_pdanse_ssm, test_danse_supervised_ssm
from test_kalmannet_ssm import test_kalmannet_ssm
from test_dmm_causal_ssm import test_dmm_causal_ssm
from get_one_step_ahead_lin_meas import get_y_pred_linear_meas


def metric_to_func_map(metric_name):
    """Function to map from a metric name to the respective function.
    The metric 'time_elapsed' couldn't be included here, as it is calculated inline,
    and hence needs to be manually assigned.

    Args:
        metric_name (str): Metric name

    Returns:
        fn : function call that helps to calculate the given metric
    """
    if metric_name == "nmse":
        fn = nmse_loss
    elif metric_name == "nmse_std":
        fn = nmse_loss_std
    elif metric_name == "mse_dB":
        fn = mse_loss_dB
    elif metric_name == "mse_dB_std":
        fn = mse_loss_dB_std
    return fn


def get_f_function(ssm_type):
    if "LorenzSSM" in ssm_type:
        f_fn = f_lorenz
        f_ukf_fn = f_lorenz_ukf
    elif "ChenSSM" in ssm_type:
        f_fn = f_chen
        f_ukf_fn = f_chen_ukf
    elif "RosslerSSM" in ssm_type:
        f_fn = f_rossler
        f_ukf_fn = f_rossler_ukf
    elif "Nonlinear1DSSM" in ssm_type:
        f_fn = f_nonlinear1d
        f_ukf_fn = f_nonlinear1d_ukf
    return f_fn, f_ukf_fn


def test_on_ssm_model(
    device="cpu",
    learnable_model_files=None,
    test_data_file=None,
    test_logfile=None,
    evaluation_mode="full",
    metrics_list=None,
    models_list=None,
    dirname=None,
    figdirname=None,
    n_particles_pf=100
):
    model_file_saved_danse = (
        learnable_model_files["danse"] if "danse" in models_list else None
    )
    model_file_saved_danse_supervised = (
        learnable_model_files["danse_supervised"]
        if "danse_supervised" in models_list
        else None
    )
    model_file_saved_pdanse = (
        learnable_model_files["pdanse"]
        if "pdanse" in models_list
        else None
    )
    model_file_saved_knet = (
        learnable_model_files["kalmannet"] if "kalmannet" in models_list else None
    )
    model_file_saved_dmm = (
        learnable_model_files["dmm_st-l"] if "dmm_st-l" in models_list else None
    )

    ssm_type, h_fn_type, rnn_type, nsup, m, n, T, _, sigma_e2_dB, smnr_dB = parse(
        "{}_{}_pdanse_opt_{}_nsup_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_sigmae2_{:f}dB_smnr_{:f}dB",
        model_file_saved_pdanse.split("/")[-2],
    )

    J = 5
    J_test = J_TEST
    n_particles = n_particles_pf
    # smnr_dB = 20
    decimate = True
    use_Taylor = False

    orig_stdout = sys.stdout
    f_tmp = open(test_logfile, "a")
    sys.stdout = f_tmp

    if not os.path.isfile(test_data_file):
        print("Dataset is not present, creating at {}".format(test_data_file))
        print(
            "Dataset is not present, creating at {}".format(test_data_file),
            file=orig_stdout,
        )
        # My own data generation scheme
        (
            m,
            n,
            ssm_type_test,
            h_fn_type_test,
            T_test,
            N_test,
            sigma_e2_dB_test,
            smnr_dB_test,
        ) = parse(
            "test_trajectories_m_{:d}_n_{:d}_{}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl",
            test_data_file.split("/")[-1],
        )
        # N_test = 100 # No. of trajectories at test time / evaluation
        X = torch.zeros((N_test, T_test, m))
        Y = torch.zeros((N_test, T_test, n))
        Cw = torch.zeros((N_test, n, n))

        ssm_dict, _ = get_parameters(
            n_states=m, n_obs=n, measurment_fn_type=h_fn_type_test, device="cpu"
        )
        ssm_model_test_dict = ssm_dict[ssm_type_test]
        f_fn, f_ukf_fn = get_f_function(ssm_type_test)

        if "LorenzSSM" in ssm_type_test:
            delta = DELTA_T_LORENZ63  # If decimate is True, then set this delta to 1e-5 and run it for long time
            delta_d = DELTA_T_LORENZ63 / DECIMATION_FACTOR_LORENZ63
            ssm_model_test = LorenzSSM(
                n_states=m,
                n_obs=n,
                J=J,
                delta=delta,
                delta_d=delta_d,
                decimate=decimate,
                measurement_fn_type=h_fn_type_test,
                alpha=0.0,
                H=get_H_DANSE(type_=ssm_type_test, n_states=m, n_obs=n),
                mu_e=np.zeros((m,)),
                mu_w=np.zeros((n,)),
                use_Taylor=use_Taylor,
            )
            U_test = torch.empty((N_test, m)).type(torch.FloatTensor)
            decimation_factor = DECIMATION_FACTOR_LORENZ63

        elif "ChenSSM" in ssm_type_test:
            delta = DELTA_T_CHEN  # If decimate is True, then set this delta to 1e-5 and run it for long time
            delta_d = DELTA_T_CHEN / DECIMATION_FACTOR_CHEN
            ssm_model_test = LorenzSSM(
                n_states=m,
                n_obs=n,
                J=J,
                delta=delta,
                delta_d=delta_d,
                decimate=decimate,
                measurement_fn_type=h_fn_type_test,
                alpha=1.0,
                H=get_H_DANSE(type_=ssm_type_test, n_states=m, n_obs=n),
                mu_e=np.zeros((m,)),
                mu_w=np.zeros((n,)),
                use_Taylor=use_Taylor,
            )
            U_test = torch.empty((N_test, m)).type(torch.FloatTensor)
            decimation_factor = DECIMATION_FACTOR_CHEN

        elif "RosslerSSM" in ssm_type_test:
            delta = DELTA_T_ROSSLER  # If decimate is True, then set this delta to 1e-5 and run it for long time
            delta_d = DELTA_T_ROSSLER / DECIMATION_FACTOR_ROSSLER
            ssm_model_test = RosslerSSM(
                n_states=m,
                n_obs=n,
                J=J_test,
                delta=delta,
                delta_d=delta_d,
                a=ssm_model_test_dict["a"],
                b=ssm_model_test_dict["b"],
                H=get_H_DANSE(type_=ssm_type_test, n_states=m, n_obs=n),
                c=ssm_model_test_dict["c"],
                measurement_fn_type=h_fn_type_test,
                decimate=decimate,
                mu_e=np.zeros((m,)),
                mu_w=np.zeros((n,)),
                use_Taylor=use_Taylor,
            )
            U_test = torch.empty((N_test, m)).type(torch.FloatTensor)
            decimation_factor = DECIMATION_FACTOR_ROSSLER

        elif "Nonlinear1DSSM" in ssm_type_test:
            ssm_model_test = Nonlinear1DSSM(
                n_states=m,
                n_obs=n,
                a=ssm_model_test_dict["a"],
                b=ssm_model_test_dict["b"],
                c=ssm_model_test_dict["c"],
                d=ssm_model_test_dict["d"],
                measurement_fn_type=h_fn_type_test,
                mu_e=ssm_model_test_dict["mu_e"],
                mu_w=ssm_model_test_dict["mu_w"],
            )
            ssm_model_test.delta = 1.0
            decimation_factor = 1
            u_test = torch.Tensor(
                [ssm_model_test.generate_driving_noise(k) for k in range(T_test)]
            ).type(torch.FloatTensor)
            U_test = u_test.unsqueeze(-1).unsqueeze(0).repeat(N_test, 1, 1)

        print(
            "Test data generated using sigma_e2: {} dB, SMNR: {} dB".format(
                sigma_e2_dB_test, smnr_dB_test
            )
        )

        idx_test = 0
        while idx_test < N_test:
            x_ssm_i, y_ssm_i, cw_ssm_i = ssm_model_test.generate_single_sequence(
                T=int(T_test * decimation_factor),
                sigma_e2_dB=sigma_e2_dB_test,
                smnr_dB=smnr_dB_test,
            )
            if not np.isnan(x_ssm_i).any():
                X[idx_test, :, :] = torch.from_numpy(x_ssm_i).type(torch.FloatTensor)
                Y[idx_test, :, :] = torch.from_numpy(y_ssm_i).type(torch.FloatTensor)
                Cw[idx_test, :, :] = torch.from_numpy(cw_ssm_i).type(torch.FloatTensor)
                idx_test += 1

        test_data_dict = {}
        test_data_dict["X"] = X
        test_data_dict["Y"] = Y
        test_data_dict["Cw"] = Cw
        test_data_dict["U"] = U_test
        test_data_dict["model"] = ssm_model_test
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:
        print("Dataset at {} already present!".format(test_data_file))
        m, n, ssm_type_test, h_fn_type_test, T_test, N_test, sigma_e2_dB_test, smnr_dB_test = parse(
            "test_trajectories_m_{:d}_n_{:d}_{}_{}_data_T_{:d}_N_{:d}_sigmae2_{:f}dB_SMNR_{:f}dB.pkl",
            test_data_file.split("/")[-1],
        )
        test_data_dict = load_saved_dataset(filename=test_data_file)
        X = test_data_dict["X"]
        Y = test_data_dict["Y"]
        Cw = test_data_dict["Cw"]
        U_test = test_data_dict["U"]
        ssm_model_test = test_data_dict["model"]
        f_fn, f_ukf_fn = get_f_function(ssm_type_test)

    assert (
        h_fn_type == h_fn_type_test
    ), "Loaded model and test data are not corresponding to the same type of measurement fn. in the dataset"

    print("*" * 100)
    print("*" * 100, file=orig_stdout)
    # i_test = np.random.choice(N_test)
    print("sigma_e2: {}dB, smnr: {}dB".format(sigma_e2_dB_test, smnr_dB_test))
    print(
        "sigma_e2: {}dB, smnr: {}dB".format(sigma_e2_dB_test, smnr_dB_test),
        file=orig_stdout,
    )

    ###########################################################################################################
    # NOTE: The following couple lines of code are only for rapid testing for debugging of code,
    # We take the first two samples from the testing dataset and pass them on to the prediction techniques
    #idx = np.random.choice(X.shape[0], 2, replace=False)
    #print(idx, file=orig_stdout)
    #Y = Y[idx]
    #X = X[idx]
    #Cw = Cw[idx]
    #if "Nonlinear1DSSM" in ssm_type_test:
    #    U_test = U_test[idx]
    ###########################################################################################################

    # Collecting all estimator results
    X_estimated_dict = dict.fromkeys(models_list, {})

    #####################################################################################################################################################################
    # Estimator baseline: Least Squares
    N_test, Ty, dy = Y.shape
    N_test, Tx, dx = X.shape

    # Get the estimate using the least-squares (LS) baseline!
    H_tensor = torch.from_numpy(
        jacobian(
            ssm_model_test.h_fn,
            torch.randn(
                ssm_model_test.n_states,
            ),
        ).numpy()
    ).type(torch.FloatTensor)
    H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0), N_test, dim=0)
    # X_LS = torch.einsum('ijj,ikj->ikj',torch.pinverse(H_tensor),Y)
    start_time_ls = timer()
    X_LS = torch.zeros_like(X)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            X_LS[i, j, :] = (
                torch.pinverse(H_tensor[i]) @ Y[i, j, :].reshape((dy, 1))
            ).reshape((dx,))
    time_elapsed_ls = timer() - start_time_ls
    X_estimated_dict["true"] = X
    X_estimated_dict["meas"] = Y
    X_estimated_dict["ls"]["est"] = X_LS

    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Model-based filters
    print("Fed to Model-based filters: ", file=orig_stdout)
    print(
        "sigma_e2: {}dB, smnr: {}dB, delta_t: {}".format(
            sigma_e2_dB_test, smnr_dB_test, DELTA_T_LORENZ63
        ),
        file=orig_stdout,
    )

    print("Fed to Model-based filters: ")
    print(
        "sigma_e2: {}dB, smnr: {}dB, delta_t: {}".format(
            sigma_e2_dB_test, smnr_dB_test, DELTA_T_LORENZ63
        )
    )

    ssm_model_test.sigma_e2 = dB_to_lin(sigma_e2_dB_test)
    ssm_model_test.setStateCov(sigma_e2=dB_to_lin(sigma_e2_dB_test))

    # Estimator: EKF
    if "ekf" in models_list:
        print("Testing EKF ...", file=orig_stdout)
        X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf, time_elapsed_ekf = test_ekf_ssm(
            X_test=X,
            Y_test=Y,
            ssm_model_test=ssm_model_test,
            f_fn=f_fn,
            h_fn=cart2sph3dmod_ekf if h_fn_type_test == "cart2sph3dmod" else get_measurement_fn(fn_name=h_fn_type_test),
            Cw_test=Cw,
            U_test=U_test if "Nonlinear1DSSM" in ssm_type_test else None,
            device=device,
            use_Taylor=use_Taylor,
        )
        X_estimated_dict["ekf"]["est"] = X_estimated_ekf
        X_estimated_dict["ekf"]["est_cov"] = Pk_estimated_ekf
    else:
        X_estimated_ekf, Pk_estimated_ekf, _, time_elapsed_ekf = None, None, None, None

    # Estimator: UKF
    if "ukf" in models_list:
        print("Testing UKF ...", file=orig_stdout)
        X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf, time_elapsed_ukf = test_ukf_ssm(
            X_test=X,
            Y_test=Y,
            ssm_model_test=ssm_model_test,
            f_ukf_fn=f_ukf_fn,
            h_ukf_fn=get_measurement_fn(fn_name=h_fn_type_test),
            U_test=U_test if "Nonlinear1DSSM" in ssm_type_test else None,
            Cw_test=Cw,
            device=device,
        )
        X_estimated_dict["ukf"]["est"] = X_estimated_ukf
        X_estimated_dict["ukf"]["est_cov"] = Pk_estimated_ukf

        """
        X_estimated_ukf_pred, Pk_estimated_ukf_pred, _, _, _, _, _, _, _ = (
            test_ukf_ssm_one_step(
                X_test=X,
                Y_test=Y,
                ssm_model_test=ssm_model_test,
                f_ukf_fn=f_ukf_fn,
                Cw_test=Cw,
                device=device,
            )
        )
    
        Y_estimated_ukf_pred, Py_estimated_ukf_pred = get_y_pred_linear_meas(
            X_estimated_pred_test=X_estimated_ukf_pred,
            Pk_estimated_pred_test=Pk_estimated_ukf_pred,
            Cw_test=Cw,
            ssm_model_test=ssm_model_test,
        )
        """
        Y_estimated_ukf_pred, Py_estimated_ukf_pred = None, None
    else:
        X_estimated_ukf, Pk_estimated_ukf, _, time_elapsed_ukf = None, None, None, None
        Y_estimated_ukf_pred, Py_estimated_ukf_pred = None, None

    # Estimator: PF
    if "pf" in models_list:
        print("Testing PF ...", file=orig_stdout)
        X_estimated_pf, Pk_estimated_pf, mse_arr_pf, time_elapsed_pf = test_pf_ssm(
            X_test=X,
            Y_test=Y,
            ssm_model_test=ssm_model_test,
            f_fn=f_fn,
            n_particles=n_particles,
            h_fn=get_measurement_fn(fn_name=h_fn_type_test),
            Cw_test=Cw,
            device=device,
            U_test=U_test if "Nonlinear1DSSM" in ssm_type else None,
        )
        X_estimated_dict["pf"]["est"] = X_estimated_pf
        X_estimated_dict["pf"]["est_cov"] = Pk_estimated_pf
    else:
        X_estimated_pf, Pk_estimated_pf, _, time_elapsed_pf = None, None, None, None

    
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Estimator: DANSE Semisupervised Plus
    if "pdanse" in models_list:
        print("Testing pDANSE ...", file=orig_stdout)
        (
            X_estimated_pred_pdanse,
            Pk_estimated_pred_pdanse,
            X_estimated_filtered_pdanse,
            Pk_estimated_filtered_pdanse,
            time_elapsed_danse_pdanse,
        ) = test_pdanse_ssm(
            Y_test=Y,
            ssm_model_test=ssm_model_test,
            h_fn_type=h_fn_type_test,
            model_file_saved_pdanse=model_file_saved_pdanse,
            Cw_test=Cw,
            rnn_type=rnn_type,
            device=device,
        )
        X_estimated_dict["pdanse"]["est"] = (
            X_estimated_filtered_pdanse
        )
        X_estimated_dict["pdanse"]["est_cov"] = (
            Pk_estimated_filtered_pdanse
        )
        """
        (
            Y_estimated_pred_pdanse,
            Py_estimated_pred_pdanse,
        ) = get_y_pred_linear_meas(
            X_estimated_pred_test=X_estimated_pred_pdanse,
            Pk_estimated_pred_test=Pk_estimated_pred_pdanse,
            Cw_test=Cw,
            ssm_model_test=ssm_model_test,
        )
        """
        (
            Y_estimated_pred_pdanse,
            Py_estimated_pred_pdanse,
        ) = None, None
    else:
        (
            X_estimated_pred_pdanse,
            Pk_estimated_pred_pdanse,
            X_estimated_filtered_pdanse,
            Pk_estimated_filtered_pdanse,
            time_elapsed_danse_pdanse,
        ) = None, None, None, None, None
        (
            Y_estimated_pred_pdanse,
            Py_estimated_pred_pdanse,
        ) = None, None
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Estimator: DANSE Semisupervised Plus
    if "danse_supervised" in models_list:
        print("Testing DANSE (Supervised +) ...", file=orig_stdout)
        (
            X_estimated_pred_danse_supervised,
            Pk_estimated_pred_danse_supervised,
            X_estimated_filtered_danse_supervised,
            Pk_estimated_filtered_danse_supervised,
            time_elapsed_danse_danse_supervised,
        ) = test_danse_supervised_ssm(
            Y_test=Y,
            ssm_model_test=ssm_model_test,
            h_fn_type=h_fn_type_test,
            model_file_saved_danse_supervised=model_file_saved_danse_supervised,
            Cw_test=Cw,
            rnn_type=rnn_type,
            device=device,
        )
        X_estimated_dict["danse_supervised"]["est"] = (
            X_estimated_filtered_danse_supervised
        )
        X_estimated_dict["danse_supervised"]["est_cov"] = (
            Pk_estimated_filtered_danse_supervised
        )

    #####################################################################################################################################################################
    # Estimator: KalmanNet (unsupervised)
    if "kalmannet" in models_list:
        print("Testing KalmanNet ...", file=orig_stdout)
        # Initialize the KalmanNet model in PyTorch
        X_estimated_filtered_knet, time_elapsed_knet = test_kalmannet_ssm(
            Y_test=Y,
            f_fn=f_fn,
            ssm_model_test=ssm_model_test,
            model_file_saved_knet=model_file_saved_knet,
            device=device,
        )
        X_estimated_dict["kalmannet"]["est"] = X_estimated_filtered_knet
    else:
        X_estimated_filtered_knet, time_elapsed_knet = None, None
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Estimator: DMM - ST-L
    if "dmm_st-l" in models_list:
        print("Testing DMM (ST-L) ...", file=orig_stdout)
        # Initialize the KalmanNet model in PyTorch
        (
            X_estimated_filtered_dmm_causal,
            Pk_estimated_filtered_dmm_causal,
            time_elapsed_dmm_causal,
        ) = test_dmm_causal_ssm(
            Y_test=Y,
            ssm_model_test=ssm_model_test,
            Cw_test=Cw,
            model_file_saved_dmm=model_file_saved_dmm,
            device=device,
        )
        X_estimated_dict["dmm_st-l"]["est"] = X_estimated_filtered_dmm_causal
        X_estimated_dict["dmm_st-l"]["est_cov"] = Pk_estimated_filtered_dmm_causal
    else:
        (
            X_estimated_filtered_dmm_causal,
            Pk_estimated_filtered_dmm_causal,
            time_elapsed_dmm_causal,
        ) = None, None, None
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    
    # Saving the estimator results
    torch.save([X, Y, X_estimated_filtered_danse_supervised, X_estimated_filtered_pdanse], os.path.join(os.path.join(figdirname, dirname, evaluation_mode), 'test_results_nsuP_{}_smnr_{}.pt'.format(nsup, smnr_dB_test))); 

    # Metrics calculation
    metrics_dict_for_one_smnr = dict.fromkeys(metrics_list, {})

    for metric_name in metrics_list:
        metric_fn = (
            metric_to_func_map(metric_name=metric_name)
            if metric_name != "time_elapsed"
            else None
        )
        if metric_fn is not None:
            metrics_dict_for_one_smnr[metric_name] = {
                "ls": metric_fn(X, X_LS).numpy().item()
                if "ls" in models_list
                else None,
                "ekf": metric_fn(X, X_estimated_ekf).numpy().item()
                if "ekf" in models_list
                else None,
                "ukf": metric_fn(X, X_estimated_ukf).numpy().item()
                if "ukf" in models_list
                else None,
                "pf": metric_fn(X, X_estimated_pf).numpy().item()
                if "pf" in models_list
                else None,
                "pdanse": metric_fn(
                    X, X_estimated_filtered_pdanse
                )
                .numpy()
                .item()
                if "pdanse" in models_list
                else None,
                "danse_supervised": metric_fn(
                    X, X_estimated_filtered_danse_supervised
                )
                .numpy()
                .item()
                if "danse_supervised" in models_list
                else None,
                "dmm_st-l": metric_fn(X, X_estimated_filtered_dmm_causal).numpy().item()
                if "dmm_st-l" in models_list
                else None,
                "kalmannet": metric_fn(X, X_estimated_filtered_knet).numpy().item()
                if "kalmannet" in models_list
                else None,
            }

    metrics_dict_for_one_smnr["time_elapsed"] = {
        "ls": time_elapsed_ls if "ls" in models_list else None,
        "ekf": time_elapsed_ekf if "ekf" in models_list else None,
        "ukf": time_elapsed_ukf if "ukf" in models_list else None,
        "pf": time_elapsed_pf if "pf" in models_list else None,
        "pdanse": time_elapsed_danse_pdanse
        if "pdanse" in models_list
        else None,
        "danse_supervised": time_elapsed_danse_danse_supervised
        if "danse_supervised" in models_list
        else None,
        "dmm_st-l": time_elapsed_dmm_causal if "dmm_st-l" in models_list else None,
        "kalmannet": time_elapsed_knet if "kalmannet" in models_list else None,
    }

    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Displaying metrics (logging and on-console print)
    for model_name in models_list:
        # Logs metrics
        print(
            "{}, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(
                model_name,
                N_test,
                metrics_dict_for_one_smnr["nmse"][model_name],
                metrics_dict_for_one_smnr["nmse_std"][model_name],
                metrics_dict_for_one_smnr["mse_dB"][model_name],
                metrics_dict_for_one_smnr["mse_dB_std"][model_name],
                metrics_dict_for_one_smnr["time_elapsed"][model_name],
            )
        )

        # On-console printing of metrics
        print(
            "{}, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(
                model_name,
                N_test,
                metrics_dict_for_one_smnr["nmse"][model_name],
                metrics_dict_for_one_smnr["nmse_std"][model_name],
                metrics_dict_for_one_smnr["mse_dB"][model_name],
                metrics_dict_for_one_smnr["mse_dB_std"][model_name],
                metrics_dict_for_one_smnr["time_elapsed"][model_name],
            ),
            file=orig_stdout,
        )
    #####################################################################################################################################################################

    #####################################################################################################################################################################
    # Plot the result
    figid_  = np.random.randint(0, X.shape[0])
    if m >= 2:
        plot_3d_state_trajectory(
            X=torch.squeeze(X[figid_, :, :], 0).numpy(),
            legend="$\\mathbf{x}^{true}$",
            m="b-",
            savefig_name="./{}/{}/{}/{}_x_true_sigmae2_{}dB_smnr_{}dB.pdf".format(
                figdirname, dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test
            ),
            savefig=True,
        )

        plot_3d_measurement_trajectory(
            Y=torch.squeeze(Y[figid_, :, :], 0).numpy(),
            legend="$\\mathbf{y}^{true}$",
            m="r-",
            savefig_name="./{}/{}/{}/{}_y_true_sigmae2_{}dB_smnr_{}dB.pdf".format(
                figdirname, dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test
            ),
            savefig=True,
        )

        if "ukf" in models_list:
            plot_3d_state_trajectory(
                X=torch.squeeze(X_estimated_ukf[figid_], 0).numpy(),
                legend="$\\hat{\mathbf{x}}_{UKF}$",
                m="k-",
                savefig_name="./{}/{}/{}/{}_x_ukf_sigmae2_{}dB_smnr_{}dB.pdf".format(
                    figdirname, dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test
                ),
                savefig=True,
            )

        if "pf" in models_list:
            plot_3d_state_trajectory(
                X=torch.squeeze(X_estimated_pf[figid_], 0).numpy(),
                legend="$\\hat{\mathbf{x}}_{PF}$",
                m="k-",
                savefig_name="./{}/{}/{}/{}_x_pf_sigmae2_{}dB_smnr_{}dB.pdf".format(
                    figdirname, dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test
                ),
                savefig=True,
            )

        if "pdanse" in models_list:
            plot_3d_state_trajectory(
                X=torch.squeeze(
                    X_estimated_filtered_pdanse[figid_], 0
                ).numpy(),
                legend="$\\hat{\mathbf{x}}_{pDANSE}$",
                m="k-",
                savefig_name="./{}/{}/{}/{}_x_pdanse_sigmae2_{}dB_smnr_{}dB.pdf".format(
                    figdirname, dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test
                ),
                savefig=True,
            )

        if "danse_supervised" in models_list:
            plot_3d_state_trajectory(
                X=torch.squeeze(
                    X_estimated_filtered_danse_supervised[figid_], 0
                ).numpy(),
                legend="$\\hat{\mathbf{x}}_{DANSE-Sup}$",
                m="k-",
                savefig_name="./{}/{}/{}/{}_x_danse-sup_plus_sigmae2_{}dB_smnr_{}dB.pdf".format(
                    figdirname, dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test
                ),
                savefig=True,
            )

        if "dmm_st-l" in models_list:
            plot_3d_state_trajectory(
                X=torch.squeeze(X_estimated_filtered_dmm_causal[figid_], 0).numpy(),
                legend="$\\hat{\mathbf{x}}_{DMM}$",
                m="k-",
                savefig_name="./{}/{}/{}/{}_x_dmm_causal_sigmae2_{}dB_smnr_{}dB.pdf".format(
                    figdirname, dirname, evaluation_mode, ssm_type_test, sigma_e2_dB_test, smnr_dB_test
                ),
                savefig=True,
            )

    
    plot_state_trajectory_axes_all(
        X=torch.squeeze(X[figid_, :, :], 0).numpy(),
        X_est_EKF=torch.squeeze(X_estimated_ekf[figid_, :, :], 0).numpy()
        if "ekf" in models_list
        else None,
        X_est_UKF=torch.squeeze(X_estimated_ukf[figid_, :, :], 0).numpy()
        if "ukf" in models_list
        else None,
        X_est_PF=torch.squeeze(X_estimated_pf[figid_, :, :], 0).numpy()
        if "pf" in models_list
        else None,
        X_est_pDANSE=torch.squeeze(
            X_estimated_filtered_pdanse[figid_], 0
        ).numpy()
        if "pdanse" in models_list
        else None,
        X_est_DANSE_Supervised=torch.squeeze(
            X_estimated_filtered_danse_supervised[figid_], 0
        ).numpy()
        if "danse_supervised" in models_list
        else None,
        X_est_DMM=torch.squeeze(X_estimated_filtered_dmm_causal[figid_], 0).numpy()
        if "dmm_st-l" in models_list
        else None,
        X_est_KNET=torch.squeeze(X_estimated_filtered_knet[figid_], 0).numpy()
        if "kalmannet" in models_list
        else None,
        savefig=True,
        savefig_name="./{}/{}/{}/AxesAll_sigmae2_{}dB_smnr_{}dB.pdf".format(
            figdirname, dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test
        ),
    )

    plot_state_trajectory_w_lims(
        X=torch.squeeze(X[figid_, :, :], 0).numpy(),
        X_est_UKF=torch.squeeze(X_estimated_ukf[figid_, :, :], 0).numpy()
        if "ukf" in models_list
        else None,
        X_est_UKF_std=np.sqrt(
            torch.diagonal(
                torch.squeeze(Pk_estimated_ukf[figid_, :, :, :], 0), offset=0, dim1=1, dim2=2
            ).numpy()
        )
        if "ukf" in models_list
        else None,
        X_est_PF=torch.squeeze(X_estimated_pf[figid_, :, :], 0).numpy()
        if "pf" in models_list
        else None,
        X_est_PF_std=np.sqrt(
            torch.diagonal(
                torch.squeeze(Pk_estimated_pf[figid_, :, :, :], 0), offset=0, dim1=1, dim2=2
            ).numpy()
        )
        if "pf" in models_list
        else None,
        X_est_pDANSE=torch.squeeze(
            X_estimated_filtered_pdanse[figid_], 0
        ).numpy()
        if "pdanse" in models_list
        else None,
        X_est_pDANSE_std=np.sqrt(
            torch.diagonal(
                torch.squeeze(
                    Pk_estimated_filtered_pdanse[figid_, :, :, :], 0
                ),
                offset=0,
                dim1=1,
                dim2=2,
            ).numpy()
        )
        if "pdanse" in models_list
        else None,
        X_est_DANSE_sup=torch.squeeze(
            X_estimated_filtered_danse_supervised[figid_], 0
        ).numpy()
        if "danse_supervised" in models_list
        else None,
        X_est_DANSE_sup_std=np.sqrt(
            torch.diagonal(
                torch.squeeze(
                    Pk_estimated_filtered_danse_supervised[figid_, :, :, :], 0
                ),
                offset=0,
                dim1=1,
                dim2=2,
            ).numpy()
        )
        if "danse_supervised" in models_list
        else None,
        savefig=True,
        savefig_name="./{}/{}/{}/Axes_w_lims_sigmae2_{}dB_smnr_{}dB.pdf".format(
            figdirname, dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test
        ),
    )
    
    """
    plot_meas_trajectory_w_lims(
        Y=torch.squeeze(Y[0, :, :], 0).numpy(),
        Y_pred_UKF=torch.squeeze(Y_estimated_ukf_pred[0, :, :], 0).numpy(),
        Y_pred_UKF_std=np.sqrt(
            torch.diagonal(
                torch.squeeze(Py_estimated_ukf_pred[0, :, :, :], 0),
                offset=0,
                dim1=1,
                dim2=2,
            ).numpy()
        ),
        Y_pred_DANSE=torch.squeeze(Y_estimated_pred[0], 0).numpy(),
        Y_pred_DANSE_std=np.sqrt(
            torch.diagonal(
                torch.squeeze(Py_estimated_pred[0], 0), offset=0, dim1=1, dim2=2
            ).numpy()
        ),
        Y_pred_pDANSE=torch.squeeze(
            Y_estimated_pred_pdanse[0], 0
        ).numpy(),
        Y_pred_pDANSE_std=np.sqrt(
            torch.diagonal(
                torch.squeeze(Py_estimated_pred_pdanse[0], 0),
                offset=0,
                dim1=1,
                dim2=2,
            ).numpy()
        ),
        savefig=True,
        savefig_name="./figs/{}/{}/Meas_y_lims_sigmae2_{}dB_smnr_{}dB.pdf".format(
            dirname, evaluation_mode, sigma_e2_dB_test, smnr_dB_test
        ),
    )
    """
    # plt.show()
    sys.stdout = orig_stdout
    return metrics_dict_for_one_smnr


if __name__ == "__main__":
    # Testing parameters
    ssm_name = "Lorenz"
    h_fn_type = "relu"
    m = 3
    n = 3
    n_particles = 100
    T_train = 200
    N_train = 1000
    T_test = 2000
    N_test = 100
    sigma_e2_dB_test = -10.0
    device = "cpu"
    nsup = 20
    nsup_supervised = 20
    bias = None  # By default should be positive, equal to 10.0
    p = None  # Keep this fixed at zero for now, equal to 0.0
    mode = "full"
    if mode == "low" or mode == "high":
        evaluation_mode = "partial_opt_{}_bias_{}_p_{}".format(mode, bias, p)
    else:
        bias = None
        p = None
        evaluation_mode = (
            "ModTest_diff_smnr_nsup_{}_Ntrain_{}_Ttrain_{}_Ntest_{}_T_test_{}_refactored_nparticles_{}".format(
                nsup, N_train, T_train, N_test, T_test, n_particles
            )
        )
    ssmtype = (
        "{}SSMn{}_{}".format(ssm_name, n, h_fn_type)
        if n < m
        else "{}SSM_{}".format(ssm_name, h_fn_type)
    )  # Hardcoded for {}SSMrn{} (random H low rank), {}SSMn{} (deterministic H low rank),
    dirname = "{}SSMn{}x{}_{}".format(ssm_name, m, n, h_fn_type)
    figdirname="figs_final_with_relu_sup"
    os.makedirs("./{}/{}/{}".format(figdirname, dirname, evaluation_mode), exist_ok=True)

    smnr_dB_arr = np.array([0.0, 10.0, 20.0, 30.0])
    smnr_dB_dict_arr = ["{}dB".format(smnr_dB) for smnr_dB in smnr_dB_arr]

    list_of_models_comparison = ["ls", "danse_supervised", "pdanse"]
    list_of_display_fmts = ["gp-.", "rd--", "ks--", "bo-", "mo-", "ys-", "co-"]
    list_of_metrics = ["nmse", "nmse_std", "mse_dB", "mse_dB_std", "time_elapsed"]

    metrics_dict = dict.fromkeys(list_of_metrics, {})

    for metric in list_of_metrics:
        for model_name in list_of_models_comparison:
            metrics_dict[metric][model_name] = np.zeros((len(smnr_dB_arr)))

    metrics_multidim_mat = np.zeros(
        (len(list_of_metrics), len(list_of_models_comparison), len(smnr_dB_arr))
    )

    model_file_saved_dict = {
        # "danse": dict.fromkeys(smnr_dB_dict_arr, {}),
        # "kalmannet":dict.fromkeys(smnr_dB_dict_arr, {}),
        # "dmm_st-l":dict.fromkeys(smnr_dB_dict_arr, {}),
        #"danse_supervised":dict.fromkeys(smnr_dB_dict_arr, {}),
        "pdanse": dict.fromkeys(smnr_dB_dict_arr, {}),
    }

    test_data_file_dict = {}

    for j, smnr_dB_label in enumerate(smnr_dB_dict_arr):
        test_data_file_dict[smnr_dB_label] = (
            "./data/synthetic_data/test_trajectories_m_{}_n_{}_{}_data_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB.pkl".format(
                m, n, ssmtype, T_test, N_test, sigma_e2_dB_test, smnr_dB_arr[j]
            )
        )

    for key in model_file_saved_dict.keys():
        for j, smnr_dB_label in enumerate(smnr_dB_dict_arr):
            if key == "kalmannet":
                saved_model = "KNetUoffline"
            else:
                saved_model = key

            N_train_actual = N_train
            #N_train_actual = (
            #    nsup
            #    if nsup is not None and saved_model in ["danse_supervised"]
            #    else N_train
            #)

            if saved_model == "pdanse" and nsup is not None:
                model_file_saved_dict[key][smnr_dB_label] = glob.glob(
                    "./models/*{}_{}_*nsup_{}_m_{}_n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                        ssmtype,
                        "pdanse",
                        nsup,
                        m,
                        n,
                        T_train,
                        N_train_actual,
                        sigma_e2_dB_test,
                        smnr_dB_arr[j],
                    )
                )[-1]

            elif saved_model == "KNetUoffline":
                model_file_saved_dict[key][smnr_dB_label] = glob.glob(
                    "./models/*{}_{}_*m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                        ssmtype,
                        saved_model,
                        m,
                        n,
                        T_train,
                        N_train_actual,
                        sigma_e2_dB_test,
                        smnr_dB_arr[j],
                    )
                )[-1]
            elif saved_model == "danse_supervised" and nsup is not None:
                model_file_saved_dict[key][smnr_dB_label] = glob.glob(
                    "./models/*{}_{}_*nsup_{}_m_{}_n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                        ssmtype,
                        "danse_supervised",
                        nsup_supervised,
                        m,
                        n,
                        T_train,
                        N_train_actual,
                        sigma_e2_dB_test,
                        smnr_dB_arr[j],
                    )
                )[-1]
            else:
                model_file_saved_dict[key][smnr_dB_label] = glob.glob(
                    "./models/*{}_{}_*opt_gru_m_{}_*n_{}_*T_{}_N_{}*sigmae2_{}dB_smnr_{}dB*/*best*".format(
                        ssmtype,
                        saved_model,
                        m,
                        n,
                        T_train,
                        N_train_actual,
                        sigma_e2_dB_test,
                        smnr_dB_arr[j],
                    )
                )[-1]

    print(model_file_saved_dict["danse_supervised"])
    print(model_file_saved_dict["pdanse"])

    test_logfile = "./{}/{}_test_{}_T_{}_N_{}_ModTest.log".format(
        figdirname, ssmtype, evaluation_mode, T_test, N_test
    )
    test_jsonfile = "./{}/{}_test_{}_T_{}_N_{}_ModTest.json".format(
        figdirname, ssmtype, evaluation_mode, T_test, N_test
    )

    for idx_smnr, smnr_dB_label in enumerate(smnr_dB_dict_arr):
        test_data_file_i = test_data_file_dict[smnr_dB_label]
        learnable_model_files_i = {
            key: model_file_saved_dict[key][smnr_dB_label]
            for key in model_file_saved_dict.keys()
        }

        metrics_dict_i = test_on_ssm_model(
            device=device,
            learnable_model_files=learnable_model_files_i,
            test_data_file=test_data_file_i,
            test_logfile=test_logfile,
            evaluation_mode=evaluation_mode,
            metrics_list=list_of_metrics,
            models_list=list_of_models_comparison,
            dirname=dirname,
            figdirname=figdirname,
            n_particles_pf=n_particles
        )

        for idx_metric, idx_model_name in itertools.product(
            list(range(len(list_of_metrics))),
            list(range(len(list_of_models_comparison))),
        ):
            metrics_multidim_mat[idx_metric, idx_model_name, idx_smnr] = metrics_dict_i[
                list_of_metrics[idx_metric]
            ][list_of_models_comparison[idx_model_name]]
    
    metrics_dict = {}
    metrics_dict["result_mat"] = metrics_multidim_mat
    with open(test_jsonfile, "w") as f:
        f.write(json.dumps(metrics_dict, cls=NDArrayEncoder, indent=2))

    plt.rcParams["font.family"] = "serif"
    display_metrics = [
        idx_metric
        for idx_metric in range(len(list_of_metrics))
        if "_std" not in list_of_metrics[idx_metric]
    ]

    for idx_display_metric in range(len(display_metrics)):
        if list_of_metrics[idx_display_metric] != "time_elapsed":
            plt.figure()
            for j, model_name in enumerate(list_of_models_comparison):
                plt.errorbar(
                    smnr_dB_arr,
                    metrics_multidim_mat[idx_display_metric, j, :],
                    fmt=list_of_display_fmts[j],
                    yerr=metrics_multidim_mat[idx_display_metric + 1, j, :],
                    linewidth=1.5,
                    label="{}".format(model_name.upper()),
                )
            plt.xlabel("SMNR (in dB)")
            plt.ylabel("{} (in dB)".format(list_of_metrics[idx_display_metric].upper()))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.subplot(212)
            tikzplotlib.save(
                "./{}/{}/{}/{}_vs_SMNR_{}.tex".format(
                    figdirname,
                    dirname,
                    evaluation_mode,
                    list_of_metrics[idx_display_metric].upper(),
                    ssm_name,
                )
            )
            plt.savefig(
                "./{}/{}/{}/{}_vs_SMNR_{}.pdf".format(
                    figdirname,
                    dirname,
                    evaluation_mode,
                    list_of_metrics[idx_display_metric].upper(),
                    ssm_name,
                )
            )
        else:
            plt.figure()
            for j, model_name in enumerate(list_of_models_comparison):
                plt.plot(
                    smnr_dB_arr,
                    metrics_multidim_mat[idx_display_metric, j, :],
                    list_of_display_fmts[j],
                    linewidth=1.5,
                    label="{}".format(model_name.upper()),
                )
            plt.xlabel("SMNR (in dB)")
            plt.ylabel(
                "{} (in secs)".format(list_of_metrics[idx_display_metric].upper())
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.subplot(212)
            tikzplotlib.save(
                "./{}/{}/{}/{}_vs_SMNR_{}.tex".format(
                    figdirname,
                    dirname,
                    evaluation_mode,
                    list_of_metrics[idx_display_metric].upper(),
                    ssm_name,
                )
            )
            plt.savefig(
                "./{}/{}/{}/{}_vs_SMNR_{}.pdf".format(
                    figdirname,
                    dirname,
                    evaluation_mode,
                    list_of_metrics[idx_display_metric].upper(),
                    ssm_name,
                )
            )