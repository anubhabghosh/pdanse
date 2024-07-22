#####################################################
# Creators: Anubhab Ghosh, Antoine Honoré
# Feb 2023
#####################################################
import copy
import math
import sys
from itertools import cycle
from timeit import default_timer as timer

import numpy as np
import torch

# from utils.plot_functions import plot_state_trajectory, plot_state_trajectory_axes
from torch import nn, optim
from torch.autograd import Variable

from src.rnn import RNN_model
from utils.utils import (
    ConvergenceMonitor,
    count_params,
    create_diag
)


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    return None


def push_model(nets, device="cpu"):
    nets = nets.to(device=device)
    return nets


class SemiDANSE(nn.Module):
    def __init__(
        self,
        n_states,
        n_obs,
        mu_w,
        C_w,
        H,
        mu_x0,
        C_x0,
        batch_size,
        rnn_type,
        rnn_params_dict,
        kappa=0.2,
        device="cpu",
    ):
        super(SemiDANSE, self).__init__()

        self.device = device

        # Initialize the paramters of the state estimator
        self.n_states = n_states
        self.n_obs = n_obs

        # Initializing the parameters of the initial state
        self.mu_x0 = self.push_to_device(mu_x0)
        self.C_x0 = self.push_to_device(C_x0)

        # Initializing the parameters of the measurement noise
        self.mu_w = self.push_to_device(mu_w)
        self.C_w = self.push_to_device(C_w)

        # Initialize the observation model matrix
        self.H = self.push_to_device(H)

        self.batch_size = batch_size

        # Initialize RNN type
        self.rnn_type = rnn_type

        # Initialize the parameters of the RNN
        self.rnn = RNN_model(**rnn_params_dict[self.rnn_type]).to(self.device)

        # Initialize various means and variances of the estimator

        # Prior parameters
        self.mu_xt_yt_current = None
        self.L_xt_yt_current = None

        # Marginal parameters
        self.mu_yt_current = None
        self.L_yt_current = None

        # Set the percentage of data to be used as supervision regularization
        self.kappa = kappa

        # Posterior parameters
        self.mu_xt_yt_prev = None
        self.L_xt_yt_prev = None

    def push_to_device(self, x):
        """Push the given tensor to the device"""
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def compute_prior_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev):
        self.mu_xt_yt_prev = mu_xt_yt_prev
        self.L_xt_yt_prev = create_diag(L_xt_yt_prev)
        return self.mu_xt_yt_prev, self.L_xt_yt_prev

    def compute_marginal_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev, Cw_batch):
        Cw_batch_seq = torch.repeat_interleave(
            Cw_batch.unsqueeze(1), L_xt_yt_prev.shape[1], dim=1
        )
        # print(self.H.device, self.mu_xt_yt_prev.device, self.mu_w.device)
        self.mu_yt_current = torch.einsum(
            "ij,ntj->nti", self.H, mu_xt_yt_prev
        ) + self.mu_w.squeeze(-1)
        self.L_yt_current = (
            self.H @ L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + Cw_batch_seq
        )  # + self.C_w

    def compute_posterior_mean_vars(self, Yi_batch, Cw_batch):
        Cw_batch_seq = torch.repeat_interleave(
            Cw_batch.unsqueeze(1), self.L_xt_yt_prev.shape[1], dim=1
        )
        # print((self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1)).shape, self.L_xt_yt_prev.shape, Cw_batch_seq.shape)
        Re_t_inv = torch.inverse(
            self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + Cw_batch_seq
        )  # torch.inverse(self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w)
        self.K_t = self.L_xt_yt_prev @ (self.H.T @ Re_t_inv)
        self.mu_xt_yt_current = self.mu_xt_yt_prev + torch.einsum(
            "ntij,ntj->nti",
            self.K_t,
            (Yi_batch - torch.einsum("ij,ntj->nti", self.H, self.mu_xt_yt_prev)),
        )
        # self.L_xt_yt_current = self.L_xt_yt_prev - (torch.einsum('ntij,ntjk->ntik',
        #                    self.K_t, self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1) + self.C_w) @ torch.transpose(self.K_t, 2, 3))
        self.L_xt_yt_current = self.L_xt_yt_prev - (
            torch.einsum(
                "ntij,ntjk->ntik",
                self.K_t,
                self.H @ self.L_xt_yt_prev @ torch.transpose(self.H, 0, 1)
                + Cw_batch_seq,
            )
            @ torch.transpose(self.K_t, 2, 3)
        )
        return self.mu_xt_yt_current, self.L_xt_yt_current

    def compute_logpdf_Gaussian_X(self, X):
        _, T, _ = X.shape
        logprob = (
            - 0.5 * self.n_states * T * math.log(math.pi * 2)
            - 0.5 * torch.logdet(self.L_xt_yt_current).sum(1)
            - 0.5
            * torch.einsum(
                "nti,nti->nt",
                (X - self.mu_xt_yt_current),
                torch.einsum(
                    "ntij,ntj->nti",
                    torch.inverse(self.L_xt_yt_current),
                    (X - self.mu_xt_yt_current),
                ),
            ).sum(1)
        )

        return logprob

    def compute_logpdf_Gaussian_Y(self, Y):
        _, T, _ = Y.shape
        logprob = (
            - 0.5 * self.n_obs * T * math.log(math.pi * 2)
            - 0.5 * torch.logdet(self.L_yt_current).sum(1)
            - 0.5
            * torch.einsum(
                "nti,nti->nt",
                (Y - self.mu_yt_current),
                torch.einsum(
                    "ntij,ntj->nti",
                    torch.inverse(self.L_yt_current),
                    (Y - self.mu_yt_current),
                ),
            ).sum(1)
        )

        return logprob

    def compute_predictions(self, Y_test_batch, Cw_test_batch):
        mu_x_given_Y_test_batch, vars_x_given_Y_test_batch = self.rnn.forward(
            x=Y_test_batch
        )
        mu_xt_yt_prev_test, L_xt_yt_prev_test = self.compute_prior_mean_vars(
            mu_xt_yt_prev=mu_x_given_Y_test_batch,
            L_xt_yt_prev=vars_x_given_Y_test_batch,
        )
        mu_xt_yt_current_test, L_xt_yt_current_test = self.compute_posterior_mean_vars(
            Yi_batch=Y_test_batch, Cw_batch=Cw_test_batch
        )
        return (
            mu_xt_yt_prev_test,
            L_xt_yt_prev_test,
            mu_xt_yt_current_test,
            L_xt_yt_current_test,
        )

    def forward(
        self,
        Yi_batch_unsup,
        Yi_batch_sup,
        Xi_batch_sup,
        Cw_batch_unsup,
        Cw_batch_sup,
        use_sup_loss=False,
        use_unsup_loss=True,
    ):
        # Compute the unsupervised loss
        if use_unsup_loss:
            mu_batch_unsup, vars_batch_unsup = self.rnn.forward(x=Yi_batch_unsup)
            mu_xt_yt_prev_unsup, L_xt_yt_prev_unsup = self.compute_prior_mean_vars(
                mu_xt_yt_prev=mu_batch_unsup, L_xt_yt_prev=vars_batch_unsup
            )
            self.compute_marginal_mean_vars(
                mu_xt_yt_prev=mu_xt_yt_prev_unsup,
                L_xt_yt_prev=L_xt_yt_prev_unsup,
                Cw_batch=Cw_batch_unsup,
            )
            # mu_xt_yt_current_test_unsup, L_xt_yt_current_test_unsup = self.compute_posterior_mean_vars(Yi_batch=Yi_batch_unsup)
            logprob_batch_unsup = self.compute_logpdf_Gaussian_Y(Y=Yi_batch_unsup)

            if use_sup_loss:
                # Compute the supervised loss
                mu_batch_sup, vars_batch_sup = self.rnn.forward(x=Yi_batch_sup)
                mu_xt_yt_prev_sup, L_xt_yt_prev_sup = self.compute_prior_mean_vars(
                    mu_xt_yt_prev=mu_batch_sup, L_xt_yt_prev=vars_batch_sup
                )
                self.compute_marginal_mean_vars(
                    mu_xt_yt_prev=mu_xt_yt_prev_sup,
                    L_xt_yt_prev=L_xt_yt_prev_sup,
                    Cw_batch=Cw_batch_sup,
                )
                mu_xt_yt_current_sup, L_xt_yt_current_sup = (
                    self.compute_posterior_mean_vars(
                        Yi_batch=Yi_batch_sup, Cw_batch=Cw_batch_sup
                    )
                )
                logprob_batch_sup = self.compute_logpdf_Gaussian_X(X=Xi_batch_sup)
                log_pXTYT_batch_avg = logprob_batch_unsup.mean(
                    0
                ) + logprob_batch_sup.mean(0)

            else:
                log_pXTYT_batch_avg = logprob_batch_unsup.mean(0)

        else:
            if use_sup_loss:
                # Compute the supervised loss
                mu_batch_sup, vars_batch_sup = self.rnn.forward(x=Yi_batch_sup)
                mu_xt_yt_prev_sup, L_xt_yt_prev_sup = self.compute_prior_mean_vars(
                    mu_xt_yt_prev=mu_batch_sup, L_xt_yt_prev=vars_batch_sup
                )
                self.compute_marginal_mean_vars(
                    mu_xt_yt_prev=mu_xt_yt_prev_sup,
                    L_xt_yt_prev=L_xt_yt_prev_sup,
                    Cw_batch=Cw_batch_sup,
                )
                mu_xt_yt_current_sup, L_xt_yt_current_sup = (
                    self.compute_posterior_mean_vars(
                        Yi_batch=Yi_batch_sup, Cw_batch=Cw_batch_sup
                    )
                )
                logprob_batch_sup = self.compute_logpdf_Gaussian_X(X=Xi_batch_sup)
                log_pXTYT_batch_avg = logprob_batch_sup.mean(0)

            else:
                log_pXTYT_batch_avg = 0.0

        # log_pXTYT_batch_avg = (logprob_batch_unsup.sum(0) + logprob_batch_sup.sum(0)) / Yi_batch_unsup.shape[0]

        return log_pXTYT_batch_avg


def train_danse_semisupervised(
    model,
    options,
    train_loader_unsup,
    val_loader_unsup,
    train_loader_sup,
    val_loader_sup,
    nepochs,
    logfile_path,
    modelfile_path,
    save_chkpoints,
    device="cpu",
    tr_verbose=False,
):
    # Push the model to device and count parameters
    model = push_model(nets=model, device=device)
    total_num_params, total_num_trainable_params = count_params(model)

    # Set the model to training
    model.train()
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model.rnn.lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=nepochs//3, gamma=0.9) # gamma was initially 0.9
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=nepochs // 6, gamma=0.9
    )  # gamma is now set to 0.8
    tr_losses = []
    val_losses = []

    if modelfile_path is None:
        model_filepath = "./models/"
    else:
        model_filepath = modelfile_path

    # if save_chkpoints == True:
    if save_chkpoints == "all" or save_chkpoints == "some":
        # No grid search
        if logfile_path is None:
            training_logfile = "./log/danse_semisupervised_{}.log".format(
                model.rnn_type
            )
        else:
            training_logfile = logfile_path

    elif save_chkpoints is None:
        # Grid search
        if logfile_path is None:
            training_logfile = "./log/gs_training_danse_semisupervised_{}.log".format(
                model.rnn_type
            )
        else:
            training_logfile = logfile_path

    # Call back parameters
    num_patience = 3
    min_delta = options["rnn_params_dict"][model.rnn_type][
        "min_delta"
    ]  # 1e-3 for simpler model, for complicated model we use 1e-2
    # min_tol = 1e-3 # for tougher model, we use 1e-2, easier models we use 1e-5
    best_val_loss = np.inf
    tr_loss_for_best_val_loss = np.inf
    best_model_wts = None
    best_val_epoch = None
    orig_stdout = sys.stdout
    f_tmp = open(training_logfile, "a")
    sys.stdout = f_tmp

    # Convergence monitoring (checks the convergence but not ES of the val_loss)
    model_monitor = ConvergenceMonitor(tol=min_delta, max_epochs=num_patience)

    # This checkes the ES of the val loss, if the loss deteriorates for specified no. of
    # max_epochs, stop the training
    # model_monitor = ConvergenceMonitor_ES(tol=min_tol, max_epochs=num_patience)

    print(
        "------------------------------ Training begins --------------------------------- \n"
    )
    print("Config: {} \n".format(options))
    print("\n Config: {} \n".format(options), file=orig_stdout)
    print(
        "No. of trainable parameters: {}\n".format(total_num_trainable_params),
        file=orig_stdout,
    )
    print("No. of trainable parameters: {}\n".format(total_num_trainable_params))

    # Start time
    starttime = timer()
    try:
        for epoch in range(nepochs):
            tr_running_loss = 0.0
            tr_loss_epoch_sum = 0.0
            val_loss_epoch_sum = 0.0
            val_mse_loss_epoch_sum = 0.0

            n_batches_sup_train = len(train_loader_sup)
            #n_batches_unsup_train = len(train_loader_unsup)
            use_sup_loss_train = True

            for i, (tr_data_sup, tr_data_unsup) in enumerate(
                zip(cycle(train_loader_sup), train_loader_unsup)
            ):
                if i + 1 > n_batches_sup_train:
                    use_sup_loss_train = False

                tr_Y_batch_sup, tr_X_batch_sup, tr_Cw_batch_sup = tr_data_sup
                tr_Y_batch_unsup, tr_X_batch_unsup, tr_Cw_batch_unsup = tr_data_unsup

                # print(i+1, n_batches_sup_train, n_batches_unsup_train, tr_Y_batch_unsup.shape, tr_Y_batch_sup.shape, file=orig_stdout)
                # print("Supervision flag (train) status: {}".format(use_sup_loss_train))
                # print("Supervision flag (train) status: {}".format(use_sup_loss_train), file=orig_stdout)

                optimizer.zero_grad()

                Y_train_batch_sup = (
                    Variable(tr_Y_batch_sup, requires_grad=False)
                    .type(torch.FloatTensor)
                    .to(device)
                )
                Cw_train_batch_sup = (
                    Variable(tr_Cw_batch_sup, requires_grad=False)
                    .type(torch.FloatTensor)
                    .to(device)
                )
                X_train_batch_sup = (
                    Variable(tr_X_batch_sup[:, :, :], requires_grad=False)
                    .type(torch.FloatTensor)
                    .to(device)
                )

                Y_train_batch_unsup = (
                    Variable(tr_Y_batch_unsup, requires_grad=False)
                    .type(torch.FloatTensor)
                    .to(device)
                )
                Cw_train_batch_unsup = (
                    Variable(tr_Cw_batch_unsup, requires_grad=False)
                    .type(torch.FloatTensor)
                    .to(device)
                )
                #X_train_batch_unsup = (
                #    Variable(tr_X_batch_unsup[:, :, :], requires_grad=False)
                #    .type(torch.FloatTensor)
                #    .to(device)
                #)

                if Y_train_batch_unsup.shape[0] > 0 and Y_train_batch_sup.shape[0] > 0:
                    log_pXY_train_batch_avg = -model.forward(
                        Y_train_batch_unsup,
                        Y_train_batch_sup,
                        X_train_batch_sup,
                        Cw_train_batch_unsup,
                        Cw_train_batch_sup,
                        use_sup_loss_train,
                    )
                elif (
                    Y_train_batch_unsup.shape[0] == 0 and Y_train_batch_sup.shape[0] > 0
                ):
                    log_pXY_train_batch_avg = -model.forward(
                        Y_train_batch_unsup,
                        Y_train_batch_sup,
                        X_train_batch_sup,
                        Cw_train_batch_unsup,
                        Cw_train_batch_sup,
                        use_sup_loss_train,
                        use_unsup_loss=False,
                    )
                elif (
                    Y_train_batch_unsup.shape[0] > 0 and Y_train_batch_sup.shape[0] == 0
                ):
                    log_pXY_train_batch_avg = -model.forward(
                        Y_train_batch_unsup,
                        Y_train_batch_sup,
                        X_train_batch_sup,
                        Cw_train_batch_unsup,
                        Cw_train_batch_sup,
                        use_sup_loss=False,
                    )

                log_pXY_train_batch_avg.backward()
                optimizer.step()

                # print statistics
                tr_running_loss += log_pXY_train_batch_avg.item()
                tr_loss_epoch_sum += log_pXY_train_batch_avg.item()

                if i % 100 == 99 and (
                    (epoch + 1) % 100 == 0
                ):  # print every 10 mini-batches
                    # print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 100))
                    # print("Epoch: {}/{}, Batch index: {}, Training loss: {}".format(epoch+1, nepochs, i+1, tr_running_loss / 100), file=orig_stdout)
                    tr_running_loss = 0.0

            scheduler.step()

            endtime = timer()
            # Measure wallclock time
            time_elapsed = endtime - starttime

            n_batches_sup_val = len(val_loader_sup)
            #n_batches_unsup_val = len(val_loader_unsup)
            use_sup_loss_val = True

            with torch.no_grad():
                for i, (val_data_sup, val_data_unsup) in enumerate(
                    zip(cycle(val_loader_sup), val_loader_unsup)
                ):
                    if i + 1 > n_batches_sup_val:
                        use_sup_loss_val = False

                    val_Y_batch_sup, val_X_batch_sup, val_Cw_batch_sup = val_data_sup
                    val_Y_batch_unsup, val_X_batch_unsup, val_Cw_batch_unsup = (
                        val_data_unsup
                    )

                    Y_val_batch_sup = (
                        Variable(val_Y_batch_sup, requires_grad=False)
                        .type(torch.FloatTensor)
                        .to(device)
                    )
                    X_val_batch_sup = (
                        Variable(val_X_batch_sup[:, :, :], requires_grad=False)
                        .type(torch.FloatTensor)
                        .to(device)
                    )
                    Cw_val_batch_sup = (
                        Variable(val_Cw_batch_sup, requires_grad=False)
                        .type(torch.FloatTensor)
                        .to(device)
                    )

                    Y_val_batch_unsup = (
                        Variable(val_Y_batch_unsup, requires_grad=False)
                        .type(torch.FloatTensor)
                        .to(device)
                    )
                    X_val_batch_unsup = (
                        Variable(val_X_batch_unsup[:, :, :], requires_grad=False)
                        .type(torch.FloatTensor)
                        .to(device)
                    )
                    Cw_val_batch_unsup = (
                        Variable(val_Cw_batch_unsup, requires_grad=False)
                        .type(torch.FloatTensor)
                        .to(device)
                    )

                    (
                        val_mu_X_predictions_batch,
                        val_var_X_predictions_batch,
                        val_mu_X_filtered_batch,
                        val_var_X_filtered_batch,
                    ) = model.compute_predictions(Y_val_batch_unsup, Cw_val_batch_unsup)

                    log_pXY_val_batch_avg = -model.forward(
                        Y_val_batch_unsup,
                        Y_val_batch_sup,
                        X_val_batch_sup,
                        Cw_val_batch_unsup,
                        Cw_val_batch_sup,
                        use_sup_loss_val,
                    )
                    val_loss_epoch_sum += log_pXY_val_batch_avg.item()

                    val_mse_loss_batch = mse_criterion(
                        X_val_batch_unsup.to(device), val_mu_X_filtered_batch
                    )
                    # print statistics
                    val_mse_loss_epoch_sum += val_mse_loss_batch.item()

            # Loss at the end of each epoch
            tr_loss = tr_loss_epoch_sum / (
                len(train_loader_unsup) + len(train_loader_sup)
            )
            val_loss = val_loss_epoch_sum / (
                len(val_loader_unsup) + len(val_loader_sup)
            )
            val_mse_loss = val_mse_loss_epoch_sum / (
                len(val_loader_unsup) + len(val_loader_sup)
            )

            # Record the validation loss per epoch
            if (
                epoch + 1
            ) > nepochs // 3:  # nepochs/6 for complicated, 100 for simpler model
                model_monitor.record(val_loss)

            # Displaying loss at an interval of 200 epochs
            if tr_verbose is True and (((epoch + 1) % 50) == 0 or epoch == 0):
                print(
                    "Epoch: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE:{:.9f}".format(
                        epoch + 1, model.rnn.num_epochs, tr_loss, val_loss, val_mse_loss
                    ),
                    file=orig_stdout,
                )
                # save_model(model, model_filepath + "/" + "{}_ckpt_epoch_{}.pt".format(model.model_type, epoch+1))

                print(
                    "Epoch: {}/{}, Training NLL:{:.9f}, Val. NLL:{:.9f}, Val. MSE: {:.9f}, Time_Elapsed:{:.4f} secs".format(
                        epoch + 1,
                        model.rnn.num_epochs,
                        tr_loss,
                        val_loss,
                        val_mse_loss,
                        time_elapsed,
                    )
                )

            # Checkpointing the model every few  epochs
            # if (((epoch + 1) % 500) == 0 or epoch == 0) and save_chkpoints == True:
            if (((epoch + 1) % 100) == 0 or epoch == 0) and save_chkpoints == "all":
                # Checkpointing model every few epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(
                    model,
                    model_filepath
                    + "/"
                    + "danse_semisupervised_{}_ckpt_epoch_{}.pt".format(
                        model.rnn_type, epoch + 1
                    ),
                )
            elif (((epoch + 1) % nepochs) == 0) and save_chkpoints == "some":
                # Checkpointing model at the end of training epochs, in case of grid_search is being done, save_chkpoints = None
                save_model(
                    model,
                    model_filepath
                    + "/"
                    + "danse_semisupervised_{}_ckpt_epoch_{}.pt".format(
                        model.rnn_type, epoch + 1
                    ),
                )

            # Save best model in case validation loss improves
            """
            best_val_loss, best_model_wts, best_val_epoch, patience, check_patience = callback_val_loss(model=model,
                                                                                                    best_model_wts=best_model_wts,
                                                                                                    val_loss=val_loss,
                                                                                                    best_val_loss=best_val_loss,
                                                                                                    best_val_epoch=best_val_epoch,
                                                                                                    current_epoch=epoch+1,
                                                                                                    patience=patience,
                                                                                                    num_patience=num_patience,
                                                                                                    min_delta=min_delta,
                                                                                                    check_patience=check_patience,
                                                                                                    orig_stdout=orig_stdout)
            if check_patience == True:
                print("Monitoring validation loss for criterion", file=orig_stdout)
                print("Monitoring validation loss for criterion")
            else:
                pass
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss # Save best validation loss
                tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
                best_val_epoch = epoch+1 # Corresponding value of epoch
                best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model
            """
            # Saving every value
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)

            # Check monitor flag
            if model_monitor.monitor(epoch=epoch + 1) is True:
                if tr_verbose is True:
                    print(
                        "Training convergence attained! Saving model at Epoch: {}".format(
                            epoch + 1
                        ),
                        file=orig_stdout,
                    )

                print("Training convergence attained at Epoch: {}!".format(epoch + 1))
                # Save the best model as per validation loss at the end
                best_val_loss = val_loss  # Save best validation loss
                tr_loss_for_best_val_loss = (
                    tr_loss  # Training loss corresponding to best validation loss
                )
                best_val_epoch = epoch + 1  # Corresponding value of epoch
                best_model_wts = copy.deepcopy(
                    model.state_dict()
                )  # Weights for the best model
                # print("\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(best_val_epoch, tr_loss_for_best_val_loss, best_val_loss))
                # save_model(model, model_filepath + "/" + "{}_usenorm_{}_ckpt_epoch_{}.pt".format(model.model_type, usenorm_flag, epoch+1))
                break

            # else:

            # print("Model improvement attained at Epoch: {}".format(epoch+1))
            # best_val_loss = val_loss # Save best validation loss
            # tr_loss_for_best_val_loss = tr_loss # Training loss corresponding to best validation loss
            # best_val_epoch = epoch+1 # Corresponding value of epoch
            # best_model_wts = copy.deepcopy(model.state_dict()) # Weights for the best model

        # Save the best model as per validation loss at the end
        print(
            "\nSaving the best model at epoch={}, with training loss={}, validation loss={}".format(
                best_val_epoch, tr_loss_for_best_val_loss, best_val_loss
            )
        )

        # if save_chkpoints == True:
        if save_chkpoints == "all" or save_chkpoints == "some":
            # Save the best model using the designated filename
            if best_model_wts is not None:
                model_filename = "danse_semisupervised_{}_ckpt_epoch_{}_best.pt".format(
                    model.rnn_type, best_val_epoch
                )
                torch.save(best_model_wts, model_filepath + "/" + model_filename)
            else:
                model_filename = "danse_semisupervised_{}_ckpt_epoch_{}_best.pt".format(
                    model.rnn_type, epoch + 1
                )
                print("Saving last model as best...")
                save_model(model, model_filepath + "/" + model_filename)
        # elif save_chkpoints == False:
        elif save_chkpoints is None:
            pass

    except KeyboardInterrupt:
        if tr_verbose is True:
            print(
                "Interrupted!! ...saving the model at epoch:{}".format(epoch + 1),
                file=orig_stdout,
            )
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch + 1))
        else:
            print("Interrupted!! ...saving the model at epoch:{}".format(epoch + 1))

        if save_chkpoints is not None:
            model_filename = "danse_semisupervised_{}_ckpt_epoch_{}_latest.pt".format(
                model.rnn_type, epoch + 1
            )
            torch.save(model, model_filepath + "/" + model_filename)

    print(
        "------------------------------ Training ends --------------------------------- \n"
    )
    # Restoring the original std out pointer
    sys.stdout = orig_stdout

    return tr_losses, val_losses, best_val_loss, tr_loss_for_best_val_loss, model
