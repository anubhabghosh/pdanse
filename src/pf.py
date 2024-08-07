#################################################################################
# Creator: Anubhab Ghosh, Apr 2024
# Adapted from: https://github.com/KalmanNet/KalmanNet_TSP/tree/main
# Implementing a particle filter with the help of the pyparticleest package
#################################################################################
import pyparticleest.models.nlg
import pyparticleest.simulator as simulator
import torch
from timeit import default_timer as timer
from utils.utils import mse_loss
import numpy as np


class PFModel(pyparticleest.models.nlg.NonlinearGaussianInitialGaussian):
    def __init__(
        self,
        n_states,
        n_obs,
        f=None,
        h=None,
        Q=None,
        R=None,
        n_particles=100,
        delta_t=1.0,
        device="cpu",
    ):
        # Initialize the device
        self.device = device

        # Initializing the system model
        self.n_states = n_states  # Setting the number of states of the particle filter
        self.n_obs = n_obs
        self.f = f  # State transition function (relates x_k, u_k to x_{k+1})
        self.h = h  # Output function (relates state x_k to output y_k)

        self.n_particles = n_particles
        self.Q_k = self.push_to_device(
            Q
        )  # Covariance matrix of the process noise, we assume process noise w_k ~ N(0, Q)
        self.R_k = self.push_to_device(
            R
        )  # Covariance matrix of the measurement noise, we assume mesaurement noise v_k ~ N(0, R)

        self.initialize()
        self.g = lambda x: torch.squeeze(self.h(x))

    def calc_f(self, particles, u, t):
        N_p = particles.shape[0]
        particles_f = np.empty((N_p, self.n))
        for k in range(N_p):
            particles_f[k, :] = self.f(torch.tensor(particles[k, :]))
        return particles_f

    def calc_g(self, particles, t):
        N_p = particles.shape[0]
        particles_g = np.empty((N_p, self.m))
        for k in range(N_p):
            particles_g[k, :] = self.g(torch.tensor(particles[k, :]))
        return particles_g

    def push_to_device(self, x):
        """Push the given tensor to the device"""
        return torch.from_numpy(x).type(torch.FloatTensor).to(self.device)

    def initialize(self, R_k=None):
        self.x0 = torch.ones((self.n_states,)).numpy()
        self.P0 = (torch.eye(self.n_states) * 1e-5).numpy()
        if R_k is not None:
            self.R_k = self.push_to_device(R_k)
        super(PFModel, self).__init__(
            x0=self.x0, Px0=self.P0, Q=self.Q_k.numpy(), R=self.R_k.numpy()
        )

    def run_mb_filter(self, X, Y, Cw, U=None):
        _, Ty, dy = Y.shape
        _, Tx, dx = X.shape

        if len(Y.shape) == 3:
            N, T, d = Y.shape
        elif len(Y.shape) == 2:
            T, d = Y.shape
            N = 1
            Y = Y.reshape((N, Ty, d))

        traj_estimated = torch.zeros((N, Tx, self.n_states), device=self.device).type(
            torch.FloatTensor
        )
        Pk_estimated = torch.zeros(
            (N, Tx, self.n_states, self.n_states), device=self.device
        ).type(torch.FloatTensor)

        MSE_PF_linear_arr = torch.zeros((N,)).type(torch.FloatTensor)

        start = timer()
        for i in range(0, N):
            self.initialize(R_k=Cw[i])
            y_in = Y[i, :, :].numpy().squeeze()
            simulator_ = simulator.Simulator(self, u=None, y=y_in)
            simulator_.simulate(self.n_particles, 0, filter="PF", meas_first=True)
            traj_estimated[i, :, :] = torch.from_numpy(
                simulator_.get_filtered_mean()
            ).type(torch.FloatTensor)
            filtered_particles_estimated, filtered_weights = (
                simulator_.get_filtered_estimates()
            )
            filtered_particles_estimated = (
                torch.from_numpy(filtered_particles_estimated)
                .type(torch.FloatTensor)
                .transpose(1, 0, 2)
            )
            filtered_weights = (
                torch.from_numpy(filtered_weights)
                .type(torch.FloatTensor)
                .transpose(1, 0)
            )
            filtered_deviations_estimated = (
                filtered_particles_estimated - traj_estimated[i, :, :]
            )
            Pk_estimated[i, :, :, :] = torch.einsum(
                "nt,ntij->tij",
                filtered_weights,
                (
                    torch.einsum(
                        "nti,ntj->ntij",
                        filtered_deviations_estimated,
                        filtered_deviations_estimated,
                    )
                ),
            )

            # MSE_PF_linear_arr[i] = mse_loss(traj_estimated[i], X[i]).item()
            MSE_PF_linear_arr[i] = (
                mse_loss(X[i, 1:, :], traj_estimated[i, 1:, :]).mean().item()
            )
            # print("pf, sample: {}, mse_loss: {}".format(i+1, MSE_PF_linear_arr[i]))

        end = timer()
        t = end - start

        mse_pf_dB_avg = torch.mean(10 * torch.log10(MSE_PF_linear_arr), dim=0)
        print("PF - MSE LOSS:", mse_pf_dB_avg, "[dB]")
        print(
            "PF - MSE STD:",
            torch.std(10 * torch.log10(MSE_PF_linear_arr), dim=0),
            "[dB]",
        )
        # Print Run Time
        print("Inference Time:", t)

        return traj_estimated, Pk_estimated, MSE_PF_linear_arr.mean(), mse_pf_dB_avg
