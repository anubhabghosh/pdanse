#####################################################
# Creator: Anubhab Ghosh
# Nov 2023
#####################################################

import sys
from os import path

import numpy as np
import torch
from scipy.integrate import solve_ivp

from bin.measurement_fns import get_measurement_fn
from utils.utils import dB_to_lin

# __file__ should be defined in this case
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)


class LinearSSM(object):
    def __init__(
        self,
        n_states=4,
        n_obs=1,
        delta=0.1,
        measurement_fn_type="dist_sq",
        mu_e=None,
        mu_w=None,
    ):
        self.n_states = n_states
        self.n_obs = n_obs
        self.delta = delta
        self.measurement_fn_type = measurement_fn_type
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.h_fn = get_measurement_fn(fn_name=self.measurement_fn_type)
        self.construct_F()
        self.construct_H()

    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * torch.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * torch.eye(self.n_obs)

    def construct_F(self):
        self.F_mat = np.array(
            [
                [0.0, 0.0, self.delta, 0.0],
                [0.0, 0.0, 0.0, self.delta],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

    def construct_H(self):
        self.H_mat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def f_fn(self, x):
        return self.F_mat @ x

    def generate_state_sequence(self, T, sigma_e2_dB):
        self.sigma_44e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        x_k_arr = np.zeros(T, self.n_states)
        e_k_arr = np.random.multivariate_normal(mean=self.mu_w, cov=self.Cw, size=(T,))
        for t in range(0, T - 1):
            x_k_arr[t + 1] = self.f_fn(x_k_arr[t]) + e_k_arr[t]
        return x_k_arr

    def generate_measurement_sequence(self, x_k_arr, T, smnr_dB):
        signal_power = np.var(self.h_fn(x_k_arr))
        self.sigma_w2 = signal_power / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(mean=self.mu_w, cov=self.Cw, size=(T,))
        y_k_arr = np.zeros((T, self.n_obs))

        for t in range(0, T):
            y_k_arr[t] = self.h_fn(self.H_mat @ x_k_arr[t]) + w_k_arr[t]

        return y_k_arr

    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):
        x_seq = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_seq = self.generate_measurement_sequence(x_lorenz=x_seq, T=T, smnr_dB=smnr_dB)
        # print(x_lorenz.shape, y_lorenz.shape)
        return x_seq, y_seq, self.Cw


class Nonlinear1DSSM(object):
    def __init__(
        self,
        n_states,
        n_obs,
        a=0.5,
        b=25.0,
        c=8.0,
        d=0.05,
        measurement_fn_type="square",
        mu_e=None,
        mu_w=None,
    ):
        self.n_states = n_states
        self.n_obs = n_obs
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.measurement_fn_type = measurement_fn_type
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.h_fn = get_measurement_fn(fn_name=self.measurement_fn_type)

    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_driving_noise(self, k):
        u_k = self.c * np.cos(1.2 * k)  # Previous idea (considering start at k=0)
        # u_k = np.cos(self.c * (k+1))  # Current modification (considering start at k=0)
        return u_k

    def f_fn(self, x_k):
        x_k_plus_1 = self.a * x_k + self.b * (x_k / (1.0 + x_k**2))
        return x_k_plus_1

    def generate_state_sequence(self, T, sigma_e2_dB):
        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        x_k_arr = np.zeros((T, self.n_states))
        e_k_arr = np.random.multivariate_normal(mean=self.mu_e, cov=self.Ce, size=(T,))
        for t in range(0, T - 1):
            u_k = self.generate_driving_noise(t)
            x_k_arr[t + 1] = self.f_fn(x_k_arr[t]) + u_k + e_k_arr[t]

        return x_k_arr

    def generate_measurement_sequence(self, x_k_arr, T, smnr_dB):
        signal_power = np.var(self.h_fn(x_k_arr))
        self.sigma_w2 = signal_power / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(mean=self.mu_w, cov=self.Cw, size=(T,))
        y_k_arr = np.zeros((T, self.n_obs))

        for t in range(0, T):
            y_k_arr[t] = self.h_fn(x_k_arr[t]) + w_k_arr[t]

        return y_k_arr

    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):
        x_seq = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_seq = self.generate_measurement_sequence(x_k_arr=x_seq, T=T, smnr_dB=smnr_dB)
        return x_seq, y_seq, self.Cw


class LorenzSSM(object):
    def __init__(
        self,
        n_states,
        n_obs,
        J,
        delta,
        delta_d,
        alpha=0.0,
        measurement_fn_type="square",
        decimate=False,
        mu_e=None,
        mu_w=None,
        H=None,
        use_Taylor=True,
    ) -> None:
        self.n_states = n_states
        self.J = J
        self.delta = delta
        self.alpha = alpha  # alpha = 0 -- Lorenz attractor, alpha = 1 -- Chen attractor
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.measurement_fn_type = measurement_fn_type
        self.h_fn = get_measurement_fn(fn_name=self.measurement_fn_type)
        self.mu_e = mu_e
        if H is None:
            self.H = np.eye(self.n_obs)
        else:
            self.H = H
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor

    def A_fn(self, z):
        return np.array(
            [
                [-(10 + 25 * self.alpha), (10 + 25 * self.alpha), 0],
                [(28 - 35 * self.alpha), (29 * self.alpha - 1), -z],
                [0, z, -(8.0 + self.alpha) / 3],
            ]
        )

    def linear_h_fn(self, x):
        """
        Linear measurement setup y = x + w
        """
        if type(x).__module__ == np.__name__:
            H_ = np.copy(self.H)
        elif type(x).__module__ == torch.__name__:
            H_ = torch.from_numpy(self.H).type(torch.FloatTensor)

        if len(x.shape) == 1:
            y_ = H_ @ x
        elif len(x.shape) > 1:
            if type(x).__module__ == np.__name__:
                y_ = np.einsum("ij,nj->ni", H_, x)
            elif type(x).__module__ == torch.__name__:
                y_ = H_ @ x
        return y_

    def f_linearize(self, x):
        self.F = np.eye(self.n_states)
        for j in range(1, self.J + 1):
            # self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)
            self.F += np.linalg.matrix_power(
                self.A_fn(x[0]) * self.delta, j
            ) / np.math.factorial(j)

        return self.F @ x

    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T, sigma_e2_dB):
        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        self.decimation_factor = int(self.delta / self.delta_d)
        x_lorenz = np.zeros((T, self.n_states)) + 1e-6
        e_k_arr = np.random.multivariate_normal(self.mu_e, self.Ce, size=(T,))
        # print(x_lorenz.shape)
        for t in range(0, T - 1):
            x_lorenz[t + 1] = self.f_linearize(x_lorenz[t]) + e_k_arr[t]

        if self.decimate:
            x_lorenz_d = x_lorenz[0 : T : self.decimation_factor, :]
        else:
            x_lorenz_d = np.copy(x_lorenz)

        return x_lorenz_d

    def generate_measurement_sequence(self, x_lorenz, T, smnr_dB=10.0):
        # signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        hx_lorenz = np.asarray([self.h_fn(x_lorenz[i,:]).reshape((-1,)) for i in range(x_lorenz.shape[0])])
        signal_p = np.var(hx_lorenz)
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_lorenz = np.zeros((T, self.n_obs))

        # print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))

        # print(self.H.shape, x_lorenz.shape, y_lorenz.shape)
        for t in range(0, T):
            if self.measurement_fn_type == "linear" or self.measurement_fn_type == "identity":
                y_lorenz[t] = self.linear_h_fn(x_lorenz[t]) + w_k_arr[t]
            else:
                y_lorenz[t] = self.h_fn(x_lorenz[t]) + w_k_arr[t]

        return y_lorenz

    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):
        # print(T)
        x_lorenz = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        #x_lorenz = (x_lorenz - np.mean(x_lorenz, 0)) / np.std(x_lorenz, 0)
        y_lorenz = self.generate_measurement_sequence(
            x_lorenz=x_lorenz, T=T // self.decimation_factor, smnr_dB=smnr_dB
        )

        # print(x_lorenz.shape, y_lorenz.shape)
        return x_lorenz, y_lorenz, self.Cw


class RosslerSSM(object):
    def __init__(
        self,
        n_states,
        n_obs,
        J,
        delta,
        delta_d,
        a=0.1,
        b=0.1,
        c=14.0,
        decimate=False,
        measurement_fn_type="square",
        mu_e=None,
        mu_w=None,
        H=None,
        use_Taylor=True,
    ) -> None:
        self.n_states = n_states
        self.J = J
        self.delta = delta
        self.a = a
        self.b = b
        self.c = c
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.measurement_fn_type = measurement_fn_type
        self.h_fn = get_measurement_fn(fn_name=self.measurement_fn_type)
        self.mu_e = mu_e
        if H is None:
            self.H = np.eye(self.n_obs)
        else:
            self.H = H
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor

    def A_fn(self, z):
        return np.array(
            [[0, -1, -1], [1, self.a, 0], [0, 0, (self.b / z[2]) + (z[0] - self.c)]]
        )

    # def A_fn(self, z):
    #    return np.array([
    #                [-10, 10, 0],
    #                [28, -1, -z],
    #                [0, z, -8.0/3]
    #            ])

    def linear_h_fn(self, x):
        """
        Linear measurement setup y = x + w
        """
        if type(x).__module__ == np.__name__:
            H_ = np.copy(self.H)
        elif type(x).__module__ == torch.__name__:
            H_ = torch.from_numpy(self.H).type(torch.FloatTensor)

        if len(x.shape) == 1:
            y_ = H_ @ x
        elif len(x.shape) > 1:
            if type(x).__module__ == np.__name__:
                y_ = np.einsum("ij,nj->ni", H_, x)
            elif type(x).__module__ == torch.__name__:
                y_ = H_ @ x
        return y_

    def f_linearize(self, x):
        self.F = np.eye(self.n_states)
        for j in range(1, self.J + 1):
            # self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)
            # print(self.A_fn(x))
            self.F += np.linalg.matrix_power(
                self.A_fn(x) * self.delta, j
            ) / np.math.factorial(j)

        return self.F @ x

    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)
        self.Ce[self.n_states - 1, self.n_states - 1] = (
            1e-10  # Set the noise covariance in the z-dim to be zero, basically making it noise-free as it is zero mean!
        )

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T, sigma_e2_dB):
        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        self.decimation_factor = int(self.delta / self.delta_d)
        x_rossler = np.zeros((T, self.n_states))
        x_rossler[0, :] = np.ones(
            self.n_states
        )  # * np.abs(np.random.normal(0, np.sqrt(self.sigma_e2), (1,)))
        e_k_arr = np.random.multivariate_normal(self.mu_e, self.Ce, size=(T,))

        # print(x_rossler[0, :])

        for t in range(0, T - 1):
            x_rossler[t + 1] = self.f_linearize(x_rossler[t]) + e_k_arr[t]

        if self.decimate is True:
            x_rossler_d = x_rossler[0 : T : self.decimation_factor, :]
        else:
            x_rossler_d = np.copy(x_rossler)

        return x_rossler_d

    def generate_measurement_sequence(self, x_rossler, T, smnr_dB=10.0):
        signal_p = np.var(self.h_fn(x_rossler))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_rossler = np.zeros((T, self.n_obs))

        # print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))

        # print(self.H.shape, x_rossler.shape, y_rossler.shape)
        for t in range(0, T):
            if self.measurement_fn_type == "linear" or self.measurement_fn_type == "identity":
                y_rossler[t] = self.linear_h_fn(x_rossler[t]) + w_k_arr[t]
            else:
                y_rossler[t] = self.h_fn(x_rossler[t]) + w_k_arr[t]

        return y_rossler

    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):
        x_rossler = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_rossler = self.generate_measurement_sequence(
            x_rossler=x_rossler, T=T // self.decimation_factor, smnr_dB=smnr_dB
        )
        return x_rossler, y_rossler, self.Cw


def L96(t, x, N=20, F_mu=8, sigma_e2=0.1):
    """Lorenz 96 model with constant forcing
    Adapted from: https://www.wikiwand.com/en/Lorenz_96_model
    """
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    F_N = np.random.normal(
        loc=F_mu, scale=np.sqrt(sigma_e2), size=(N,)
    )  # Incorporating Process noise through the forcing constant
    for i in range(N):
        # print(F_N[i])
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F_N[i]
    return d


class Lorenz96SSM(object):
    def __init__(
        self,
        n_states,
        n_obs,
        delta,
        delta_d,
        F_mu=8,
        decimate=False,
        measurement_fn_type="square",
        mu_w=None,
        H=None,
        method="RK45",
    ) -> None:
        self.n_states = n_states
        self.delta = delta
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.measurement_fn_type = measurement_fn_type
        self.h_fn = get_measurement_fn(fn_name=self.measurement_fn_type)
        self.F_mu = F_mu
        if H is None:
            self.H = np.eye(self.n_obs)
        else:
            self.H = H
        self.mu_w = mu_w
        self.method = method

    def linear_h_fn(self, x):
        """
        Linear measurement setup y = x + w
        """
        if len(x.shape) == 1:
            y_ = self.H @ x
        elif len(x.shape) > 1:
            y_ = np.einsum("ij,nj->ni", self.H, x)
        return y_

    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T_time, sigma_e2_dB):
        self.decimation_factor = int(self.delta / self.delta_d)
        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        x0 = self.F_mu * np.ones(self.n_states)  # Initial state (equilibrium)
        x0[0] += self.delta  # Add small perturbation to the first variable
        sol = solve_ivp(
            L96,
            t_span=(0.0, T_time),
            y0=x0,
            args=(
                self.n_states,
                self.F_mu,
                self.sigma_e2,
            ),
            method=self.method,
            t_eval=np.arange(0.0, T_time, self.delta),
            max_step=self.delta,
        )

        x_lorenz = np.concatenate((sol.y.T, x0.reshape((1, -1))), axis=0)
        assert (
            x_lorenz.shape[-1] == self.n_states
        ), "Shape mismatch for generated state trajectory"

        T = x_lorenz.shape[0]

        if self.decimate is True:
            x_lorenz_d = x_lorenz[0 : T : self.decimation_factor, :]
        else:
            x_lorenz_d = np.copy(x_lorenz)

        return x_lorenz_d

    def generate_measurement_sequence(self, T, x_lorenz, smnr_dB=10.0):
        # signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        signal_p = np.var(self.h_fn(x_lorenz))
        # print("Signal power: {:.3f}".format(signal_p))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_lorenz = np.zeros((T, self.n_obs))

        # print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))

        # print(self.H.shape, x_lorenz.shape, y_lorenz.shape)
        for t in range(0, T):
            if self.measurement_fn_type == "linear" or self.measurement_fn_type == "identity":
                y_lorenz[t] = self.linear_h_fn(x_lorenz[t]) + w_k_arr[t]
            else:
                y_lorenz[t] = self.h_fn(x_lorenz[t]) + w_k_arr[t]

        return y_lorenz

    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):
        T_time = T * self.delta
        x_lorenz96 = self.generate_state_sequence(
            T_time=T_time, sigma_e2_dB=sigma_e2_dB
        )
        y_lorenz96 = self.generate_measurement_sequence(
            x_lorenz=x_lorenz96, T=T // self.decimation_factor, smnr_dB=smnr_dB
        )

        return x_lorenz96, y_lorenz96, self.Cw
