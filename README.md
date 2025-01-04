# Particle DANSE: ICASSP 2025
This repository contains code related to the ICASSP 2025 paper: 

*Particle-based Data-driven Nonlinear State Estimation of Model-free Process from Nonlinear Measurements*

Authors: Anubhab Ghosh, Yonina C. Eldar and Saikat Chatterjee

## Dependencies 
It is recommended to build an environment either in [`pip`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [`conda`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and install the following packages (I used `conda` as personal preference):
- PyTorch (1.6.0)
- Python (>= 3.8.20) with standard packages as part of an Anaconda installation such as Numpy, Scipy, Matplotlib, etc. The settings for the code were:
    - Numpy (1.23.5)
    - Matplotlib (3.7.2)
    - Scipy (1.10.1)
    - Scikit-learn (1.2.2)

- `pyparticleest` (for implementation of Particle filter (PF)): [Github](https://github.com/jerkern/pyParticleEst) [Docs](https://pyparticleest.readthedocs.io/en/latest/index.html)
- `filterpy` (1.4.5) (for implementation of Unscented Kalman Filter (UKF)): [https://filterpy.readthedocs.io/en/latest/](https://filterpy.readthedocs.io/en/latest/)
- Jupyter notebook (>= 6.4.6) (for result analysis)
- Tikzplotlib (for figures) [https://github.com/nschloe/tikzplotlib](https://github.com/nschloe/tikzplotlib) (for a possible bug related to `webcolors` check this StackOverflow [page](https://stackoverflow.com/questions/78672058/tikzplotlib-module-throws-attribute-error-module-webcolors-has-no-attribute)

## Datasets used 

The experiments were carried out using synthetic data generated with linear and non-linear SSMs:

- Non-linear state space models (Non-linear SSMs): In our case, we used chaotic attractors:
    - Lorenz attractor
 
and a rectified linear measurement function (ReLU(x) = max(0, x)) as the nonlinearity. 
  
Details about these models and their underlying dynamics can be found in `./bin/ssm_models.py`. 

## Reference models (implemented in PyTorch + Numpy)

- Particle filter (PF)

NOTE: The testing code also has functionality to test against other model-based filters such as the extended Kalman filter (EKF) and the unscented Kalman filter (UKF)

## GPU Support

The training-based methods: pDANSE was run on a single NVIDIA-Tesla P100 GPU with 16 GB of memory. 

## Code organization
This would be the required organization of files and folders for reproducing results. If certain folders are not present, they should be created at that level.

````
- main_pdanse_opt.py (main function for training 'pDANSE' model)
- ...

- data/ (contains stored datasets in .pkl files)
| - synthetic_data/ (contains datasets related to SSM models in .pkl files)

- src/ (contains model-related files)
| - pdanse.py (for training pDANSE)
| - rnn.py (class definition of the RNN model for pDANSE)
| ...

- log/ (contains training and evaluation logs, losses in `.json`, `.log` files)
- models/ (contains saved model checkpoints as `.pt` files)
- figs/ (contains resulting model figures)
- utils/ (contains helping functions)
- tests/ (contains files and functions for evaluation at test time)
- config/ (contains the parameter file)
| - parameters_opt.py (Python file containing relevant parameters for different architectures)

- bin/ (contains data generation files)
| - ssm_models.py (contains the classes for state space models)
| - generate_data.py (contains code for generating training datasets)

- run/ (folder containing the shell scripts to run the `main` scripts or data-generation scripts at one go for different smnr_dB)
| - run_main_pdanse.sh

- analysis_ipynbs/ (contains Jupyter notebooks to collect and visualize results)

````

## Brief outline of pDANSE training

1. Generate data by calling `bin/generate_data.py`. This can be done in a simple manner by editing and calling the shell script `run_generate_data.sh`. Data gets stored at `data/synthetic_data/`. For e.g. to generate trajectory data with 1000 samples with each trajectory of length 100, from a Lorenz Attractor model (m=3, n=3), with $\sigma_{e}^{2}= -10$ dB, and $\text{SMNR}$ = $0$ dB, the syntax should be 
````
[PYTHON KERNEL] ./bin/generate_data.py --n_states 3 --n_obs 3 --num_samples 1000 --sequence_length 100 --sigma_e2_dB -10 --smnr 0 --dataset_type LorenzSSM --output_path ./data/synthetic_data/
````
2. Edit the hyper-parameters for the DANSE architecture in `./config/parameters_opt.py`.

3. Run the training for DANSE by calling `main_pdanse_opt.py`.  E.g. to run a pDANSE model employing a GRU architecture as the RNN, with using $\kappa=2 \\%$ supervised data, i.e. $N_{sup}=\kappa \times N / 100 =0.02 \times 1000 = 20$ labelled data samples and remaining $(1 - \kappa) * N / 1000$ unlabelled data samples ($\because N = 1000$) using the Lorenz attractor dataset as described above, the syntax should be
```
[PYTHON KERNEL] main_pdanse_opt.py \
--mode train \
--rnn_model_type gru \
--dataset_type LorenzSSM \
--n_sup 20 \
--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_relu_data_T_100_N_1000_sigmae2_-10.0dB_smnr_0.0dB.pkl \
--splits ./data/synthetic_data/splits_m_3_n_3_LorenzSSM_relu_data_T_100_N_1000_sigmae2_-10.0dB_smnr_0.0dB.pkl
```
For the `datafile` and `splits` arguments:
`N` denotes the number of sample trajectories, `T` denotes the length of each sample trajectory. 

4. To reproduce experiments, for multiple SMNRs, run the shell script `./run/run_main_pdanse.sh`

## Evaluation

Once files are created, the evaluation can be done by calling the script in `/tests/test_models_with_danse.py`. Paths to model files and log files should be edited in the script directly. The results can be visualized using Jupyter notebooks found in `analysis_ipynbs/`
