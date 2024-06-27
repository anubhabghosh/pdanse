#!/bin/bash
n_states=3
n_obs=2
N=500
T=1000
dataset_type="LorenzSSMn${n_obs}"
sigma_e2_dB=-10.0
model_type="gru"
script_name="main_danse_semisupervised_opt.py"
n_sup=10

for smnr in 0.0
do
	python3.7 ${script_name} \
	--mode train \
	--rnn_model_type ${model_type} \
	--dataset_type ${dataset_type} \
    	--n_sup ${n_sup} \
	--datafile ./data/synthetic_data/trajectories_m_${n_states}_n_${n_obs}_${dataset_type}_data_T_${T}_N_${N}_sigmae2_${sigma_e2_dB}dB_smnr_$(echo $smnr)dB.pkl \
	--splits ./data/synthetic_data/splits_m_${n_states}_n_${n_obs}_${dataset_type}_data_T_${T}_N_${N}_sigmae2_${sigma_e2_dB}dB_smnr_$(echo $smnr)dB.pkl
done
