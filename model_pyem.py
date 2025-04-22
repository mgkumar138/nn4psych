import numpy as np
from scipy.special import expit
from scipy import stats
import sys
from pyEM.math import calc_fval

def norm2beta(x, max_val=5):
    return max_val / (1 + np.exp(-x))

def norm2alpha(x):
    return expit(x)

def fit(params, bucket_positions, bag_positions, context, prior=None, output='npl'):
    '''
    See Loosen et al (2023) Supplement for details (https://link.springer.com/article/10.3758/s13428-024-02427-y#Sec43)
    '''
    nparams = len(params)

    H = norm2alpha(params[0]) # hazard rate: frequency that the model expected extreme events 
    LW = norm2alpha(params[1]) # likelihood weight: the degree to which extremeness of an outcome factored into identification of change-points/oddballs
    UU = norm2alpha(params[2]) # uncertainty underestimation: degree to which uncertainty is inappropriately reduced on each trial
    sigma_motor = norm2beta(params[3]) # update variance: the base width of the distribution over possible bucket positions centred on the inferred helicopter location
    sigma_LR = norm2beta(params[4]) # update variance slope: the degree to which the width of the distribution over bucket positions increases with larger normative updates
    # drift_scale = norm2alpha(params[6]) # drift scale: the rate at which the helicopter was assumed to be drifting in the oddball condition
    sigma_N = 20 # this is the standard deviation of the normal distribution used to calculate the where the bag will be placed N(mu = helicopter position, sigma = 20)    
    
    # make sure params are in range
    all_bounds = [0, 1]
    for p in [H, LW, UU,]:
        if (p < all_bounds[0]) or (p > all_bounds[1]):
            return 1e7

    ntrials = bucket_positions.shape[0]
    pred_error = bag_positions - bucket_positions

    learning_rate    = np.zeros((ntrials,))
    omega            = np.zeros((ntrials,))
    tau              = np.zeros((ntrials+1,))
    U_val            = np.zeros((ntrials,))
    N_val            = np.zeros((ntrials,))
    pred_bucket_placement = np.zeros((ntrials,))
    bucket_update    = np.zeros((ntrials,))
    normative_update = np.zeros((ntrials,))
    L_normative_update = np.zeros((ntrials,))
    negll  = np.zeros((ntrials,))

    # initialize tau
    tau_0 = .5 / UU
    for t in range(ntrials):
        # changepoint probability (𝛺) 
        U_val[t] = stats.uniform.pdf(pred_error[t], 0, 300) ** LW
        if t == 0:
            sigma_t = sigma_N / tau_0
        else:
            sigma_t = sigma_N / tau[t]
        N_val[t] = stats.norm.pdf(pred_error[t], 0, sigma_t) ** LW
        omega[t] = (H * U_val[t]) / (H * U_val[t] + (1 - H) * N_val[t])
        
        # relative uncertainty (𝜏)
        if t == 0:
            this_tau = ((omega[t] * sigma_N) + ((1 - omega[t]) * sigma_t * tau_0) + (omega[t] * (1 - omega[t]) * (pred_error[t] * (1 - tau_0))**2)) / ((omega[t] * sigma_N) + ((1 - omega[t]) * sigma_t * tau_0) + (omega[t] * (1 - omega[t]) * (pred_error[t] * (1 - tau_0))**2) + sigma_N)
        else:
            this_tau = ((omega[t] * sigma_N) + ((1 - omega[t]) * sigma_t * tau[t]) + (omega[t] * (1 - omega[t]) * (pred_error[t] * (1 - tau[t]))**2)) / ((omega[t] * sigma_N) + ((1 - omega[t]) * sigma_t * tau[t]) + (omega[t] * (1 - omega[t]) * (pred_error[t] * (1 - tau[t]))**2) + sigma_N)
        tau[t+1] = this_tau / UU
        
        if context == 'changepoint':
            learning_rate[t] = omega[t] + tau[t] - (omega[t] * tau[t])
        elif context == 'oddball':
            learning_rate[t] = tau[t] - (omega[t] * tau[t])

        normative_update[t] = learning_rate[t] * pred_error[t]
        sigma_update = sigma_motor + normative_update[t] * sigma_LR
        
        bucket_update[t] = bucket_positions[t]-bucket_positions[t-1]
        L_normative_update[t] = stats.norm.pdf(bucket_update[t], loc=normative_update[t], scale=sigma_update)

        pred_bucket_placement[t] = bucket_positions[t-1] + normative_update[t]

        negll[t] = -np.log(L_normative_update[t]+1e-10)
    
    sum_negll = np.nansum(negll)
    # CALCULATE NEGATIVE POSTERIOR LIKELIHOOD FROM NEGLL AND PRIOR (OR NEGLL)
    if (output == 'npl') or (output == 'nll'):
        fval = calc_fval(sum_negll, params, prior=prior, output=output)
        return fval
    
    elif output == 'all':
        subj_dict = {'params'               : [H, LW, UU, sigma_motor, sigma_LR],
                     'bucket_positions'     :bucket_positions, 
                     'bag_positions'        :bag_positions, 
                     'context'              :context,
                     'pred_bucket_placement':pred_bucket_placement,
                     'learning_rate'        :learning_rate,
                     'pred_error'           :pred_error,
                     'omega'                :omega,
                     'tau'                  :tau,
                     'U_val'                :U_val,
                     'N_val'                :N_val,
                     'bucket_update'        :bucket_update,
                     'normative_update'     :normative_update,
                     'L_normative_update'   :L_normative_update,
                     'negll'                :sum_negll,
                     'BIC'                  : nparams * np.log(ntrials) + 2*sum_negll}
        return subj_dict

#%% run pyem model
import os
import pickle

#pull data from get_behavior.py
file_dir = "data/rnn_behav/model_params_101000/"

# load file_dir/gamma_cp_list.pkl
with open(os.path.join(file_dir, 'gamma_cp_list.pkl'), 'rb') as f:
    cp_array = pickle.load(f)

# load file_dir/gamma_ob_list.pkl
with open(os.path.join(file_dir, 'gamma_ob_list.pkl'), 'rb') as f:
    ob_array = pickle.load(f)

## load model_list_ob
with open(os.path.join(file_dir, 'gamma_dict.pkl'), 'rb') as f:
    model_list = pickle.load(f)

# # each array has: [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers]
agent_list_cp = [[x[1], x[2], 'changepoint'] for x in cp_array]
agent_list_ob = [[x[1], x[2], 'oddball'] for x in ob_array]

# outfit_cp = EMfit(agent_list_cp, fit, ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR'], mstep_maxit=20, convergence_custom='relative_npl', verbose=2)
# outfit_ob = EMfit(agent_list_ob, fit, ['H', 'LW', 'UU', 'sigma_motor', 'sigma_LR', 'sigma_LR'], mstep_maxit=20, convergence_custom='relative_npl', verbose=2)