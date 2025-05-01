import os
import pkgutil
import importlib

#to-do= add if __name__ == "__main__": to other scripts


###runs through the entire pipeline

# Dynamically import all modules in the current folder (excluding this script)
current_file = os.path.splitext(os.path.basename(__file__))[0]
folder_path = os.path.dirname(os.path.abspath(__file__))
for finder, name, ispkg in pkgutil.iter_modules([folder_path]):
    if name != current_file:
        importlib.import_module(name)

import config
import model_rnn, model_bayesian
import utils_data, utils_calcs



#either draw from preset hyperparams (in config) or make new ones to

#choose the task parameters (if not using pre-configured task)


#setup the RNN model

model_rnn = model_rnn.ActorCritic(config)
#decide on the SLURM settings or run locally

#train the RNN model

pretrain_rnn_with_heli_v5(model_rnn, config) #pretrain the model with the helicopter task

#save the model weights post training
#to-do: print save location from config


#plot the results of training (post training and post test set)

#to-do- separate pretraining plots out

#%% part 2 - run the RNN model with fixed weights

# decide how many under what task conditions to run analyses for (epochs)

data_dir, config = config.ref_info(version = "V3")

#setup model if not done so
model_rnn = model_rnn.ActorCritic(config, data_dir, save_dir = "data/rnn_behav/model_params_101000/30_epochs/") 


# save behav and rnn activity output from fixed weights 

test_rnn(model_rnn, config) 


#%% part 3 - analyze the saved outputs from the RNN model

# decide threshold to filter the models (by RNN hyperparam)
#currently setup to filter by Δarea performance

model_filters = utils_data.get_model_filters(config, threshold = 10) 


# analyze with behav plots

behav_dict = utils_data.get_batch_behav(config, model_filters)
plots_behav(behav_dict) 

# analyze with rnn plots
#needs seeds / (setup in compile.py?)
rnn_act_dict = utils_data.get_rnn_activity(config, model_filters)
plots_rnn(rnn_act_dict)

# analyze with bayesian models

model_pyem(config)
#need shawn's model figures here


# %% part 4 - other analysis in progress, model checks