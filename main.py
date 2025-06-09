#%% part 0 - setup files

#to-do= add if __name__ == "__main__": to other scripts
# separate env for model_bayesian so separate the import
# link bash file setup to config? 
# link task config to pretrain_rnn_with_heli_v5.py
# link model_rnn.py to pretrain_rnn_with_heli_v5.py
# link model weights save output location to main.py
# find out where compile.py is used in the pipeline (takes best seed from what data?)
# same with analyze_compiled 

###runs through the entire pipeline
import os
import pkgutil
import importlib

#local modules
import config
import model_rnn
import utils_data, utils_calcs
import test_rnn
import plots_behav, plots_rnn

# Dynamically import all modules in the current folder (excluding this script and 'model_bayesian') 
# current_file = os.path.splitext(os.path.basename(__file__))[0]
# folder_path = os.path.dirname(os.path.abspath(__file__))
# for finder, name, ispkg in pkgutil.iter_modules([folder_path]):
#     if name not in {current_file, "model_bayesian"}:
#         importlib.import_module(name)

# Set the version here or make a new version in config.py
version = "V5"  # or "V3"

#%% part 1 - pretrain the RNN model

grid = config.pretrain_hp_info(version)

# Write export_hyperparams.py for bash
with open("export_pretrain_params.sh", "w") as f:
    for key, val in grid.items():
        # Handle both list and int types for values
        if isinstance(val, (list, tuple)):
            values = val
        else:
            values = [val]
        bash_array = ' '.join(map(str, values))
        # Write the evaluated values directly as a shell variable assignment
        f.write(f"{key}_values=({bash_array})\n")

#run slurm_seeds.sh on cluster

#save the model weights post training
#to-do: print save location from config

#plot the results of training (post training and post test set)

#to-do- separate pretraining plots out

#%% part 2 - test the trained RNN  

## to test a single model: 
task_info = config.task_info()
rnn_info = config.rnn_info() 

#init rnn model
actor_critic = model_rnn.ActorCritic(input_dim = rnn_info["input_dim"],
                                  hidden_dim = rnn_info["hidden_dim"],
                                  action_dim = rnn_info["action_dim"],
                                  gain = rnn_info["gain"],
                                  noise = rnn_info["noise"])


# save behav and rnn activity output from fixed weights 
all_states, rnn_activity = model_rnn.rnn_predict(rnn_model = actor_critic, 
         model_path = "model_params_101000/70.0_V3_0.95g_0.0rm_30bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_29s.pth",
         trials = task_info["n_trials"],
         contexts = task_info["contexts"],
         epochs = task_info["epochs"],
         reset_memory = rnn_info["reset_memory"],
) 

#%% part 2b - test the trained RNN with hyperparameter sweeps 

weights_dir, rnn_weights_dict = config.rnn_weights_info(version = "V5") 
rnn_info = config.rnn_info() #get the reference info for the RNN model

actor_critic = model_rnn.ActorCritic(input_dim = rnn_info["input_dim"],
                                  hidden_dim = rnn_info["hidden_dim"],
                                  action_dim = rnn_info["action_dim"],
                                  gain = rnn_info["gain"],
                                  noise = rnn_info["noise"])

save_dir_behav = "data/rnn_behav/model_params_101000/test/"
save_dir_rnn = "data/rnn_activity/model_params_101000/test/"
test_epochs = 30


test_rnn.test_rnn(model = actor_critic,
        hp_list = rnn_weights_dict,
         epochs = test_epochs,
         data_dir = weights_dir,
         save_dir_behav = save_dir_behav,
         save_dir_rnn = save_dir_rnn
)

#%% part 3 - analyze the saved outputs from the RNN model


weights_dir, rnn_weights_dict = config.rnn_weights_info(version = "V5") 
rnn_info = config.rnn_info() #get the reference info for the RNN model

# decide threshold to filter the models (by RNN hyperparam)
#currently setup to filter by Δarea performance

model_filters = utils_data.get_model_filters(
    hp_list = rnn_weights_dict,
    data_dir = save_dir_behav,
    threshold = 10
)

# analyze with behav plots

hp_names = list(rnn_weights_dict.keys())
behav_dict = utils_data.get_batch_behav(
    file_dir = weights_dir
    hp_list = hp_names,
    model_filters = model_filters
)
plots_behav.run_all_plots(behav_dict)

# rnn plts
#needs seeds / (setup in compile.py?)
rnn_act_dict = utils_data.get_rnn_activity(
    file_dir = "data/rnn_behav/model_params_101000",
    hp_list = hp_names,
    model_filters = model_filters
)
plots_rnn.plot_combined_state_space_from_dict(
    rnn_act_dict = rnn_act_dict, 
    model_path = "./model_params/36.0_V3_0.0ns_Nonelb_Noneub_0.95g_64n_40000e_2s.pth")



# rnn analyses - fixed points (setup to run with individual models)

analyze_rnn = analyze_rnn_fp.analyze_fixed_points(
    model = actor_critic,
    rnn_act_dict = rnn_act_dict,
    model_name = "12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s"
)

fixedpt_analysis.run_fp_analysis(
    model = actor_critic,
    rnn_act_dict = rnn_info_dict,
    model_name = "12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s",
    save_dir = "data/fixed_points/model_params_101000/"
)
# analyze with bayesian models

model_pyem(config)
#need shawn's model figures here


# %% part 4 - other analysis in progress, model checks

# reproduce behavioral data
#run human_data.py



