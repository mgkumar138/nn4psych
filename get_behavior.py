#%%

import numpy as np
import matplotlib.pyplot as plt
import utils_data, utils_calcs, ref_info
import torch
from tasks import PIE_CP_OB_v2
from torch.distributions import Categorical
import glob

import os
import pickle


analysis = 'all'
epochs = 30 #different format if more than 1 epoch (need to standardize this later)
#current format was made for pyem, but only saves 1st epoch

data_dir, ref_info = ref_info.ref_info(version = "V3")
save_dir = "data/rnn_behav/model_params_101000/30_epochs/" 
os.makedirs(save_dir, exist_ok=True)

bias = False


if analysis == 'gamma' or analysis == "all":
    # influence of gamma

    gammas = ref_info['gammas'] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    # gammas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'gammas':gammas,'states':[]}

    gamma_dict = {}
    gamma_cp_list = [] 
    gamma_ob_list = []
    gamma_models = []
    
    for g, gamma in enumerate(gammas):
        
        file_names = data_dir + ref_info['gamma']['file_pattern']

        # file_names= data_dir+f"*_V3_{gamma}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"

        print(gamma, len(models))

        for m, model in enumerate(models):
            
            all_states = utils_calcs.get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            gamma_dict[m, g] = {"gamma", gamma}
            if epochs == 1:
                gamma_cp_list.append(all_states[0,0])
                gamma_ob_list.append(all_states[0,1])  
            else:
                gamma_cp_list.append(all_states[:,0])
                gamma_ob_list.append(all_states[:,1]) 
            gamma_models.append(model)

            with open(os.path.join(save_dir, "gamma_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "gamma_dict.pkl"), "wb") as f:
                pickle.dump(gamma_dict, f)
            with open(os.path.join(save_dir, "gamma_cp_list.pkl"), "wb") as f:
                pickle.dump(gamma_cp_list, f)
            with open(os.path.join(save_dir, "gamma_ob_list.pkl"), "wb") as f:
                pickle.dump(gamma_ob_list, f)


if analysis == 'rollout' or analysis == 'all':
    # influence of rollout
    rollouts = [5, 10, 20, 30, 50, 75, 100, 150, 200] # took out 40 as running into action taking issues
    all_param_states = {'rollouts':rollouts,'states':[]}

    rollout_dict = {}
    rollout_cp_list = [] 
    rollout_ob_list = []
    rollout_models = []
    
    for g, rollout in enumerate(rollouts):

        file_names= data_dir+f"*_V3_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"
        print(rollout, len(models))

        for m,model in enumerate(models):
            
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            rollout_dict[m, g] = {"rollout", rollout}
            if epochs == 1:
                rollout_cp_list.append(all_states[0,0])
                rollout_ob_list.append(all_states[0,1])  
            else:
                rollout_cp_list.append(all_states[:,0])
                rollout_ob_list.append(all_states[:,1])
            rollout_models.append(model)  

            with open(os.path.join(save_dir, "rollout_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "rollout_dict.pkl"), "wb") as f:
                pickle.dump(rollout_dict, f)
            with open(os.path.join(save_dir, "rollout_cp_list.pkl"), "wb") as f:
                pickle.dump(rollout_cp_list, f)
            with open(os.path.join(save_dir, "rollout_ob_list.pkl"), "wb") as f:
                pickle.dump(rollout_ob_list, f)

# introduce variables into sampling
if analysis == 'preset' or analysis == 'all':
    # influence of preset

    presets = [0.0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'presets':presets,'states':[]}

    preset_dict = {}
    preset_cp_list = [] 
    preset_ob_list = []
    preset_models = []
    
    for g, preset in enumerate(presets):

        file_names = data_dir+f"*_V3_0.95g_{preset}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"
        print(preset, len(models))

        for m,model in enumerate(models):
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            preset_dict[m, g] = {"preset", preset}
            if epochs == 1:
                preset_cp_list.append(all_states[0,0])
                preset_ob_list.append(all_states[0,1])  
            else:
                preset_cp_list.append(all_states[:,0])
                preset_ob_list.append(all_states[:,1])
            preset_models.append(model)

            with open(os.path.join(save_dir, "preset_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "preset_dict.pkl"), "wb") as f:
                pickle.dump(preset_dict, f)
            with open(os.path.join(save_dir, "preset_cp_list.pkl"), "wb") as f:
                pickle.dump(preset_cp_list, f)
            with open(os.path.join(save_dir, "preset_ob_list.pkl"), "wb") as f:
                pickle.dump(preset_ob_list, f)

if analysis == 'scale' or analysis == 'all':
    # influence of scale

    scales =  [0.25, 0.5,0.75, 1.0, 1.25, 1.5] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    all_param_states = {'scales':scales,'states':[]}

    scale_dict = {}
    scale_cp_list = [] 
    scale_ob_list = []
    scale_models = []
    
    for g, scale in enumerate(scales):

        # file_names= data_dir+f"*_V5_0.95g_0.0rm_50bz_0.0td_{scale}tds_64n_50000e_10md_5.0rz_*s.pth"
        file_names = data_dir+f"*_V3_0.95g_0.0rm_100bz_0.0td_{scale}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth" 
        models = glob.glob(file_names)
        models = [f for f in models if float(f.split("\\")[-1].split("_")[0]) > 5] # remove items in file_names that begin with "-"
        print(scale, len(models))


        for m,model in enumerate(models):
            all_states = get_area(model, epochs=epochs)
            all_param_states['states'].append(all_states)

            scale_dict[m, g] = {"scale", scale}
            if epochs == 1:
                scale_cp_list.append(all_states[0,0])
                scale_ob_list.append(all_states[0,1])  
            else: 
                scale_cp_list.append(all_states[:,0])
                scale_ob_list.append(all_states[:,1])
            scale_models.append(model)

            with open(os.path.join(save_dir, "scale_all_param_states.pkl"), "wb") as f:
                pickle.dump(all_param_states, f)
            with open(os.path.join(save_dir, "scale_dict.pkl"), "wb") as f:
                pickle.dump(scale_dict, f)
            with open(os.path.join(save_dir, "scale_cp_list.pkl"), "wb") as f:
                pickle.dump(scale_cp_list, f)
            with open(os.path.join(save_dir, "scale_ob_list.pkl"), "wb") as f:
                pickle.dump(scale_ob_list, f)

#combined_dict doesn't work because of overlapping keys

if analysis == 'all': 
    cp_array  = [gamma_cp_list, rollout_cp_list, preset_cp_list, scale_cp_list]
    ob_array  = [gamma_ob_list, rollout_ob_list, preset_ob_list, scale_ob_list]
    mod_array = [gamma_models, rollout_models, preset_models, scale_models]

    with open(os.path.join(save_dir, "combined_cp_array_filtered.pkl"), "wb") as f:
        pickle.dump(cp_array, f)
    with open(os.path.join(save_dir, "combined_ob_array_filtered.pkl"), "wb") as f:
        pickle.dump(ob_array, f)
    with open(os.path.join(save_dir, "combined_mod_array_filtered.pkl"), "wb") as f:
        pickle.dump(mod_array, f)