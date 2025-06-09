#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=str, required=False, help='analysis', default='rollout')
args, unknown = parser.parse_known_args()
print(args)

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from tasks import PIE_CP_OB_v2
# from torch.distributions import Categorical
# import glob
# import utils_data as utils_data
# import utils_calcs as utils_calcs






analysis = args.analysis
epochs = 1
seeds = 50

data_dir = "./model_params_2/"
bias = False


if analysis == 'gamma':
    # influence of gamma

    gammas = [0.999, 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    areas = np.zeros([len(gammas), seeds, 2])
    validms = np.zeros(len(gammas), dtype=int)
    for g, gamma in enumerate(gammas):
        
        file_names= data_dir+f"*_V5_{gamma}g_0.0rm_50bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
        # file_names= data_dir+f"*_V3_{gamma}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(gamma, len(models))
        validms[g] = len(models)

        for m,model in enumerate(models):
            
            all_states = utils_calcs.get_area(model, epochs=epochs)
            areas[g,m], _, _ = utils_calcs.get_lrs(all_states)


    plot_param_area(gammas, areas, '$\gamma$', validms)



if analysis == 'rollout':
    # influence of rollout
    rollouts = [5, 10,20, 30, 40, 50, 75, 100, 150, 200] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    areas = np.zeros([len(rollouts), seeds, 2])
    validms = np.zeros(len(rollouts), dtype=int)
    for g, rollout in enumerate(rollouts):

        file_names= data_dir+f"*_V5_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
        # file_names= data_dir+f"*_V3_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(rollout, len(models))
        validms[g] = len(models)

        for m,model in enumerate(models):
            
            all_states = utils_calcs.get_area(model, epochs=epochs)
            areas[g,m], _, _ = utils_calcs.get_lrs(all_states)


    plot_param_area(rollouts, areas, '$t_{rollout}$',validms, logx=True)

# introduce variables into sampling
if analysis == 'preset':
    # influence of rollout

    presets = [0.0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1

    areas = np.zeros([len(presets), seeds, 2])
    validms = np.zeros(len(presets), dtype=int)
    for g, preset in enumerate(presets):

        file_names= data_dir+f"*_V5_0.95g_{preset}rm_50bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
        # file_names = data_dir+f"*_V3_0.95g_{preset}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(preset, len(models))
        validms[g] = len(models)

        for m,model in enumerate(models):
            all_states = utils_calcs.get_area(model, epochs=epochs, reset_memory=preset)
            areas[g, m], _, _ = utils_calcs.get_lrs(all_states)

    plot_param_area(presets, areas, '$p_{reset}$',validms, logx=False)


if analysis == 'noise':
    # influence of rollout

    noises = [0.0, 0.00001, 0.000025, 0.0001, 0.00025, 0.001, 0.0025, 0.01] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
        
    areas = np.zeros([len(noises), seeds, 2])
    validms = np.zeros(len(noises), dtype=int)
    for g, noise in enumerate(noises):

        file_names= data_dir+f"*_V5_0.95g_0.0rm_50bz_{noise}td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
        # file_names = data_dir+f"*_V3_0.95g_0.0rm_100bz_{noise}td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth" 
        models = glob.glob(file_names)
        print(noise, len(models))
        validms[g] = len(models)

        for m,model in enumerate(models):
            all_states = utils_calcs.get_area(model, epochs=epochs)
            areas[g, m], _, _ = utils_calcs.get_lrs(all_states)


    plot_param_area(noises, areas, '$\sigma_{noise}$',validms, logx=True)


if analysis == 'scale':
    # influence of rollout

    scales = [0.1, 0.25, 0.5,0.75, 0.9, 1.0, 1.1, 1.25, 1.5] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
        
    areas = np.zeros([len(scales), seeds, 2])
    validms = np.zeros(len(scales), dtype=int)
    for g, scale in enumerate(scales):

        file_names= data_dir+f"*_V5_0.95g_0.0rm_50bz_0.0td_{scale}tds_64n_50000e_10md_5.0rz_*s.pth"
        # file_names = data_dir+f"*_V3_0.95g_0.0rm_100bz_0.0td_{scale}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth" 
        models = glob.glob(file_names)
        print(scale, len(models))
        validms[g] = len(models)

        for m,model in enumerate(models):
            all_states = utils_calcs.get_area(model, epochs=epochs)
            areas[g, m], _, _ = utils_calcs.get_lrs(all_states)


    plot_param_area(scales, areas, '$\\beta_{\delta}$',validms)


