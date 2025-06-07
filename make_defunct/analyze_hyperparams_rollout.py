#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=str, required=False, help='analysis', default='rollout')
args, unknown = parser.parse_known_args()
print(args)

# import numpy as np
# import matplotlib.pyplot as plt
# from utils_funcs import ActorCritic
# import torch
# from tasks import PIE_CP_OB_v2
# from torch.distributions import Categorical
# import glob
# from utils_funcs import saveload

def get_lrs_v2(states, threshold=20):
    true_state = states[2]  # bag position
    predicted_state = states[1]  # bucket position
    prediction_error = (true_state - predicted_state)[:-1]
    update = np.diff(predicted_state)

    idx = prediction_error !=0
    prediction_error= prediction_error[idx]
    update = update[idx]
    learning_rate = update / prediction_error

    prediction_error = abs(prediction_error)
    idx = prediction_error>threshold
    pes = prediction_error[idx]
    lrs = np.clip(learning_rate,0,1)[idx]

    sorted_indices = np.argsort(pes)
    prediction_error_sorted = pes[sorted_indices]
    learning_rate_sorted = lrs[sorted_indices]

    return prediction_error_sorted, learning_rate_sorted

def get_lrs(states):
    epochs = states.shape[0]
    pess, lrss, area = [],[], []
    for c in range(2):
        pes,lrs = [],[]
        for e in range(epochs):
            pe, lr = get_lrs_v2(states[e, c])

            pes.append(pe)
            lrs.append(lr)

        pes = np.concatenate(pes)
        lrs = np.concatenate(lrs)
        sorted_indices = np.argsort(pes)
        prediction_error_sorted = pes[sorted_indices]
        learning_rate_sorted = lrs[sorted_indices]

        pess.append(prediction_error_sorted)
        lrss.append(learning_rate_sorted)
        area.append(np.trapz(learning_rate_sorted, prediction_error_sorted))
    return area, pess, lrss

def get_area(model_path, epochs=100, reset_memory=0.0):
    hidden_dim = 64
    trials = 200

    model = ActorCritic(9, hidden_dim, 3)
    model.load_state_dict(torch.load(model_path))

    # print(f'Load Model {model_path}')
    contexts = ["change-point","oddball"] #"change-point","oddball"

    all_states = np.zeros([epochs, 2, 5, trials])
    for epoch in range(epochs):
        for tt, context in enumerate(contexts):
            env = PIE_CP_OB_v2(condition=context, max_time=300, total_trials=trials, 
                    train_cond=False, max_displacement=10, reward_size=2)
            
            hx = torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5
            for trial in range(trials):

                next_obs, done = env.reset()
                norm_next_obs = env.normalize_states(next_obs)
                next_state = np.concatenate([norm_next_obs, env.context, np.array([0.0])])
                next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

                hx = hx.detach()
                # if trial_counter % reset_memory == 0:
                # if np.random.random_sample()< reset_memory:
                #     hx += (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5)

                while not done:

                    if np.random.random_sample()< reset_memory:
                        hx = (torch.randn(1, 1, hidden_dim) * 1/hidden_dim**0.5)

                    actor_logits, critic_value, hx = model(next_state, hx)
                    probs = Categorical(logits=actor_logits)
                    action = probs.sample()

                    # Take action and observe reward
                    next_obs, reward, done = env.step(action.item())

                    # Prep next state
                    norm_next_obs = env.normalize_states(next_obs)
                    next_state = np.concatenate([norm_next_obs, env.context, np.array([reward])])
                    next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)

            all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])

    areas, pess, lrss = get_lrs(all_states)
    return np.array(areas)

def get_mean_ci(x, valididx):
    m = []
    s = []
    numparams = x.shape[0]
    for p in range(numparams):
        idx = int(valididx[p])
        m.append(np.mean(x[p,:idx],axis=0))
        s.append(np.std(x[p,:idx],axis=0)/np.sqrt(idx))
    m = np.array(m)
    s = np.array(s)
    return m, s

def plot_param_area(param, areas, xlabel, validms, logx=False, legend=False):

    saveload(f'./analysis/{xlabel}_area',[param, areas], 'save')

    labels = ['CP', 'OB']
    colors= ['orange', 'brown']

    plt.figure(figsize=(3,2.5))
    for c in range(2):
        m,s = get_mean_ci(areas[:,:,c],validms)

        plt.plot(param, m, label=labels[c], color=colors[c])
        plt.fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color=colors[c])

    dfarea = areas[:,:,0] - areas[:,:,1]
    m,s = get_mean_ci(dfarea,validms)
    e = areas.shape[0]
    plt.plot(param, m, label='CP-OB', color='k', linewidth=2)
    plt.fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color='k')
    plt.xlabel(xlabel)
    if legend:
        plt.legend()
    plt.ylabel('$A$')
    if logx:
        plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'./analysis/{xlabel}_area_{e}e.png')
    plt.savefig(f'./analysis/{xlabel}_area_{e}e.svg')




analysis = args.analysis
epochs = 3
seeds = 50

data_dir = "./model_params_2/"
V = '5'
bias = False


if analysis == 'gamma':
    # influence of gamma

    gammas = [0.999, 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    areas = np.zeros([len(gammas), seeds, 2])
    validms = np.zeros(len(gammas), dtype=int)
    for g, gamma in enumerate(gammas):
        
        file_names= data_dir+f"*_V{V}_{gamma}g_0.0rm_50bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
        # file_names= data_dir+f"*_V3_{gamma}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(gamma, len(models))
        validms[g] = len(models)

        for m,model in enumerate(models):
            
            areas[g, m] = get_area(model, epochs=epochs)

    plot_param_area(gammas, areas, '$\gamma$', validms)



if analysis == 'rollout':
    # influence of rollout
    rollouts = [5, 10,20, 30, 40, 50, 75, 100, 150, 200] # 0.99,0.95, 0.9,0.8,0.7, 0.5, 0.25, 0.1
    areas = np.zeros([len(rollouts), seeds, 2])
    validms = np.zeros(len(rollouts), dtype=int)
    for g, rollout in enumerate(rollouts):

        file_names= data_dir+f"*_V{V}_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
        # file_names= data_dir+f"*_V3_0.95g_0.0rm_{rollout}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
        models = glob.glob(file_names)
        print(rollout, len(models))
        validms[g] = len(models)

        for m,model in enumerate(models):
            
            areas[g, m] = get_area(model, epochs=epochs)

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
            areas[g, m] = get_area(model, epochs=epochs, reset_memory=preset)

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
            areas[g, m] = get_area(model, epochs=epochs)


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
            areas[g, m] = get_area(model, epochs=epochs)


    plot_param_area(scales, areas, '$\\beta_{\delta}$',validms)


