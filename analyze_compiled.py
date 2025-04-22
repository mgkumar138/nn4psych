'''
compiles and plots data from best run
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import utils_data

def prep_metrics(data_dir = "./best_data/0.609849751837204_200t_5md_5rz_64n_0.95g_1e-05lr"):
    [epoch_perfs, epoch_states] = utils_data.saveload(data_dir, 1, 'load')
    trials = epoch_states.shape[-1]
    epochs = epoch_states.shape[0]
    train_epochs = int(epoch_states.shape[0]*0.8)
    test_epochs = epoch_states.shape[0] - train_epochs

    print(epoch_states.shape, epoch_perfs.shape)

    return epoch_states, epoch_perfs, trials, epochs, train_epochs, test_epochs


#%%
def plot_perfs(epoch_perfs, trials, epochs):
    # Plot performance for each trial and epoch
    plt.figure(figsize=(3,2*3))

    for j in range(3):
        plt.subplot(3,1,j+1)
        for i in range(trials):
            plt.errorbar(x=np.arange(epochs), y=np.mean(epoch_perfs[:,i,j],axis=1), yerr=np.std(epoch_perfs[:,i,j],axis=1)/np.sqrt(trials))

    plt.figure(figsize=(3,2*3))

    for j in range(3):
        plt.subplot(3,1,j+1)
        for i in range(trials):
            plt.plot(np.mean(epoch_perfs[:,i,j],axis=1), label=f'trial {i}')


# %%

def calc_metrics(epoch_states, epoch_perfs, train_epochs, test_epochs):
    # Calculate metrics for each epoch
    pes = np.zeros([test_epochs, 2, trials-1])
    lrs = np.zeros([test_epochs, 2, trials-1])
    scores = np.zeros(test_epochs)
    for te in range(test_epochs):

        for c in range(2):
            states = epoch_states[train_epochs + te, c]
            true_state = states[2]  # bag position
            predicted_state = states[1]  # bucket position
            pe = abs((true_state - predicted_state))[:-1]

            update = abs(np.diff(predicted_state))

            pes[te, c] = pe
            lrs[te, c] = np.where(pe != 0, update / pe, 0)

        scores[te] = np.mean(lrs[te, 0]) - np.mean(lrs[te, 1])

    slope, intercept, r_value, p_value, std_err = linregress(np.arange(test_epochs), scores)
    
    return pes, lrs, scores, (slope, intercept, r_value, p_value, std_err)


def plot_metrics(pes, lrs, scores, regression_params):    # Plot metrics
    slope, intercept, r_value, p_value, std_err = regression_params

    plt.figure()
    plt.plot(np.arange(test_epochs), scores)
    plt.plot(np.arange(test_epochs), slope*np.arange(test_epochs)+intercept, color='r')
    plt.plot(np.arange(test_epochs), np.mean(lrs[:, 0],axis=1))
    plt.plot(np.arange(test_epochs), np.mean(lrs[:, 1],axis=1))

    plt.figure()
    plt.scatter(pes[-1,0], lrs[-1,0], color='orange')
    plt.scatter(pes[-1,1], lrs[-1,1], color='brown')
