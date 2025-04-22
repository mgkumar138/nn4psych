'''
This script compiles the state data from multiple .npz files into a single .pickle file.
takes the best performance per seed
make part of filtering process
'''

import numpy as np

def saveload(filename, variable, opt):
    import pickle
    if opt == 'save':
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variable, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            return pickle.load(file)
    
loadmodel = 1
max_displacement = 20
reward_size = 10
hidden_dim = 64
learning_rate = 0.0001
seeds = 50

epochs = 100
num_contexts = 2
trials = 200

states = np.zeros([seeds, epochs, num_contexts, 5, trials])
perf = np.zeros([seeds, epochs, num_contexts, 3, trials])

for seed in range(seeds):

    exptname = f"noheli_{loadmodel}pre_{max_displacement}md_{reward_size}rs_{hidden_dim}n_{learning_rate}lr_{seed}s"

    npz_file = np.load(f'./state_data/{exptname}.npz')

    states[seed] = npz_file['arr_0']
    epoch_G = npz_file['arr_1']
    epoch_loss = npz_file['arr_2']
    epoch_time = npz_file['arr_3']

    perf[seed] = np.concatenate([epoch_G[:,:,None,:], epoch_loss[:,:,None,:], epoch_time[:,:,None,:]],axis=2)

print(states.shape, perf.shape)

saveload(f'./data/{loadmodel}pre_{max_displacement}md_{reward_size}rs_{hidden_dim}n', [states, perf],'save')