#%%
from task import Heli_Bag
import copy
import numpy as np
import matplotlib.pyplot as plt
from model import res_ac

# task
epochs = 1000
trials = 200
max_time = 300
train_cond = False # show helicopter?
tasks = ["change-point"]
alpha = 0.5

# model 
ninput = 4+2
nrnn = 1024
nact = 3
gamma = 0.95
clr = 0.001
alr = 0.00025
seed = 0
agent = res_ac(ninput=ninput, nrnn=nrnn, nact=nact, gamma=gamma, clr=clr, alr=alr, seed=seed)
init_params = agent.get_weights().copy()
bias_action = np.array([0,0,-1])

epoch_perf = np.zeros([epochs, len(tasks), trials, 3])

#train
for epoch in range(epochs):

    idxs = np.random.choice(np.arange(len(tasks)), size=len(tasks), replace=False)

    for task_type in idxs:

        env = Heli_Bag(condition=tasks[task_type],train_cond=train_cond)
        agent.reset()

        for trial in range(trials):

            totR = 0
            totloss = 0

            obs, done = env.reset()

            state = np.concatenate([obs, env.context])[:,None]
            r = agent.get_rnn(state)
            prev_v = agent.get_value(r)
            A = agent.get_action(r, bias_action)

            while not done:

                obs, reward, done = env.step(A)
                totR += reward
                # print(agent.aprob, A, env.bucketpos)

                state = np.concatenate([obs, env.context])[:,None]
                r = agent.get_rnn(state)
                v = agent.get_value(r)
                A = agent.get_action(r, bias_action)

                td = reward + gamma * v[0,0] - prev_v[0,0]
                totloss += td**2

                agent.learn(td)

                # print(state.max(), r.max(), reward, v.max(),prev_v.max(), agent.wcri.max())

                prev_v = v.copy()
                # state = next_state.copy()


            print(f'E {epoch}, T {trial}, t {env.time}, R {totR:.3f}, L {totloss:.3f}')

            epoch_perf[epoch, task_type, trial] = np.array([totR, totloss, env.time])
                

ylabels = ['G','Loss','Time']
plt.figure(figsize=(3,2*3))
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(np.arange(epochs), np.mean(epoch_perf[:,:,:,i],axis=2))
    plt.ylabel(ylabels[i])
plt.xlabel('Epoch')
plt.tight_layout()