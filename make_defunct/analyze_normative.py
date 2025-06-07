#%%
import matplotlib.pyplot as plt
import numpy as np
from model_bayesian import BayesianModel

def plot_states(states):

    for c, context in enumerate(contexts):
        [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers] = states[c]

        plt.figure(figsize=(6, 3))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.scatter(trials, bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=1, edgecolors='k')
        plt.plot(trials, helicopter_positions, label='Helicopter', color='green', linewidth=4)
        plt.plot(trials, bucket_positions, label='Bucket Position', color='orange', alpha=1, linewidth=2)

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {context}")
        plt.legend(frameon=True)
        plt.tight_layout()

states = np.load('./data/env_data_change-point.npy') #[trials, bucket_position, bag_position, helicopter_position]


contexts = ["changepoint","oddball"]
all_states = []
for c, context in enumerate(contexts):
    model = BayesianModel(states, model_type = context)
    states = model.sim_data(total_trials=200, model_name = "flexible_normative_model", condition = context)
    all_states.append(states)


plot_states(all_states)