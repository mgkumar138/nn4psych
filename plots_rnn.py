'''
This file contains the code for plotting the results of the RNN model.
'''
    import model_rnn
    import config
    import utils_data

def plot_combined_state_space(Hs, Rs, Os):
    '''
    -Takes in RNN weights and returns a plot of the state space
    '''
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    contexts = ["Change-point","Oddball"]
    plt.figure(figsize=(4, 8))
    
    for i, (h_list, r_list, o_list, context) in enumerate(zip(Hs, Rs, Os, contexts)):
        # Convert lists to numpy arrays
        h_array = torch.stack(h_list).detach().numpy()
        r_array = np.array(r_list)
        o_array = np.array(o_list)

        # Perform PCA for dimensionality reduction to 2D
        pca = PCA(n_components=2)
        h_proj = pca.fit_transform(h_array)
        pc1, pc2 = pca.explained_variance_ratio_

        # Calculate differences between consecutive hidden states for vector field
        vectors = h_proj[1:] - h_proj[:-1]

        # Determine marker colors for reward and hazards
        colors = np.where(r_array > 0, 'green', 'gray') # Default colors based on reward
        # Overlay red color where hazard is indicated (o_array > 0)
        colors[o_array > 0] = 'red'

        # Determine marker sizes based on reward
        sizes = 20# + ((r_array - 0) / (1 - 0)) * (200 - 20)  # Scale sizes from 20 to 200

        # Plot PCA projection with vector field
        plt.subplot(len(contexts),1, i+1)
        
        # Draw paths with a quiver plot
        plt.quiver(h_proj[:-1, 0], h_proj[:-1, 1],
                   vectors[:, 0], vectors[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7)
        
        # Scatter plot the hidden states with color coding
        plt.scatter(h_proj[:, 0], h_proj[:, 1], c=colors, s=sizes, alpha=1)
        
        # Highlight start and end points
        plt.scatter(h_proj[0, 0], h_proj[0, 1], color='blue', s=100, alpha=1, marker='s', label='Start Point')
        plt.scatter(h_proj[-1, 0], h_proj[-1, 1], color='blue', s=100, alpha=1, marker='x', label='End Point')
        
        plt.scatter([],[], label=context, color='r')
        plt.scatter([],[], label='Bag Drop', color='g')
        plt.scatter([],[], label='Bucket Mvmt', color='gray')

        plt.title(context)
        plt.suptitle(f'$\gamma={gamma}, \\beta_\delta={tds}, p_{{reset}}={prm}, t_{{mem}}={troll}$')
        plt.title(f"{context}")
        plt.xlabel(f'PC 1 (Var={pc1:.3f})')
        plt.ylabel(f'PC 2, (Var={pc2:.3f})')
        plt.grid(True)
        if i>0:
            plt.legend()

    plt.tight_layout()
    plt.show()

def plot_combined_state_space_v2(Hs, Rs, Os, contexts):
    #from analyze_rnn_fp 
    plt.figure(figsize=(12, 6))

    for i, (h_list, r_list, o_list, context) in enumerate(zip(Hs, Rs, Os, contexts)):
        # Convert lists to numpy arrays
        h_array = torch.stack(h_list).detach().numpy()
        r_array = np.array(r_list)
        o_array = np.array(o_list)

        # Perform PCA for dimensionality reduction to 2D
        pca = PCA(n_components=2)
        h_proj = pca.fit_transform(h_array)

        # Calculate differences between consecutive hidden states for vector field
        vectors = h_proj[1:] - h_proj[:-1]

        # Determine marker colors for reward and hazards
        colors = np.where(r_array > 0, 'green', 'gray')  # Default colors based on reward
        # Overlay red color where hazard is indicated (o_array > 0)
        colors[o_array > 0] = 'red'

        # Determine marker sizes based on reward
        sizes = 20  # + ((r_array - 0) / (1 - 0)) * (200 - 20)  # Scale sizes from 20 to 200

        # Plot PCA projection with vector field
        plt.subplot(1, len(contexts), i + 1)

        # Draw paths with a quiver plot
        plt.quiver(h_proj[:-1, 0], h_proj[:-1, 1],
                   vectors[:, 0], vectors[:, 1],
                   angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.7)

        # Scatter plot the hidden states with color coding
        plt.scatter(h_proj[:, 0], h_proj[:, 1], c=colors, s=sizes, alpha=1)

        # Highlight start and end points
        plt.scatter(h_proj[0, 0], h_proj[0, 1], color='blue', s=100, alpha=1, marker='s', label='Start Point')
        plt.scatter(h_proj[-1, 0], h_proj[-1, 1], color='blue', s=100, alpha=1, marker='x', label='End Point')

        plt.scatter([], [], label=context, color='r')
        plt.scatter([], [], label='Bag Drop', color='g')
        plt.scatter([], [], label='Bucket Mvmt', color='gray')

        plt.title(f'Combined State Space - {context}')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

def calc_rnn_significant_units():

    '''in progress'''
    
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    from jax import random, vmap

    def calculate_volatility(window_size=50, gammaPlot = .5):
        '''
        Calculate volatility over time based on prediction errors from the history of rewards.
        Parameters:
        window_size: size of the window for calculating volatility
        gammaPlot: discount factor for future rewards
        '''
        history = np.load(r'C:\Users\aman0087\Documents\Github\nn4psych\data\history_contextual.npy')   
        prediction_errors = []

        for t in range(len(history) - 1):
            reward_t = history[t][0]
            reward_t_plus_1 = history[t + 1][0]
            prediction_error = reward_t + gammaPlot * (reward_t_plus_1 - reward_t)
            prediction_errors.append(prediction_error)

        volatility_over_time = np.array([
            np.var(prediction_errors[i:i + window_size]) 
            for i in range(len(prediction_errors) - window_size + 1)
        ])
        #hardfix because last 50 doesn't have a volatility
        random_values = np.random.rand(window_size)
        volatility_over_time = np.concatenate((volatility_over_time, random_values))
        
        return volatility_over_time

    
    def custom_linregress(x, y):
        n = x.size
        x_mean = jnp.mean(x)
        y_mean = jnp.mean(y)
        xy_cov = jnp.mean(x * y) - x_mean * y_mean
        xx_cov = jnp.mean(x * x) - x_mean * x_mean

        slope = xy_cov / xx_cov
        intercept = y_mean - slope * x_mean
        r_value = xy_cov / jnp.sqrt(xx_cov * (jnp.mean(y * y) - y_mean * y_mean))
        p_value = 2 * (1 - jax.scipy.stats.norm.cdf(jnp.abs(r_value) * jnp.sqrt((n - 2) / (1 - r_value ** 2))))
        std_err = jnp.sqrt((jnp.mean((y - (slope * x + intercept)) ** 2)) / (n - 2))

        return slope, intercept, r_value, p_value, std_err

    def perform_regression(timepoint_volatility, unit_activation, threshold):
        _, _, _, p_value, _ = custom_linregress(timepoint_volatility, unit_activation)
        return p_value < threshold
   
    # Example LSTM activations (400 episodes, 50 epochs, 64 hidden units)
    activations = np.load(r'C:\Users\aman0087\Documents\Github\nn4psych\data\activity_contextual.npy')
    activations = activations.reshape(400 * 50, 64)
    proportions = []
    significance_threshold = .05
        
    volatility_over_time = calculate_volatility(activations)

    vectorized_regression = vmap(perform_regression, in_axes=(None, 0, None))

    # Perform regression for each time point
    for t in range(activations.shape[0]):
        timepoint_volatility = volatility_over_time[t]
        threshold = significance_threshold / activations.shape[1]

        # Apply the vectorized regression function to all hidden units at this time point
        significant_units = jnp.sum(vectorized_regression(timepoint_volatility, activations[t], threshold))
        proportions.append(significant_units / activations.shape[1])

    # Convert proportions to a NumPy array before saving
    proportions_np = np.array(proportions)
    np.save('data/proportions.npy', proportions_np)

def plot_rnn_significant_units():
    '''
    Plot the proportion of significant hidden units over time.
    -in progress
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(proportions_np)), proportions_np, label='Proportion of Significant Hidden Units', color='red')
    plt.xlabel('Time')
    plt.ylabel('Proportion of Significant Units')
    plt.title('Proportion of Significant Hidden Units Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('plots/proportion_significant_units.png')

def run_all_plots(rnn_act_dict, model_path):
    '''
    hold
    '''

  

def plot_combined_state_space_from_dict(rnn_act_dict, hp_to_use, model_path):
    """
    Extract Hs, Rs, Os for a specific model_path from rnn_act_dict and plot using plot_combined_state_space.
    Args:
        rnn_act_dict: dict, output from utils_data.get_rnn_activity
        hp_to_use: str, hyperparameter name (e.g. 'gamma')
        model_path: str, path to the model to plot
    """
    model_list = rnn_act_dict[hp_to_use]['model_list']
    try:
        model_idx = model_list.index(model_path)
    except ValueError:
        raise ValueError(f"Model path {model_path} not found in model_list for {hp_to_use}.")
    Hs = rnn_act_dict[hp_to_use]['Hs'][model_idx]
    Rs = rnn_act_dict[hp_to_use]['Rs'][model_idx]
    Os = rnn_act_dict[hp_to_use]['Os'][model_idx]
    plot_combined_state_space(Hs, Rs, Os)