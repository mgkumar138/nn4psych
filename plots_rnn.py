'''
This file contains the code for plotting the results of the RNN model.
'''

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

