#%%
#Load all data 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
import scipy.stats as stats

#v1 plots - largely out of usage

def plot_update_by_prediction_error(prediction_error, update, condition="change-point"):
    # Plot update x pe
    plt.figure(figsize=(3, 2))
    plt.scatter(prediction_error, update, alpha=0.5, color='blue', label='Data Points')

    slope, intercept, r_value, p_value, std_err = linregress(prediction_error, update)
    plt.plot(prediction_error, slope * prediction_error + intercept, color='k', label=f'm={slope:.3f}, c={intercept:.2f},r={r_value:.3f}, p={p_value:.3f}')

    sorted_indices = np.argsort(prediction_error)
    prediction_error_sorted = prediction_error[sorted_indices]
    update_sorted = update[sorted_indices]

    window_size = 10
    smoothed_update = uniform_filter1d(update_sorted, size=window_size)
    plt.plot(prediction_error_sorted, smoothed_update, color='red', label='Smooth')

    plt.xlabel('Prediction Error')
    plt.ylabel('Update (Predicted State at t+1 - State at t)')
    plt.title(f'{condition}: Update by prediction error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the plot
    plt.savefig(f'plots/update_by_prediction_error_{condition}.png')

def plot_learning_rate_by_prediction_error(prediction_error, learning_rate, condition="change-point"):
    # Plot learning rate by prediction error
    plt.figure(figsize=(3, 2))
    plt.scatter(prediction_error, learning_rate, alpha=0.5, color='green', label='Data Points')
    
    slope, intercept, r_value, p_value, std_err = linregress(prediction_error, learning_rate)
    # Plot the regression line
    plt.plot(prediction_error, slope * prediction_error + intercept, color='orange', label=f'm={slope:.3f}, c={intercept:.2f},r={r_value:.3f}, p={p_value:.3f}')

    sorted_indices = np.argsort(prediction_error)
    prediction_error_sorted = prediction_error[sorted_indices]
    learning_rate_sorted = learning_rate[sorted_indices]

    window_size = 10
    smoothed_learning_rate = uniform_filter1d(learning_rate_sorted, size=window_size)
    plt.plot(prediction_error_sorted, smoothed_learning_rate, color='red', label='Smooth')

    plt.xlabel('Prediction Error')
    plt.ylabel('Learning Rate')
    plt.title(f'{condition}: Learning Rate by Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Save the plot
    plt.savefig(f'plots/learning_rate_by_prediction_error_{condition}.png')

def plot_states_and_learning_rate(true_state, predicted_state, learning_rate, condition="change-point"):
    trials = np.arange(len(true_state))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Trial')
    ax1.set_ylabel('State', color='blue')
    ax1.plot(trials, true_state, label='True State', color='blue')
    ax1.plot(trials, predicted_state, label='Predicted State', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='green')
    ax2.plot(trials[:-1], learning_rate, label='Learning Rate', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title(f'{condition}: True State, Predicted State, and Learning Rate over Trials')
    plt.grid(True)
    plt.show()
    plt.savefig(f'plots/states_and_learning_rate_over_trials_{condition}.png')

def plot_learning_rate_histogram(learning_rate, condition="change-point"):
    bins = np.arange(0, 1.1, 0.1)
    plt.figure(figsize=(10, 6))
    plt.hist(learning_rate, bins=bins, edgecolor='black')
    plt.xlabel('Learning Rate')
    plt.ylabel('Frequency')
    plt.title(f'{condition}: Histogram of Learning Rate')
    plt.grid(True)
    plt.show()
    plt.savefig(f'plots/learning_rate_histogram_{condition}.png')

def plot_lr_after_hazard(learning_rate, hazard_distance, condition="change-point"):

    bins = np.arange(0, 1.1, 0.1)
    bin_indices = np.digitize(learning_rate, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    update_size_indices = np.where(bin_indices == 0, 0, np.where((bin_indices >= 1) & (bin_indices <= 8), 1, 2))
    # Count intersections between hazard_distance and update_size_indices
    interaction_counts = {
        "small": {},
        "medium": {},
        "large": {}
    }

    for hd, usi in zip(hazard_distance, update_size_indices):
        if usi == 0:
            category = "small"
        elif usi == 1:
            category = "medium"
        else:
            category = "large"
        
        if hd in interaction_counts[category]:
            interaction_counts[category][hd] += 1
        else:
            interaction_counts[category][hd] = 1

    # Prepare data for plotting
    categories = ["small", "medium", "large"]
    colors = ["blue", "green", "red"]
    x = sorted({hd for counts in interaction_counts.values() for hd in counts.keys()})

    plt.figure(figsize=(10, 6))

    for category, color in zip(categories, colors):
        counts = [interaction_counts[category].get(hd, 0) for hd in x]
        plt.plot(x, counts, label=category.capitalize(), color=color)

    plt.xlabel('Hazard Distance')
    plt.ylabel('Count')
    plt.title(f'{condition}: Interactions by Hazard Distance and Size')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'plots/interactions_line_graph_{condition}.png')

#% v2 plots - taken from other scripts, mostly single epoch data

def plot_states(states):
    contexts = ["Change-point","Oddball"]
    for c, context in enumerate(contexts):
        [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers] = states[c]

        plt.figure(figsize=(4, 2.5))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.scatter(trials, bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=1, edgecolors='k')
        plt.plot(trials, helicopter_positions, label='Helicopter', color='green', linewidth=3)
        plt.plot(trials, bucket_positions, label='Bucket Position', color='orange', alpha=1, linewidth=3)

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"{context}\n$\gamma={gamma}, \\beta_\delta={tds}, p_{{reset}}={prm}, t_{{rollout}}={troll}$")
        plt.legend(frameon=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'./analysis/{context}_states.png')
        plt.savefig(f'./analysis/{context}_states.svg')

def plot_lrs(states, scale=0.1):
    epochs = states.shape[0]
    pess, lrss, area = [], [], []
    for c in range(2):
        pes, lrs = [], []
        for e in range(epochs):
            pe, lr = utils_calcs.get_lrs_v2(states[e, c])

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

    plt.figure(figsize=(3, 2))
    colors = ['orange', 'brown']
    labels = ['CP', 'OB']
    for i in range(2):
        window_size = int(len(lrss[i]) * scale)
        smoothed_learning_rate = uniform_filter1d(lrss[i], size=window_size)
        plt.plot(pess[i], smoothed_learning_rate, color=colors[i], linewidth=2, label=labels[i])
    plt.legend()
    plt.xlabel('Prediction error')
    plt.ylabel('Learning rate')
    plt.title(f'CB={area[0]:.1f}, OB={area[1]:.1f}, A={(area[0] - area[1]):.1f}')
    plt.tight_layout()
    return pess, lrss, area

def plot_behavior(states, context,epoch, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
    trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers = states
    # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
    plt.plot(trials, bag_positions, label='Bag', color='red', marker='o', linestyle='-.', alpha=0.5, ms=2)
    plt.plot(trials, helicopter_positions, label='Heli', color='green', linestyle='--',ms=2)
    plt.plot(trials, bucket_positions, label='Bucket', color='b',marker='o', linestyle='-.', alpha=0.5,ms=2)

    plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
    plt.xlabel('Trial')
    plt.ylabel('Position')
    plt.title(f"{context}, E:{epoch}")
    plt.legend(fontsize=6)

#% v3 plots - modified to use averaged batch_data

def plot_lrs_v3_batch(behav_dict, scale=0.1):
    """
    Modified to include error bars and confidence intervals for averaged data across all runs.
    """
    for rnn_param, data in behav_dict.items():
        model_groups = data['model_list']
        # Group runs by parameter value (from the model_groups values)
        group_dict = {}
        for key, val in model_groups.items():
            param_val = next((x for x in val if isinstance(x, (int, float))), None)
            if param_val is None:
                continue
            # use the first element of the key as the run index
            run_id = key[0]
            group_dict.setdefault(param_val, []).append(run_id)

        unique_param_vals = sorted(group_dict.keys())
        n_plots = len(unique_param_vals)
        fig, axs = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
        if n_plots == 1:
            axs = [axs]

        for ax, param_val in zip(axs, unique_param_vals):
            runs = group_dict[param_val]

            # Aggregate data across all runs for CP and OB
            pe_cp_all = np.concatenate([data['pe_sorted_cp'][run] for run in runs])
            lr_cp_all = np.concatenate([data['lr_sorted_cp'][run] for run in runs])
            pe_ob_all = np.concatenate([data['pe_sorted_ob'][run] for run in runs])
            lr_ob_all = np.concatenate([data['lr_sorted_ob'][run] for run in runs])

            # Sort and smooth the data for CP
            sorted_indices_cp = np.argsort(pe_cp_all)
            pe_cp_sorted = pe_cp_all[sorted_indices_cp]
            lr_cp_sorted = lr_cp_all[sorted_indices_cp]
            window_size_cp = max(1, int(len(lr_cp_sorted) * scale))
            smoothed_lr_cp = uniform_filter1d(lr_cp_sorted, size=window_size_cp)

            # Sort and smooth the data for OB
            sorted_indices_ob = np.argsort(pe_ob_all)
            pe_ob_sorted = pe_ob_all[sorted_indices_ob]
            lr_ob_sorted = lr_ob_all[sorted_indices_ob]
            window_size_ob = max(1, int(len(lr_ob_sorted) * scale))
            smoothed_lr_ob = uniform_filter1d(lr_ob_sorted, size=window_size_ob)

            # Calculate confidence intervals for CP and OB
            ci_cp = stats.sem(lr_cp_sorted) * stats.t.ppf((1 + 0.95) / 2., len(lr_cp_sorted) - 1)
            ci_ob = stats.sem(lr_ob_sorted) * stats.t.ppf((1 + 0.95) / 2., len(lr_ob_sorted) - 1)

            # Plot CP data (solid) with error bars
            ax.errorbar(pe_cp_sorted, smoothed_lr_cp, yerr=ci_cp, fmt='-', color='blue', linewidth=2, label='CP')

            # Plot OB data (dashed) with error bars
            ax.errorbar(pe_ob_sorted, smoothed_lr_ob, yerr=ci_ob, fmt='--', color='green', linewidth=2, label='OB')

            # Calculate and display average areas
            avg_area_cp = np.trapz(smoothed_lr_cp, pe_cp_sorted)
            avg_area_ob = np.trapz(smoothed_lr_ob, pe_ob_sorted)
            avg_area_diff = avg_area_cp - avg_area_ob
            ax.set_title(f"{rnn_param} = {param_val}, avg area CP={avg_area_cp:.2f}, OB={avg_area_ob:.2f}, diff={avg_area_diff:.2f}")
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Learning Rate')
            ax.grid(True)
            ax.legend()

        fig.tight_layout()
        plt.show()
    
def plot_lr_bins_post_hazard_batch(behav_dict):
    """
    nassar2021 fig3a-d (modified)
    For each RNN parameter, group runs by the unique parameter value (extracted from model_list),
    then create one figure with two subplots per unique value (one for CP and one for OB)
    showing histograms of learning rate frequencies (bin size = 0.1) using at most num_runs runs.
    """
    bins = np.arange(0, 1.1, 0.1)

    for rnn_param, data in behav_dict.items():
        # Group runs by unique parameter value
        group_dict = {}
        for key, val in data['model_list'].items():
            param_val = next((x for x in val if isinstance(x, (int, float))), None)
            if param_val is None:
                continue
            run_id = key[0]
            group_dict.setdefault(param_val, []).append(run_id)

        unique_param_vals = sorted(group_dict.keys())
        n_plots = len(unique_param_vals)
        # Create subplots with 2 columns: one for CP and one for OB
        fig, axs = plt.subplots(n_plots, 2, figsize=(16, 4 * n_plots))
        if n_plots == 1:
            axs = axs.reshape(1, -1)

        for i, param_val in enumerate(unique_param_vals):
            runs = group_dict[param_val]
            # Aggregate learning rates for CP and OB across all runs
            cp_lr_all = np.concatenate([data['lr_unsorted_cp'][run] for run in runs])
            ob_lr_all = np.concatenate([data['lr_unsorted_ob'][run] for run in runs])

            # Compute histogram probabilities for CP and OB
            cp_hist, _ = np.histogram(cp_lr_all, bins=bins, density=True)
            ob_hist, _ = np.histogram(ob_lr_all, bins=bins, density=True)

            # Left subplot: CP histogram
            ax_cp = axs[i, 0]
            ax_cp.bar(bins[:-1], cp_hist, width=0.1, alpha=0.7, label='CP', edgecolor='black')
            ax_cp.set_title(f'{rnn_param} = {param_val} CP')
            ax_cp.set_xlabel('Learning Rate')
            ax_cp.set_ylabel('Probability')
            ax_cp.legend()
            ax_cp.grid(True)

            # Right subplot: OB histogram
            ax_ob = axs[i, 1]
            ax_ob.bar(bins[:-1], ob_hist, width=0.1, alpha=0.7, label='OB', edgecolor='black')
            ax_ob.set_title(f'{rnn_param} = {param_val} OB')
            ax_ob.set_xlabel('Learning Rate')
            ax_ob.set_ylabel('Probability')
            ax_ob.legend()
            ax_ob.grid(True)

        fig.suptitle(f'{rnn_param}: Learning Rate Histogram (CP and OB) Averaged Across Runs', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def compute_hazard_distance(hazard_triggers, until_next=False):
    """
    Compute hazard distance from hazard trigger array.
    
    When until_next is False (default):
      For each trial, the hazard distance is reset to 0 when a hazard is triggered (value==1),
      and otherwise is incremented by 1.
      
    When until_next is True:
      The returned hazard distance represents the number of trials until the next trigger.
      
    returns: hazard distance array; same size (and dtype) as hazard_triggers.
    """
    n = len(hazard_triggers)
    hd = np.zeros(n, dtype=int)
    
    if not until_next:
        current = 0
        for i, trigger in enumerate(hazard_triggers):
            if trigger == 1:
                current = 0
            hd[i] = current
            current += 1
    else:
        # Compute distance until the next hazard trigger by iterating backwards.
        current = None
        for i in range(n - 1, -1, -1):
            if hazard_triggers[i] == 1:
                current = 0
                hd[i] = 0
            else:
                if current is None:
                    # No hazard ahead: count distance to the end.
                    current = n - 1 - i
                else:
                    current += 1
                hd[i] = current
                
    return hd

def plot_lr_curve_post_hazard(behav_data):
    """
    Modified to average data across all runs for each unique parameter value.
    """
    for rnn_param, data in behav_data.items():
        # Group runs by unique parameter value using the model_list
        group_dict = {}
        for key, val in data['model_list'].items():
            param_val = next((x for x in val if isinstance(x, (int, float))), None)
            if param_val is None:
                continue
            run_id = key[0]
            group_dict.setdefault(param_val, []).append(run_id)

        unique_param_vals = sorted(group_dict.keys())
        for param_val in unique_param_vals:
            runs = group_dict[param_val]

            # Aggregate hazard distances and learning rates for CP and OB across all runs
            cp_hd_all = np.concatenate([compute_hazard_distance(data['cp_array'][run][4]) for run in runs])
            ob_hd_all = np.concatenate([compute_hazard_distance(data['ob_array'][run][4]) for run in runs])
            cp_lr_all = np.concatenate([data['lr_unsorted_cp'][run] for run in runs])
            ob_lr_all = np.concatenate([data['lr_unsorted_ob'][run] for run in runs])

            categories = {
                'Non-updates': lambda lr: lr < 0.1,
                'Moderate updates': lambda lr: (lr >= 0.1) & (lr < 0.9),
                'Large updates': lambda lr: lr >= 0.9
            }

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharey=True)
            fig.suptitle(f'{rnn_param} = {param_val}: LR Post-Hazard Analysis (Averaged)', fontsize=16)

            for row_idx, (cat_label, condition) in enumerate(categories.items()):
                cp_inds = np.where(condition(cp_lr_all))[0]
                ob_inds = np.where(condition(ob_lr_all))[0]

                cp_hd_filtered = cp_hd_all[cp_inds]
                ob_hd_filtered = ob_hd_all[ob_inds]

                cp_counts = np.bincount(cp_hd_filtered, minlength=max(cp_hd_all) + 1)
                ob_counts = np.bincount(ob_hd_filtered, minlength=max(ob_hd_all) + 1)

                cp_probs = cp_counts / cp_counts.sum() if cp_counts.sum() > 0 else cp_counts
                ob_probs = ob_counts / ob_counts.sum() if ob_counts.sum() > 0 else ob_counts

                ax_cp = axs[row_idx, 0]
                ax_cp.bar(range(len(cp_probs)), cp_probs, color='blue', alpha=0.7)
                ax_cp.set_title(f'CP - {cat_label}')
                ax_cp.set_xlabel('Hazard Distance')
                ax_cp.set_ylabel('Probability')
                ax_cp.grid(True)

                ax_ob = axs[row_idx, 1]
                ax_ob.bar(range(len(ob_probs)), ob_probs, color='green', alpha=0.7)
                ax_ob.set_title(f'OB - {cat_label}')
                ax_ob.set_xlabel('Hazard Distance')
                ax_ob.set_ylabel('Probability')
                ax_ob.grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

def plot_update_ratio(behav_data):
    """
    nassar2021 fig3 k/l
    Plots p(moderate update) (x-axis) against p(non-update) (y-axis).
    Generate 4 x 2 subplots (for each RNN parameter and each condition: CP and OB).
    One dot per hyperparameter value, colored by the hyperparameter range.
    """
    # Collect all RNN parameter names
    rnn_params = list(behav_data.keys())
    n_params = len(rnn_params)
    
    fig, axs = plt.subplots(n_params, 2, figsize=(10, 4 * n_params), sharex=True, sharey=True)
    if n_params == 1:
        axs = np.array([axs])
    
    # For each RNN parameter
    for row_idx, rnn_param in enumerate(rnn_params):
        data = behav_data[rnn_param]
        # Group runs by hyperparameter value from model_list (keys are assumed to contain run_id and param value)
        group_dict = {}
        for key, val in data['model_list'].items():
            # Extract the hyperparameter value (the first numeric value in the list)
            param_val = next((x for x in val if isinstance(x, (int, float))), None)
            if param_val is None:
                continue
            run_id = key[0]
            group_dict.setdefault(param_val, []).append(run_id)
        
        # Determine range of hyperparameter values for colormap normalization
        hp_vals = sorted(group_dict.keys())
        norm = plt.Normalize(min(hp_vals), max(hp_vals))
        cmap = plt.get_cmap("viridis")
        
        # For both CP and OB conditions
        for col_idx, cond in enumerate(["cp", "ob"]):
            ax = axs[row_idx, col_idx]
            for hp in hp_vals:
                runs = group_dict[hp]
                # Aggregate unsorted learning rates for the condition across runs
                lr_list = [data[f'lr_unsorted_{cond}'][run] for run in runs]
                lr_all = np.concatenate(lr_list)
                p_med, p_non, p_total = utils_calcs.compute_update_ratios(lr_all)
                ax.scatter(p_med, p_non, s=100, color=cmap(norm(hp)), label=f'{hp:.2f}')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('p(Moderate Update)')
            ax.set_ylabel('p(Non Update)')
            ax.set_title(f'{rnn_param.upper()} - {"CP" if cond=="cp" else "OB"}')
            ax.grid(True)
            # Show colorbar only once per row (if desired)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Hyperparameter Value')
    
    plt.tight_layout()
    plt.show()

def plot_all_update_ratios(behav_dict, hazard_distance_filter=None):
    '''
    3D plot of update ratios for all RNN parameters across conditions.
    Allows filtering of data based on a specific hazard distance.
    '''
    rnn_params = list(behav_dict.keys())
    n_params = len(rnn_params)
    fig, axs = plt.subplots(n_params, 2, figsize=(12, 6 * n_params), subplot_kw={'projection': '3d'})
    if n_params == 1:
        axs = np.array([axs])

    for row_idx, rnn_param in enumerate(rnn_params):
        data = behav_dict[rnn_param]
        # Group runs by hyperparameter value extracted from model_list
        group_dict = {}
        for key, val in data['model_list'].items():
            hp_val = next((x for x in val if isinstance(x, (int, float))), None)
            if hp_val is None:
                continue
            run_id = key[0]
            group_dict.setdefault(hp_val, []).append(run_id)

        hp_vals = sorted(group_dict.keys())
        norm = plt.Normalize(min(hp_vals), max(hp_vals))
        cmap = plt.get_cmap("viridis")

        # For CP and OB conditions
        for col_idx, cond in enumerate(["cp", "ob"]):
            ax = axs[row_idx, col_idx]
            # For each hyperparameter value, compute p(total update), p(moderate update), and p(non update)
            for hp in hp_vals:
                runs = group_dict[hp]
                lr_list = []
                hd = []
                for run in runs:
                    lr_list.append(data[f'lr_unsorted_{cond}'][run])
                    # Convert hazard flags to hazard distances
                    hd.append(compute_hazard_distance(data[f'{cond}_array'][run][4]))

                lr_all = np.concatenate(lr_list)
                hd_all = np.concatenate(hd)

                # Apply hazard distance filter if provided
                if hazard_distance_filter is not None:
                    mask = (hd_all >= hazard_distance_filter[0]) & (hd_all <= hazard_distance_filter[1])
                    lr_all = lr_all[mask]

                # Use compute_update_ratios to calculate proportions
                p_mod, p_non, p_total = utils_calcs.compute_update_ratios(lr_all)
                ax.scatter(p_total, p_mod, p_non, s=150, alpha=0.8, color=cmap(norm(hp)), label=f'{hp:.2f}')

            # Add colorbar for hyperparameter values
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.5)
            cbar.set_label('Hyperparameter Value')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_xlabel('p(Total Update)')
            ax.set_ylabel('p(Moderate Update)')
            ax.set_zlabel('p(Non Update)')
            title = f'{rnn_param.upper()} - {"CP" if cond == "cp" else "OB"}'
            if hazard_distance_filter is not None:
                title += f" (Hazard Distance = {hazard_distance_filter})"
            ax.set_title(title)

    fig.tight_layout()
    plt.show()

def plot_param_area(param, areas, xlabel, validms, logx=False, legend=False):

    utils_data.saveload(f'./analysis/{xlabel}_area',[param, areas], 'save')

    labels = ['CP', 'OB']
    colors= ['orange', 'brown']

    plt.figure(figsize=(3,2.5))
    for c in range(2):
        m,s = utils_calcs.get_mean_ci(areas[:,:,c],validms)

        plt.plot(param, m, label=labels[c], color=colors[c])
        plt.fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color=colors[c])

    dfarea = areas[:,:,0] - areas[:,:,1]
    m,s = utils_calcs.get_mean_ci(dfarea,validms)
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

def plot_param_area_v2(behav_dict):
    """
    Repeat plot_param_area using the behav_dict data structure.
    Create 4 subplots (one for each RNN parameter) to display area differences for CP and OB.
    """
    import matplotlib.pyplot as plt

    # Create a figure with 4 subplots (2 rows x 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Loop over each RNN parameter in the behavior dictionary with index.
    for idx, (rnn_param, data) in enumerate(behav_dict.items()):
        ax = axs[idx]
        model_list = data['model_list']

        # Group run indices by hyperparameter value
        groups = {}
        for key, param_list in model_list.items():
            hp_val = next((x for x in param_list if isinstance(x, (int, float))), None)
            if hp_val is None:
                continue
            run_id = key[0]
            groups.setdefault(hp_val, []).append(run_id)

        # Sort hyperparameter values for plotting
        hp_vals = np.array(sorted(groups.keys()))
        cp_means, cp_cis = [], []
        ob_means, ob_cis = [], []

        # For each unique hyperparameter value, compute the mean area and confidence interval
        for hp in hp_vals:
            runs = groups[hp]
            cp_areas = np.array([data['area_cp'][run] for run in runs])
            ob_areas = np.array([data['area_ob'][run] for run in runs])
            import scipy.stats as stats
            n_cp = len(cp_areas)
            m_cp = np.mean(cp_areas)
            sem_cp = np.std(cp_areas, ddof=1) / np.sqrt(n_cp) if n_cp > 1 else 0
            ci_cp = stats.t.ppf(0.975, df=n_cp-1) * sem_cp if n_cp > 1 else 0

            n_ob = len(ob_areas)
            m_ob = np.mean(ob_areas)
            sem_ob = np.std(ob_areas, ddof=1) / np.sqrt(n_ob) if n_ob > 1 else 0
            ci_ob = stats.t.ppf(0.975, df=n_ob-1) * sem_ob if n_ob > 1 else 0
            cp_means.append(m_cp)
            cp_cis.append(ci_cp)
            ob_means.append(m_ob)
            ob_cis.append(ci_ob)

        cp_means = np.array(cp_means)
        cp_cis = np.array(cp_cis)
        ob_means = np.array(ob_means)
        ob_cis = np.array(ob_cis)

        # Plot CP and OB area curves with confidence intervals on the current axis.
        ax.errorbar(hp_vals, cp_means, yerr=cp_cis, fmt='-o', color='orange', label='CP')
        ax.errorbar(hp_vals, ob_means, yerr=ob_cis, fmt='-o', color='brown', label='OB')

        # Compute difference between CP and OB and propagate error (assuming independence)
        diff = cp_means - ob_means
        diff_ci = np.sqrt(cp_cis**2 + ob_cis**2)
        ax.errorbar(hp_vals, diff, yerr=diff_ci, fmt='-o', color='k', linewidth=2, label='CP-OB')

        ax.set_xlabel(rnn_param)
        ax.set_ylabel('Area')
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.savefig('./analysis/all_params_area.png')
    plt.savefig('./analysis/all_params_area.svg')
    plt.show()

# %% Organize batch_data

import utils_calcs, utils_data
from scipy.ndimage import uniform_filter1d
import numpy as np
import utils_calcs
import matplotlib.pyplot as plt

def get_batch_behav(file_dir='data/rnn_behav/model_params_101000', RNN_param_list = ["gamma", "preset", "rollout", "scale"]):
    '''
    -takes in a list of RNN parameters to analyze made from get_behavior.py
    -returns a dictionary of the data for each parameter

    previously: 
        cp_array_'condition' = [state_vector - 5, trials - 200]
        state_vector = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
    new: 
        now going to be cp_array_'condition' = [epoch, state_vector, trials] [30,5,200]
        results = [rnn param, epoch] = {cp_array etc...}

    -get_lrs_v2 returns vector clipped by prediction error threshold

    '''
    results = {}
    for rnn_param in RNN_param_list:
        cp_array, ob_array, model_list = utils_data.unpickle_state_vector(file_dir = file_dir, RNN_param=rnn_param)


        #filter model_list here




        if len(cp_array[0]) == 5: # 5 state variables
            pe_sorted_cp, lr_sorted_cp, pe_unsorted_cp, lr_unsorted_cp, area_cp = zip(*[utils_calcs.get_lrs_v3(cp_array[i]) for i in range(len(model_list))])
            pe_sorted_ob, lr_sorted_ob, pe_unsorted_ob, lr_unsorted_ob, area_ob = zip(*[utils_calcs.get_lrs_v3(ob_array[i]) for i in range(len(model_list))])

            results[rnn_param] = {
                'cp_array': cp_array,
                'pe_sorted_cp': pe_sorted_cp,
                'lr_sorted_cp': lr_sorted_cp,
                'pe_unsorted_cp': pe_unsorted_cp,
                'lr_unsorted_cp': lr_unsorted_cp,
                'area_cp': area_cp,
                'ob_array': ob_array,
                'pe_sorted_ob': pe_sorted_ob,
                'lr_sorted_ob': lr_sorted_ob,
                'pe_unsorted_ob': pe_unsorted_ob,
                'lr_unsorted_ob': lr_unsorted_ob,
                'area_ob': area_ob,
                'model_list': model_list
            }
        elif len(cp_array[0]) == 30: #30 epochs 
            for epoch in range(len(cp_array[0])):
                pe_sorted_cp, lr_sorted_cp, pe_unsorted_cp, lr_unsorted_cp, area_cp = zip(*[get_lrs_v3(cp_array[i][epoch]) for i in range(len(model_list))])
                pe_sorted_ob, lr_sorted_ob, pe_unsorted_ob, lr_unsorted_ob, area_ob = zip(*[get_lrs_v3(ob_array[i][epoch]) for i in range(len(model_list))])

                results[rnn_param, epoch] = {
                    'cp_array': cp_array,
                    'pe_sorted_cp': pe_sorted_cp,
                    'lr_sorted_cp': lr_sorted_cp,
                    'pe_unsorted_cp': pe_unsorted_cp,
                    'lr_unsorted_cp': lr_unsorted_cp,
                    'area_cp': area_cp,
                    'ob_array': ob_array,
                    'pe_sorted_ob': pe_sorted_ob,
                    'lr_sorted_ob': lr_sorted_ob,
                    'pe_unsorted_ob': pe_unsorted_ob,
                    'lr_unsorted_ob': lr_unsorted_ob,
                    'area_ob': area_ob,
                    'model_list': model_list
                }

    return results

# behav_dict = get_batch_behav(file_dir='data/rnn_behav/model_params_101000/30_epochs')
behav_dict = get_batch_behav()

# plot_lrs_v3_batch(behav_dict, scale=0.1)
# plot_lr_bins_post_hazard_batch(behav_dict)
# plot_lr_curve_post_hazard(behav_dict)
# plot_update_ratio(behav_dict)
# plot_all_update_ratios(behav_dict)
# plot_all_update_ratios(behav_dict, hazard_distance_filter= [1,3])
plot_param_area_v2(behav_dict)


#%% V1 plots - out of date

# if __name__ == "__main__":
    #out of date
        #states = np.load('data/bayesian_models/model_predictions.npy') #[trials, bucket_position, bag_position, helicopter_position]
        #states = np.load('data/pt_rnn_context-point.npy') #[trials, bucket_position, bag_position, helicopter_position]
        #prediction_error, update, learning_rate, true_state, predicted_state,hazard_distance, hazard_trials = extract_states(states) #ERROR? Is this supposed to return slope


    # Call the functions to generate the plots
    # plot_update_by_prediction_error(prediction_error, update)
    # plot_learning_rate_by_prediction_error(prediction_error, learning_rate)
    # plot_states_and_learning_rate(true_state, predicted_state, learning_rate)
    # plot_learning_rate_histogram(learning_rate)
    # plot_lr_after_hazard(learning_rate, hazard_trials)

# %%
