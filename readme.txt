

Code outline

Main - no input into the RNN (just to make sure the structure is working)
Context- can put in reward, action, context input & integrate these together
RNN helicopter - all of above and synced to continuous helicopter task


analysis_significant_rnn_units.py - analysis of rnn unit activations (incomplete)
analyze_compiled.py - compiles and plots data from best run
analyze_hyperparams_{rnn_param}.py - plots areas for each param
analyze_normative.py - bayesian model sim and plotting
analyze_rnn_fp.py - uses FixedPointFinder and plots combined state space
analyze_rnn.py - plots combined state space 
bayesian_models.py - contains (incomplete) PYMC model setup. also setup to simulate model predictions given priors
behav_figures.py - various behav figures for individual and batch_data
code_figures_behaviour.py - outdated behav fig
compile.py - compiles the state data from multiple .npz files into a single .pickle file.
fixed_point_analysis.py - calcs for fixed points and PCA to plot into 2d space
get_behavior.py - runs trained models through task to reproduce and save behavior
nassarfig6.py - reproduces behavioural data from nassar2021
pretrain_rnn_with_heli_v* - pretrain RNN
pyem_models.py - sets up bayesian model for fitting
tasks.py - holds the discrete / continuous helicopter environment. 
train_rnn_without_heli_server.py - pretrain w/out RNN 
utils.py - calcs for PYMC model setup, unpickle state vector, filter to exclude models based on performance
utils.funcs.py - holds generic actor critic model, lr and plotting functions (out of date?)


cleanup to-do's
    Various function rewritten multiple times: 
        -move get_lrs_v2, get_lrs into utils.py
        -move get_area into rnn_utils.py
        -put plot_states, plot_lrs.py into behav_figures.py
    -move analyze_normative.py into behav_figures.py
    -pull actorCritic module from rnn_utils.py instead of rewriting in analyze_rnn_fp.py and analyze_rnn
    -analyze_rnn.py outdated / replaced with analyze_rnn_fp.py? 
    -separate individual vs batch figures in behav_figures.py or break up to two files
    -move code_figures_behaviour.py into behav_figures
    -when / where is compile.py used? 
    -move previous pretrain_rnn_with_heli versions to defunct folder? 

Workflow 

1) setup RNN with ..
2) train RNN with ....
4) filter good models with ...
5) analyze behav with ...
    -or fp with ...

Main things used for analysis- 

'state vector' - 
all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])

all RNN hyper parameters - 

    gammas  = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1]
    rollouts = [5, 10, 20, 30, 50, 75, 100, 150, 200] #skipping 40 
    presets = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    scales  = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

Trained model weight paths - 
    -number at beginning of string indicates the ΔArea (on last epoch?) (proxy for performance)
    -'val' = 
    "gamma": (
        gammas,
        "*_V3_{val}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
    ),
    "rollout": (
        rollouts,
        "*_V3_0.95g_0.0rm_{val}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
    ),
    "preset": (
        presets,
        "*_V3_0.95g_{val}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
    ),
    "scale": (
        scales,
        "*_V3_0.95g_0.0rm_100bz_0.0td_{val}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
