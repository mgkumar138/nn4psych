#%%
'''
Fit Bayesian models to the RNN-AC simulated data
'''

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pytensor
import pytensor.tensor as pt
import pymc as pm
import scipy
import scipy.stats as stats
import utils
from tasks import PIE_CP_OB


#%% 
#minimize likelihood function to get MLE estimates of parameters (no fitting)

class BayesianModel: 
    def __init__(self, states, model_type = 'changepoint'):
        self.states = states
        self.model_type = model_type
        self.prediction_error, self.update, self.learning_rate, self.true_state, self.predicted_state,self.hazard_distance, self.hazard_trials = utils.extract_states(states)
    
    def run_mle(self):
        true_switch = .2 #changepoint rate
        true_noise = .125 #heli to bag noise 
        #data values
        true_ll = self.get_llik([true_switch, true_noise], *(self.predicted_state, self.prediction_error)) #given true params
        #true MLE values
        x0 = [true_switch, true_noise]
        result = scipy.optimize.minimize(self.get_llik, x0, args=(self.predicted_state, self.prediction_error), method='BFGS')
        print(f"MLE: Ω = {result.x[0]:.2f} (true value = {true_switch})")
        print(f"MLE: τ = {result.x[1]:.2f} (true value = {true_noise})")
        print(result)


    def get_llik(self, x, *args, model_name = 'normative'):
        '''
        Get nll for MLE
        '''
        #take args from minimizer
        Ω, τ = x
        actions, δ = args
        #initialize values
        logp_actions = np.zeros(len(actions))  
        #run loop
        for t in range(len(actions) - 1): #lazy fix for prediction error for now, double check this
            #run through model
            logp_action, _ = self.flexible_normative_model(Ω = Ω,
                                                        τ = τ, 
                                                        δ = δ[t],
                                                        agent_update = self.update[t], 
                                                        context = 'changepoint')

            #store necessary data 
            logp_actions[t] = logp_action
            #update states for next trial
        return -np.sum(logp_actions[1:]) 
    
    def get_pytensor_llik(self, Ω, τ):
        """
        PyTensor function estimate log-likelihood 
        for use with PyMC sampling.

        """
        def step(δ_t, agent_upd_t, prev_logp, Ω, τ):
            '''
            Step function for scan, serves as lambda expression in pytensor.scan
            agent_update and pe are drawn for each trial
            params stay constant
            logp is accumulated and returns sum
            '''
            current_logp, _ = self.PYMC_flexible_normative_model(
                Ω=Ω, τ=τ, δ=δ_t, agent_update=agent_upd_t, context=self.model_type
            )
            #accum logp
            logp = pt.set_subtensor(prev_logp[δ_t], current_logp)
            return logp

        #transform variables into pytensor variables
        pe_         = pt.as_tensor_variable(self.prediction_error, dtype="int32")
        updates_    = pt.as_tensor_variable(self.update, dtype="int32")

        #init ll vector
        logp_series = pt.zeros(len(self.prediction_error), dtype = "float64")

        #compute ll for each trial
        logp_series, param_updates = pytensor.scan( #where is param update? 
            fn = step,
            sequences = [pe_, updates_], 
            outputs_info = [logp_series],
            non_sequences = [Ω, τ]
        )
        #neg ll of entire sequence?
        ll = pt.sum(logp_series[1:])
        
        return ll 

    def flexible_normative_model(
        self,
        #params
        Ω: float = .20,   #Ω = changepoint probability
        τ: float = .125,  #τ = relative uncertainty
        UU: float = .001,     #UU = uncertainty underestimation 
        σ_motor: float = .001, 
        σ_LR: float = .90, 
        #priors 
        H: float = .2,      #H = changepoint or oddball prob dependening on the condition
        LW: float = .99,      #LW = likelihood weight
        σ: float = .125,      #σ = total var on predictive dist
        #data
        δ: float = .5,       #δ = prediction error
        agent_update: float = 50,  #participant update
        context: str = "change_point"): 
        """
        Flexible normative model that can be used for both changepoint and oddball conditions.
        """
        #eq 4 - needs hardset priors in
        U_val = stats.uniform.pdf(δ, 0, 300) ** LW
        N_val = stats.norm.pdf(δ, 0, σ) ** LW
        Ω = utils.calculate_omega(H, U_val, N_val)

        #eq 5 - in oddball, relative uncertainty requires drift rate D (?)
        τ = utils.calculate_tau(τ, UU)

        if context == 'changepoint':
            #eq 2
            alpha = utils.calculate_alpha_changepoint(Ω, τ)
        elif context == 'oddball':
            #eq 3
            alpha = utils.calculate_alpha_oddball(τ, Ω)
        else:
            raise ValueError("model_type must be either 'changepoint' or 'oddball'")   

        #eq 1
        normative_update = utils.calculate_normative_update(alpha, δ)
        
        #eq 7, variability of update
        σ_update = utils.calculate_sigma_update(σ_motor, normative_update, σ_LR)

        #eq 6
        L_normative = utils.calculate_L_normative(agent_update, normative_update, σ_update)

        #make log likelihood
        ll = np.log(L_normative)

        #sample for simulation
        sim_action = normative_update

        return ll, sim_action
    
    def PYMC_flexible_normative_model(
        self,
        #params
        Ω: float = .20,   #Ω = changepoint probability
        τ: float = .125,  #τ = relative uncertainty
        UU: float = .001,     #UU = uncertainty underestimation 
        σ_motor: float = .001, 
        σ_LR: float = .90, 
        #priors 
        H: float = .2,      #H = changepoint or oddball prob dependening on the condition
        LW: float = .5,      #LW = likelihood weight
        σ: float = .125,      #σ = total var on predictive dist
        #data
        δ: float = .5,       #δ = prediction error
        agent_update: float = 50,  #participant update
        context: str = "changepoint"): 
        """
        Flexible normative model that can be used for both changepoint and oddball conditions.
        -for fitting with PYMC min function
        """

        #eq 4 - in logspace 
        U_val = LW * pm.logp(pm.Uniform.dist(lower = 0, upper = 300), δ)
        N_val = LW * pm.logp(pm.Normal.dist(mu = 0, sigma = σ), δ)
        Ω =  pm.logaddexp(pm.math.log(H) + U_val,
                                pm.math.log(1 - H) + N_val)

        #eq 5 - in oddball, relative uncertainty requires drift rate D (?)
        τ = utils.calculate_tau(τ, UU)

        if context == 'changepoint':
            #eq 2
            alpha = utils.calculate_alpha_changepoint(Ω, τ)
        elif context == 'oddball':
            #eq 3
            alpha = utils.calculate_alpha_oddball(τ, Ω)
        else:
            raise ValueError("model_type must be either 'changepoint' or 'oddball'")   

        #eq 1
        normative_update = utils.calculate_normative_update(alpha, δ)
        
        #eq 7, variability of update
        σ_update = utils.calculate_sigma_update(σ_motor, normative_update, σ_LR)

        #eq 6
        L_normative = pm.logp(pm.Normal.dist(mu = normative_update, sigma = σ_update), agent_update)
        #make log likelihood
        #ll = np.log(L_normative)
        ll = L_normative
        #sample for simulation
        sim_action = normative_update

        return ll, sim_action

    def sim_data(self, 
                 total_trials:int = 100, 
                 model_name:str   = "flexible_normative_model",
                 condition:str    = 'changepoint'):
        """
        Simulate data by interacting with the PIE_CP_OB environment using flexible_normative_model.
        """
        self.total_trials = total_trials
        self.model_name = model_name
        self.condition = condition

        env = PIE_CP_OB(condition=self.condition, total_trials=self.total_trials)
        #all_states[epoch, tt] = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
        
        update = 0
        for t in range(self.total_trials):
            obs, _ = env.reset()
            pred_error = 0

            for i in range(2):
                # extract necessary trial info from env
                # Use the model to get sim_action
                ll, sim_action = self.flexible_normative_model(δ = pred_error, context = self.condition, agent_update=update)
                print(obs, sim_action, env.bucket_pos)

                obs, _, _ = env.step(action = None, direct_action = sim_action)
                pred_error = abs(obs[3]) # If PE is positive, model does not know to shift back. 
                update = sim_action - update
    
        # 0 = trial index, 1 = bucket_pos, 2 = bag_pos, 3 = helicopter_pos, 4 = hazard_trigger
        states = np.array([env.trials, env.bucket_positions, env.bag_positions, env.helicopter_positions, env.hazard_triggers])
           
        np.save("./data/bayesian_models/model_predictions.npy", states)
        return states

def plot_states(states):
    contexts = ["Change-point","Oddball"]
    # for c, context in enumerate(contexts):
    [trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers] = states#[c]

    plt.figure(figsize=(4, 2.5))
    # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
    plt.scatter(trials, bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=1, edgecolors='k')
    plt.plot(trials, helicopter_positions, label='Helicopter', color='green', linewidth=4)
    plt.plot(trials, bucket_positions, label='Bucket Position', color='orange', alpha=1, linewidth=2)

    plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
    plt.xlabel('Trial')
    plt.ylabel('Position')
    # plt.legend(frameon=True)
    plt.tight_layout()

#%%
#run 
if __name__ == "__main__":
    #load data and init model
    states = np.load('./data/pt_rnn_context/env_data.npy') #[trials, bucket_position, bag_position, helicopter_position]
    #states = np.load('./data/env_data_change-point.npy') #[trials, bucket_position, bag_position, helicopter_position]
    model = BayesianModel(states, model_type = 'changepoint')
    
    #run MLE
    model.run_mle()

    # #run simulation
    model.sim_data(total_trials=100, model_name = "flexible_normative_model", condition = 'changepoint')

    #run pytensor fitting
    plot_states(states)

    #set seed
    seed = sum(map(ord, "RL_PyMC"))
    rng = np.random.default_rng(seed)

    with pm.Model() as m:
        #priors
        Ω = pm.LogitNormal('Ω', mu = 0, sigma = .5)
        τ = pm.LogitNormal('τ', mu = 0, sigma = .5 )

        like = pm.Potential(name = "like",
                           var = model.get_pytensor_llik(Ω, τ)) #need to put in actions and pe here?


        #sample
        #trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=False)

        tr = pm.sample(random_seed = rng)
    
    #plot pytensor fitting 
    az.plot_trace(data = tr)
    az.plot_posterior(data = tr, ref_val = [.2, .125]) 

    #debug code
    #pytensor test with true priors

    def step_test(δ_t, agent_upd_t, logp_series, Ω, τ):
        '''
        Step function for scan, serves as lambda expression in pytensor.scan
        agent_update and pe are drawn for each trial
        params stay constant
        logp is accumulated and returns sum
        '''
        current_logp, _ = model.PYMC_flexible_normative_model(
            Ω=Ω, τ=τ, δ=δ_t, agent_update=agent_upd_t, context=model.model_type
        )
        #accum logp
        return current_logp #fix to be like guide

    # Transform the variables into appropriate PyTensor objects
    pe_ = pt.as_tensor_variable(model.prediction_error, dtype="int32")
    updates_ = pt.as_tensor_variable(model.update, dtype="int32")

    omega = pt.scalar("omega")
    tau = pt.scalar("tau")

    logp_series = pt.ones(len(model.prediction_error), dtype = "float64")

    # Compute the Q values for each trial
    logp_series, _ = pytensor.scan(
        fn= step_test,
        sequences=[pe_, updates_], 
        outputs_info=[logp_series],
        non_sequences=[omega, tau]
    )
    
    ll = pt.sum(logp_series[1:])



#%%
#all formulas

# #eq 1
# normative_update[t] = α[t] * δ[t]
# #eq 2
# CP_LR[t] = Ω + τ - (Ω  * τ)
# #eq 3
# OB_LR[t] = τ - (τ * Ω )
# #eq 4 - needs hardset priors in
# Ω = (H * U_val ) / (
#     (H * U_val + (1 - H) * N_val)
# )
# U_val = stats.uniform.pdf(δ, 0, 300) ** LW
# N_val = stats.norm.pdf(δ, 0, σ[t]) ** LW
# #eq 5 - in oddball, relative uncertainty requires drift rate D (?)
# τ = τ / UU
# #eq 6
# L_normative = stats.norm.pdf(participant_update[t], loc=normative_update[t], scale=σ_update)
# #eq 7, variability of update
# σ_update = σ_motor + normative_update(t) * σ_LR

# #make log likelihood
# ll = -log(L_normative)


    # pytensor_llik = pytensor.function(
    #     inputs = [Ω, τ],
    #     outputs = ll,
    #     on_unused_input="ignore"
    #     )


    # true_switch = .2 #changepoint rate
    # true_noise = .125 #heli to bag noise 
    # result = pytensor_llik(true_switch, true_noise)