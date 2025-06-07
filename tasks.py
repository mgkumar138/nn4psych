#%%
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import copy

class DiscretePredictiveInferenceEnv(gym.Env):
    def __init__(self, num_actions, condition="change-point", ):
        super(DiscretePredictiveInferenceEnv, self).__init__()
        
        self.action_space = spaces.Discrete(num_actions)        
        self.action_space = spaces.Discrete(num_actions)        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.full(3,num_actions), dtype=np.float32)

        # Initialize variables
        self.helicopter_pos = 2
        self.bucket_pos = 2
        self.bag_pos = self._generate_bag_position()
        
        # Task type: "change-point" or "oddball"
        self.task_type = condition
        
        # Trial counter and data storage for rendering
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125

    def reset(self):
        self.helicopter_pos = 2
        self.bucket_pos = 2
        self.bag_pos = self._generate_bag_position()
        self.trial = 0
        
        # Reset data storage
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        
        return np.array([self.bucket_pos, self.bag_pos, abs(self.bag_pos - self.bucket_pos)], dtype=np.float32)

    def step(self, action):
        # Update bucket position based on action
        self.bucket_pos = action
        self.bucket_pos = action
        
        # Determine bag position based on task type
        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(0, (num_actions-1) )
                self.helicopter_pos = np.random.randint(0, (num_actions-1) )
            self.bag_pos = self._generate_bag_position()  # Bag follows the stable helicopter position
        else:  # "oddball"
            if np.random.rand() < self.oddball_hazard:
                self.bag_pos = np.random.randint(0, (num_actions - 1) )  # Oddball event
                self.bag_pos = np.random.randint(0, (num_actions - 1) )  # Oddball event
            else:
                self.bag_pos = self._generate_bag_position()
        
        # Store positions for rendering
        self.trials.append(self.trial)
        self.bucket_positions.append(self.bucket_pos)
        self.bag_positions.append(self.bag_pos)
        self.helicopter_positions.append(self.helicopter_pos)

        # Calculate reward
        reward = -abs(self.bag_pos - self.bucket_pos) #max reward is 0
        reward = -abs(self.bag_pos - self.bucket_pos) #max reward is 0
        
        # Increment trial count
        self.trial += 1
        
        # Compute the new observation
        observation = np.array([self.bucket_pos, self.bag_pos, abs(self.bag_pos - self.bucket_pos)], dtype=np.float32)
        
        # Determine if the episode should end (e.g., after 100 trials)
        done = self.trial >= 100
        
        return observation, reward, done, {}
    
    def _generate_bag_position(self):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = self.helicopter_pos
        # add or subtract 1 from the helicopter position at 20% chance if at 0 or 4; 10% if in 1,2,3
        if self.helicopter_pos == 0:
            if np.random.rand() < 0.2:
                bag_pos += 1
        elif self.helicopter_pos == num_actions:
            if np.random.rand() < 0.2:
                bag_pos -= 1
        else: # 1,2,3: 80% chance to stay, 10% move left, or 10% right
            if np.random.rand() < 0.8:
                pass
            elif np.random.rand() < 0.5:
                bag_pos -= 1
            else:
                bag_pos += 1
        
        # Ensure the bag position is within the 0-300 range
        return max(0, min(4, bag_pos))

    def render(self, mode='human'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')

        plt.ylim(-.2, 4.2)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition")
        plt.legend()
        plt.show()

    def close(self):
        pass

class ContinuousPredictiveInferenceEnv(gym.Env):
    def __init__(self, condition="change-point", total_trials=200):
        super(ContinuousPredictiveInferenceEnv, self).__init__()
        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.array([300, 300, 300]), dtype=np.float32)
        self.total_trials = total_trials
        # Initialize variables
        self.helicopter_pos = 150
        self.bucket_pos = 150
        self.bag_pos = self._generate_bag_position()
        
        # Task type: "change-point" or "oddball"
        self.task_type = condition
        
        # Trial counter and data storage for rendering
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125

    def reset(self):
        self.helicopter_pos = 150
        self.bucket_pos = 150
        self.bag_pos = self._generate_bag_position()
        self.trial = 0
        
        # Reset data storage
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        
        return np.array([self.bucket_pos, self.bag_pos, self.bag_pos - self.bucket_pos], dtype=np.float32)

    def step(self, action):
        #Update bucket position based on action (incrementally move left or right)
        #Update bucket position based on action (incrementally move left or right)
        if action == 0:  # Move left
            self.bucket_pos = max(0, self.bucket_pos - 30)
        elif action == 1:  # Move right
            self.bucket_pos = min(300, self.bucket_pos + 30)
        # or update bucket position independently on action
        # self.bucket_pos = (action / 10 ) * 300
        # or update bucket position independently on action
        # self.bucket_pos = (action / 10 ) * 300
        
        # Determine bag position based on task type
        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(30, 270)
            self.bag_pos = self._generate_bag_position()  # Bag follows the stable helicopter position
        else:  # "oddball"
            if np.random.rand() < self.oddball_hazard:
                self.bag_pos = np.random.randint(0, 300)  # Oddball event
            else:
                self.bag_pos = self._generate_bag_position()
        
        # Store positions for rendering
        self.trials.append(self.trial)
        self.bucket_positions.append(self.bucket_pos)
        self.bag_positions.append(self.bag_pos)
        self.helicopter_positions.append(self.helicopter_pos)

        # Calculate reward
        reward = -abs(self.bag_pos - self.bucket_pos) #is this supposed to be "1 - PE"
        # calc PE 
        self.pred_error = 1 - abs(self.bag_pos - self.bucket_pos)
        self.error = self.bag_pos - self.bucket_pos
        reward = -abs(self.bag_pos - self.bucket_pos) #is this supposed to be "1 - PE"
        # calc PE 
        self.pred_error = 1 - abs(self.bag_pos - self.bucket_pos)
        self.error = self.bag_pos - self.bucket_pos
        # Increment trial count
        self.trial += 1
        
        # Compute the new observation
        observation = np.array([self.bucket_pos, self.bag_pos, self.error], dtype=np.float32)
        observation = np.array([self.bucket_pos, self.bag_pos, self.error], dtype=np.float32)
        
        # Determine if the episode should end (e.g., after 100 trials)
        done = self.trial >= self.total_trials
        
        return observation, reward, done, {}
    
    def _generate_bag_position(self):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = int(np.random.normal(self.helicopter_pos, 20))
        # Ensure the bag position is within the 0-300 range
        return max(0, min(300, bag_pos))

    def render(self, mode='human'):
        plt.figure(figsize=(10, 6))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition")
        plt.legend()
        plt.show()

    def close(self):
        pass

class PIE_CP_OB:
    def __init__(self, condition="changepoint", total_trials=200,max_time=300, train_cond=False, 
                 max_displacement=15, reward_size=7.5, step_cost=0.0, alpha=1):
        super(PIE_CP_OB, self).__init__()
        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -300]), 
                                            high=np.array([300, 300, 300, 300]), dtype=np.float32) # helicopter pos during training, bucket pos, bag pos, bag-bucket pos, CP or OB context
        self.max_time = max_time

        self.min_obs_size = 1
        self.max_obs_size = 301
        self.bound_helicopter = 30 # keep helicopter within 30 units of the periphary
        self.total_trials = total_trials
        self.hide_variable = 0 # this means that if a variable is 0, it is supposed to be hidden from the agent's consideration

        # Initialize variables
        self.helicopter_pos =  np.random.randint(self.min_obs_size+self.bound_helicopter, self.max_obs_size-self.bound_helicopter)
        self.bucket_pos = np.random.randint(self.min_obs_size+self.bound_helicopter, self.max_obs_size-self.bound_helicopter)
        self.pred_error = 0
        self.sample_bag_pos = self._generate_bag_position(self.helicopter_pos)
        self.reward = 0
        self.max_disp = max_displacement #changed to 1 from 30
        self.reward_size = reward_size
        self.step_cost = step_cost
        self.alpha = alpha
        self.velocity = 0

        # Task type: "change-point" or "oddball"
        self.task_type = condition
        self.train_cond = train_cond  # either True or False, if True, helicopter position is shown to agent. if False helicopter position is 0

        if condition == "changepoint":
            self.context =  np.array([1,0])
        elif condition == "oddball":
            self.context =  np.array([0,1])

        
        # Trial counter and data storage for rendering
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        self.hazard_triggers = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125
    
    def normalize_states_(self,x):
        # normalize states to be between -1 to 1 to feed to network
        # return np.array([x[0]/self.maxobs, x[1]/self.maxobs , x[2]/(self.maxobs/2)])
        ranges = np.array([[0, 300], [0, 300], [0, 300], [-300, 300]])
        normalized_vector = np.array([2 * (x[i] - ranges[i, 0]) / (ranges[i, 1] - ranges[i, 0]) - 1 for i in range(len(x))])
        return normalized_vector
    
    def normalize_states(self,x):
        # normalize states to be between -1 to 1 to feed to network
        return x/300

    def reset(self):
        # reset at the start of every trial. Observation inclues: helicopter 
        self.time = 0
        self.hazard_trigger = 0
        self.velocity = 0

        if self.task_type == "changepoint":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(self.min_obs_size + self.bound_helicopter,self.max_obs_size-self.bound_helicopter)  # change helicopter position based on hazard rate
                self.hazard_trigger = 1
            self.sample_bag_pos = self._generate_bag_position(self.helicopter_pos)  # Bag follows the stable helicopter position

        else:  # "oddball"

            # slow change in helicopter position in the oddball condition with small SD
            slow_shift = int(np.random.normal(0, 7.5))
            self.helicopter_pos += slow_shift
            self.helicopter_pos = np.clip(self.helicopter_pos, self.min_obs_size + self.bound_helicopter,self.max_obs_size-self.bound_helicopter)

            if np.random.rand() < self.oddball_hazard:
                self.sample_bag_pos = np.random.randint(0, 300)  # Oddball event
                self.hazard_trigger = 1
            else:
                self.sample_bag_pos = self._generate_bag_position(self.helicopter_pos)

        if self.train_cond:
            self.obs = np.array([self.helicopter_pos, self.bucket_pos, copy.copy(self.hide_variable), self.pred_error], dtype=np.float32)  # initialize initial observation.
        else:
            self.obs = np.array([copy.copy(self.hide_variable), self.bucket_pos, copy.copy(self.hide_variable), self.pred_error], dtype=np.float32)  # initialize initial observation. 

        self.done = False

        return self.obs, self.done
    
    def step(self, action, direct_action=None):
        # idea is to have 2 separate phases within each trial. Phase 1: allow the agent to move the bucket to a desired position. Phase 2: press confirmation button to start bag drop
        #adding direct action to allow bayesian agent to choose action directly
        self.time += 1

        # Phase 1:
        # Update bucket position based on action before confirmation
        if action == 0: 
            # Move left
            self.gt = -self.max_disp
        elif action == 1:
            # Move right
            self.gt = self.max_disp
        elif action == 2:
            # stay
            self.gt = 0
            self.velocity = 0
        elif direct_action is not None:
            self.gt = direct_action

        # print(self.bucket_pos, self.xt, self.gt)
        self.velocity += self.alpha * (-self.velocity + self.gt)
        newbucket_pos = copy.copy(self.bucket_pos) + self.velocity

        if newbucket_pos > self.max_obs_size or newbucket_pos < self.min_obs_size:
            self.velocity = 0
            newbucket_pos = copy.copy(self.bucket_pos)

        # self.bucket_pos += self.gt
        self.bucket_pos = np.clip(copy.copy(newbucket_pos), a_min=self.min_obs_size,a_max=self.max_obs_size)

        # update the observation vector with new bucket position 
        self.obs = copy.copy(self.obs)
        self.obs[1] = self.bucket_pos
        self.reward = self.step_cost # either 0 to -1/self.max_obs_size # punish for every timestep
        
        # # if max time is reached. terminate trial
        # if self.time >= self.max_time-1:
        #     self.done = True
        #     self.reward = 0
            

        # Phase 2:
        # confirm bucket position to start bag drop
        if action == 2 or self.time >= self.max_time-1 or direct_action is not None:
            self.bag_pos = copy.copy(self.sample_bag_pos)
            self.pred_error = self.bag_pos - self.bucket_pos     

            # Compute the new observation
            if self.train_cond:
                self.obs = np.array([self.helicopter_pos, self.bucket_pos, self.bag_pos, self.pred_error], dtype=np.float32) 
            else:
                self.obs = np.array([copy.copy(self.hide_variable), self.bucket_pos, self.bag_pos, self.pred_error], dtype=np.float32)  # include bucket and bag position. hide pre
            
            # reward or punish inactivity
            if np.random.uniform()<0.0:
                # randomly punish for not catching bag
                self.reward = -abs(self.bag_pos - self.bucket_pos)/self.max_obs_size  # reward is negative scalar, proportional to distance between bucket and bag. Faster to train agent
            else:
                # reward follows gaussian distribution. the closer the bucket is to the bag positin, the higher the reward, with maximum 1.
                df = ((self.bag_pos - self.bucket_pos)/self.reward_size)**2
                self.reward = np.exp(-0.5*df) #* 1/(self.reward_size * np.sqrt(2*np.pi))

            # penalize if agent doesnt choose to confirm
            if self.time >= self.max_time-1:
                # self.reward += self.step_cost
                self.reward = self.step_cost

            self.trial += 1
            self.done = True

            # Store positions for rendering
            self.trials.append(self.trial)
            self.bucket_positions.append(self.bucket_pos)
            self.bag_positions.append(self.bag_pos)
            self.helicopter_positions.append(self.helicopter_pos)
            self.hazard_triggers.append(self.hazard_trigger)

        return self.obs, self.reward, self.done
    
    def _generate_bag_position(self, helicopter_pos):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = int(np.random.normal(helicopter_pos, 20))
        # Ensure the bag position is within the 0-300 range
        return np.clip(bag_pos, self.min_obs_size,self.max_obs_size)

    def render(self, epoch=0):
        plt.figure(figsize=(10, 6))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='b',marker='o', linestyle='-.', alpha=0.5)

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition - Epoch: {epoch}")
        plt.legend()
        plt.show()

        return np.array([self.trials, self.bucket_positions, self.bag_positions, self.helicopter_positions, self.hazard_triggers])

class PIE_CP_OB_v2:
    def __init__(self, condition="change-point", total_trials=200,max_time=300, train_cond=False, 
                 max_displacement=15, reward_size=7.5, step_cost=0.0, alpha=1):
        super(PIE_CP_OB_v2, self).__init__()
        
        # Observation: currentCurrent bucket position, last bag position, and prediction error
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -301, 0,0]), 
                                            high=np.array([301, 301, 301,301, 301,301]), dtype=np.float32) # helicopter pos during training, bucket pos, bag pos, bag-bucket pos, CP or OB context
        self.max_time = max_time

        self.min_obs_size = 1
        self.max_obs_size = 301
        self.bound_helicopter = 30 # keep helicopter within 30 units of the periphary
        self.total_trials = total_trials
        self.hide_variable = 0 # this means that if a variable is 0, it is supposed to be hidden from the agent's consideration

        # Initialize variables
        self.helicopter_pos =  np.random.randint(self.min_obs_size+self.bound_helicopter, self.max_obs_size-self.bound_helicopter)  # sample helicopter pos within limits
        self.bucket_pos = np.random.randint(self.min_obs_size+self.bound_helicopter, self.max_obs_size-self.bound_helicopter) # sample bucket pos within limits
        self.bag_pos = self._generate_bag_position(self.helicopter_pos) # sample bag pos with mean centered at heli pos

        self.prev_bag_pos = 0 # previous trial's bag pos
        self.prev_bucket_pos = 0 # previous trial's bucket pos
        self.pred_error = self.prev_bag_pos - self.prev_bag_pos  # pred error between bag and bucket

        self.reward = 0
        self.max_disp = max_displacement #changed to 1 from 30
        self.reward_size = reward_size
        self.step_cost = step_cost
        self.alpha = alpha
        self.velocity = 0

        # Task type: "change-point" or "oddball"
        self.task_type = condition
        self.train_cond = train_cond  # either True or False, if True, helicopter position is shown to agent. if False helicopter position is 0

        if condition == "change-point":
            self.context =  np.array([1,0])
        elif condition == "oddball":
            self.context =  np.array([0,1])

        
        # Trial counter and data storage for rendering
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        self.hazard_triggers = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125
    
   
    def normalize_states(self,x):
        # normalize states to be between -1 to 1 to feed to network
        return x/300

    def reset(self):
        # reset at the start of every trial. Observation inclues: helicopter 
        self.time = 0
        self.hazard_trigger = 0
        self.velocity = 0

        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(self.min_obs_size + self.bound_helicopter,self.max_obs_size-self.bound_helicopter)  # change helicopter position based on hazard rate
                self.hazard_trigger = 1
            self.bag_pos = self._generate_bag_position(self.helicopter_pos)  # Bag follows the stable helicopter position

        else:  # "oddball"

            # slow change in helicopter position in the oddball condition with small SD
            slow_shift = int(np.random.normal(0, 7.5))
            self.helicopter_pos += slow_shift
            self.helicopter_pos = np.clip(self.helicopter_pos, self.min_obs_size + self.bound_helicopter,self.max_obs_size-self.bound_helicopter)

            if np.random.rand() < self.oddball_hazard:
                self.bag_pos = np.random.randint(0, 300)  # Oddball event
                self.hazard_trigger = 1
            else:
                self.bag_pos = self._generate_bag_position(self.helicopter_pos)

        if self.train_cond:
            self.obs = np.array([self.helicopter_pos, self.bucket_pos, copy.copy(self.hide_variable), self.pred_error, self.prev_bucket_pos, self.prev_bag_pos], dtype=np.float32)  # initialize initial observation.
        else:
            self.obs = np.array([copy.copy(self.hide_variable), self.bucket_pos, copy.copy(self.hide_variable), self.pred_error, self.prev_bucket_pos, self.prev_bag_pos], dtype=np.float32)  # initialize initial observation. 

        self.done = False

        return self.obs, self.done
    
    def step(self, action, direct_action=None):
        # idea is to have 2 separate phases within each trial. Phase 1: allow the agent to move the bucket to a desired position. Phase 2: press confirmation button to start bag drop
        #adding direct action to allow bayesian agent to choose action directly
        self.time += 1

        # Phase 1:
        # Update bucket position based on action before confirmation
        if action == 0: 
            # Move left
            self.gt = -self.max_disp
        elif action == 1:
            # Move right
            self.gt = self.max_disp
        elif action == 2:
            # stay
            self.gt = 0
            self.velocity = 0
        elif direct_action is not None:
            self.gt = direct_action

        # print(self.bucket_pos, self.xt, self.gt)
        self.velocity += self.alpha * (-self.velocity + self.gt)
        newbucket_pos = copy.copy(self.bucket_pos) + self.velocity

        if newbucket_pos > self.max_obs_size or newbucket_pos < self.min_obs_size:
            self.velocity = 0
            newbucket_pos = copy.copy(self.bucket_pos)

        # self.bucket_pos += self.gt
        self.bucket_pos = np.clip(copy.copy(newbucket_pos), a_min=self.min_obs_size,a_max=self.max_obs_size)

        # update the observation vector with new bucket position 
        self.obs = copy.copy(self.obs)
        self.obs[1] = self.bucket_pos
        self.reward = self.step_cost # either 0 to -1/self.max_obs_size # punish for every timestep
        
        # # if max time is reached. terminate trial
        # if self.time >= self.max_time-1:
        #     self.done = True
        #     self.reward = 0
            

        # Phase 2:
        # confirm bucket position to start bag drop
        if action == 2 or self.time >= self.max_time-1 or direct_action is not None:

            # show the bag drop
            if self.train_cond:
                self.obs = np.array([self.helicopter_pos, self.bucket_pos, self.bag_pos, self.pred_error, self.prev_bucket_pos, self.prev_bag_pos], dtype=np.float32) 
            else:
                self.obs = np.array([copy.copy(self.hide_variable), self.bucket_pos, self.bag_pos, self.pred_error, self.prev_bucket_pos, self.prev_bag_pos], dtype=np.float32)  # include bucket and bag position. hide pre
            
            # reward follows gaussian distribution. the closer the bucket is to the bag positin, the higher the reward, with maximum 1.
            df = ((self.bag_pos - self.bucket_pos)/self.reward_size)**2
            self.reward = np.exp(-0.5*df)

            # compute new prediction error
            self.prev_bag_pos = copy.copy(self.bag_pos)
            self.prev_bucket_pos = copy.copy(self.bucket_pos)
            self.pred_error = self.prev_bag_pos - self.prev_bucket_pos  

            # if agent doesnt choose to confirm, bag still drops but no reward, even if the bucket is in the correct location. Induce the agent to press confirm
            if self.time >= self.max_time-1:
                self.reward = self.step_cost

            self.trial += 1
            self.done = True

            # Store positions for rendering
            self.trials.append(self.trial)
            self.bucket_positions.append(self.bucket_pos)
            self.bag_positions.append(self.bag_pos)
            self.helicopter_positions.append(self.helicopter_pos)
            self.hazard_triggers.append(self.hazard_trigger)

        return self.obs, self.reward, self.done
    
    def _generate_bag_position(self, helicopter_pos):
        """Generate a new bag position around the current helicopter location within bounds."""
        bag_pos = int(np.random.normal(helicopter_pos, 20))
        # Ensure the bag position is within the 0-300 range
        return np.clip(bag_pos, self.min_obs_size,self.max_obs_size)

    def render(self, epoch=0):
        plt.figure(figsize=(10, 6))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='b',marker='o', linestyle='-.', alpha=0.5)

        plt.ylim(-10, 310)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition - Epoch: {epoch}")
        plt.legend()
        plt.show()

        return np.array([self.trials, self.bucket_positions, self.bag_positions, self.helicopter_positions, self.hazard_triggers])

# Run
if __name__ == "__main__":
    trials = 10
    train_cond = False
    max_time = 300
    max_displacement = 15
    reward_size = 5
    alpha = 1

    for task_type in ["change-point"]:
        env = PIE_CP_OB_v2(condition=task_type,max_time=max_time, 
                        total_trials=trials, train_cond=train_cond,
                        max_displacement=max_displacement, alpha=alpha, reward_size=reward_size)
        
        for trial in range(trials):
            next_obs, done = env.reset()
            total_reward = 0

            while not done:
                action = np.random.choice(np.arange(3), 1)  # For testing, we use random actions
                next_obs, reward, done = env.step(action)
                norm_next_obs = env.normalize_states(next_obs)  # normalize vector to bound between something resonable for the RNN to handle
                next_state = np.round(np.concatenate([norm_next_obs,env.context]),5)

                total_reward += reward

                print(env.trial, env.time, action, next_state, reward, done)

    states = env.render()
    # np.save(f'./data/env_data_{task_type}', states)
    # plt.hist(np.array(env.bucket_positions).reshape(-1), bins=np.linspace(0,300,21))
