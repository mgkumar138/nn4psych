#%%
import numpy as np
import matplotlib.pyplot as plt

class Heli_Bag:
    def __init__(self, condition, train_cond=False, alpha=0.2, max_disp=0.1):
        self.truescale = np.array([0,300])

        self.env_min = 0
        self.env_max = 2
        self.scale = self.env_max/300

        self.heli_bound = 30*self.scale
        self.reward_size = 7.5*self.scale  # 0.05
        self.max_time = 300

        self.helipos = np.random.uniform(self.env_min+self.heli_bound,self.env_max-self.heli_bound,1)
        self.bagpos = self.sample_bag(self.helipos)
        self.bucketpos = np.random.uniform(self.env_min+self.heli_bound,self.env_max-self.heli_bound,1)
        self.pred_error =  np.array([0])

        self.reward = np.array([0])
        self.max_disp = max_disp  # 15
        self.alpha = alpha
        self.velocity = 0

        # Task type: "change-point" or "oddball"
        self.task_type = condition
        self.train_cond = train_cond  # either True or False, if True, helicopter position is shown to agent. if False helicopter position is 0

        if condition == "change-point":
            self.context =  np.array([1,0])
        elif condition == "oddball":
            self.context =  np.array([0,1])

        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        self.hazard_triggers = []

        # Hazard rates for the different conditions
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125
    
    def sample_bag(self, helipos):
        return np.clip(np.random.normal(loc=helipos, scale=20*self.scale), self.env_min, self.env_max)

    def reset(self):
    # reset at the start of every trial. Observation inclues: helicopter 
        self.time = 0
        self.velocity = 0
        self.hazard_trigger = 0

        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helipos = np.random.uniform(self.env_min+self.heli_bound,self.env_max-self.heli_bound,1)
                self.hazard_trigger = 1
            self.bagpos = self.sample_bag(self.helipos)  

        else:  # "oddball"

            # slow change in helicopter position in the oddball condition with small SD
            slow_shift = int(np.random.normal(0, 7.5*self.scale))
            self.helipos += slow_shift
            self.helipos = np.clip(self.helipos, self.env_min+self.heli_bound,self.env_max-self.heli_bound)

            if np.random.rand() < self.oddball_hazard:
                self.bagpos = np.random.uniform(self.env_min, self.env_max)  # Oddball event
                self.hazard_trigger = 1
            else:
                self.bagpos =  self.sample_bag(self.helipos)

        if self.train_cond:
            self.obs = np.array([self.helipos, self.bucketpos, np.array([0]), self.pred_error], dtype=np.float32)[:,0]  # initialize initial observation.
        else:
            self.obs = np.array([np.array([0]), self.bucketpos, np.array([0]), self.pred_error], dtype=np.float32)[:,0]   # initialize initial observation. 

        self.done = False

        return self.obs, self.done

    def step(self, action):
        # idea is to have 2 separate phases within each trial. Phase 1: allow the agent to move the bucket to a desired position. Phase 2: press confirmation button to start bag drop
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

        # print(self.bucket_pos, self.xt, self.gt)

        self.velocity += self.alpha * (-self.velocity + self.gt)
        newbucket_pos = self.bucketpos.copy() + self.velocity

        if newbucket_pos > self.env_max or newbucket_pos < self.env_min:
            self.velocity = 0
            newbucket_pos = self.bucketpos.copy()

        # self.bucket_pos += self.gt
        self.bucketpos = np.clip(newbucket_pos.copy(), a_min=self.env_min,a_max=self.env_max)

        # update the observation vector with new bucket position 
        self.obs = self.obs.copy()
        self.obs[1] = self.bucketpos
        self.reward = np.array([0])
        
        # Phase 2:
        # confirm bucket position to start bag drop
        if action == 2 or self.time >= self.max_time-1:
            self.pred_error = self.bagpos - self.bucketpos     

            # Compute the new observation
            if self.train_cond:
                self.obs = np.array([self.helipos, self.bucketpos, self.bagpos, self.pred_error], dtype=np.float32)[:,0] 
            else:
                self.obs = np.array([np.array([0]), self.bucketpos, self.bagpos, self.pred_error], dtype=np.float32)[:,0]   # include bucket and bag position. hide pre
            
            df = ((self.bagpos - self.bucketpos)/self.reward_size)**2
            self.reward = np.exp(-0.5*df) #* 1/(self.reward_size * np.sqrt(2*np.pi))

            # if never confirm, dont give reward
            # if action != 2:
            #     self.reward = np.array([0]) 

            self.trial += 1
            self.done = True

            # Store positions for rendering
            self.trials.append(self.trial)
            self.bucket_positions.append(self.bucketpos)
            self.bag_positions.append(self.bagpos)
            self.helicopter_positions.append(self.helipos)
            self.hazard_triggers.append(self.hazard_trigger)

        return self.obs, self.reward[0], self.done
    

    def render(self, epoch=0):
        plt.figure(figsize=(10, 6))
        # plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='blue')
        plt.plot(self.trials, self.bag_positions, label='Bag Position', color='red', marker='o', linestyle='-.', alpha=0.5)
        plt.plot(self.trials, self.helicopter_positions, label='Helicopter', color='green', linestyle='--')
        plt.plot(self.trials, self.bucket_positions, label='Bucket Position', color='b',marker='o', linestyle='-.', alpha=0.5)

        plt.ylim(-0.1, 2.1)  # Set y-axis limit from 0 to 300
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition - Epoch: {epoch}")
        plt.legend()
        plt.show()

        # return np.array([self.trials, self.bucket_positions, self.bag_positions, self.helicopter_positions, self.hazard_triggers])
    

# Run
if __name__ == "__main__":

    trials = 100
    train_cond = False
    max_time = 300
    alpha = 0.2

    actions = np.array([1,1,1, 2])
    for task_type in ["change-point"]:
        env = Heli_Bag(condition=task_type)
        
        for trial in range(trials):
            obs, done = env.reset()
            total_reward = 0

            while not done:
                action = np.random.randint(3)  
                # action = actions[env.time]
                obs, reward, done = env.step(action)
                total_reward += reward

                print(env.trial, env.time, action,obs, reward, done)

        env.render()
    
    # plt.hist(np.array(env.bucket_positions).reshape(-1), bins=np.linspace(0,2,11))

# %%
