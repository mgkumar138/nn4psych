import numpy as np

class res_ac:
    def __init__(self, ninput, nrnn, nact, gamma,clr, alr, seed):
        self.ninput = ninput
        self.nrnn = nrnn
        self.nact = nact
        self.ncri = 1
        self.gamma = gamma
        self.gain = 1.5
        self.tau = 10
        self.dt = 1 # 1 millisecond
        self.cp = 0.1
        self.alpha = self.dt/self.tau
        self.afunc = np.tanh
        self.sigma = 0.0
        self.beta_action = 1

        self.clr = clr
        self.alr = alr

        np.random.seed(seed)
        self.wx = np.random.normal(size=(self.nrnn, self.ninput)) * 1/self.ninput
        self.wh = np.random.normal(size=(self.nrnn, self.nrnn)) * self.gain/np.sqrt(self.nrnn * self.cp)
        mask = np.random.choice([True, False], size=self.wh.shape, p=[1-self.cp, self.cp])
        self.wh[mask] = 0
        self.wact = np.random.normal(size=(self.nact, self.nrnn)) * 1/self.nrnn
        self.wcri = np.random.normal(size=(self.ncri, self.nrnn)) * 1/self.nrnn

        # eigenvalues = np.linalg.eigvals(self.wh)
        # spectral_radius = max(abs(eigenvalues))
        # print("Spec Rad:", spectral_radius)
    
    def reset(self):
        self.h = np.random.normal(size=[self.nrnn,1]) * 0.001
        
    def get_rnn(self, state):
        I = self.wx @ state
        rprev = self.wh @ self.afunc(self.h)
        ns = np.random.normal(size=[self.nrnn,1]) * self.sigma

        self.h += self.alpha * (-self.h + I + rprev + ns)
        self.r = self.afunc(self.h)
        return self.r
    
    def softmax(self,x):
        x_max = np.max(x, axis=-1, keepdims=True)
        unnormalized = np.exp(x - x_max)
        return unnormalized/np.sum(unnormalized, axis=-1, keepdims=True)
    
    def get_action(self,r, bias):
        a = self.wact @ r
        self.aprob = self.softmax(self.beta_action * (a[:,0]+bias))
        A = np.random.choice(a=np.arange(self.nact), p=np.array(self.aprob))
        self.g = np.zeros([self.nact,1])
        self.g[A] = 1
        return A
        
    def get_value(self, r):
        self.v = self.wcri @ r
        return self.v

    
    def learn(self,td):
        
        # value = self.get_value(self.r)
        # newr = self.get_rnn(newstate, update_self=False)
        # newvalue = self.get_value(newr)

        # # td error
        # self.td = reward + self.gamma * newvalue - value

        # update critic & actor
        dwc = self.r.T * td 
        self.wcri += self.clr * dwc

        dwa = (self.g - self.aprob[:,None]) @ self.r.T * td
        self.wact += self.alr * dwa


    def get_weights(self):
        return [self.wx, self.wh, self.wact, self.wcri]