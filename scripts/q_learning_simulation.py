from osim.env import L2RunEnv
import numpy as np
from scipy.optimize import minimize, Bounds

DEFAULT_SEED = 20180101
rng = np.random.RandomState(DEFAULT_SEED)

env = L2RunEnv(visualize=False)
# Obtain the dimension observation space and action space
dim_obs = env.get_observation_space_size()
dim_act = env.get_action_space_size()

class qfunction:
    def __init__(self, dim_obs, dim_act, rng=None):
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.obs_coeff = rng.uniform(-1, 1, dim_obs)
        self.act_coeff = rng.uniform(-1, 1, dim_act)
        
    def __call__(self, obs, act):
        obs_term = obs * self.obs_coeff
        act_term = act * self.act_coeff
        res = np.sum(obs_term) + np.sum(act_term)
        return res
    
    def action_func(self, obs):
        func = lambda act: -self(obs, act) # take opposite value for minimization
        return func
    
    def update(self, coeff):
        self.obs_coeff = coeff[:self.dim_obs]
        self.act_coeff = coeff[self.dim_obs:]

model = np.load("model.npy")
qf = qfunction(dim_obs, dim_act)
qf.update(model)
action0 = np.zeros(dim_act)

# Initialize a new simulation
state = env.reset()
reward = 0

# Run the simulation until the framework stop it
done = False
while not done:
    # get the action based on Q function
    action_func = lambda x: -qf(state, x)
    bnds = Bounds(-1,1)
    res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
    action = res.x

    # evolve the system to the next time stamp
    state, reward, done, info = env.step(action)

