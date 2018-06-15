from osim.env import L2RunEnv
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.linear_model import LinearRegression

DEFAULT_SEED = 20180101
rng = np.random.RandomState(DEFAULT_SEED)

env = L2RunEnv(visualize=False)
# Obtain the dimension observation space and action space
dim_obs = env.get_observation_space_size()
dim_act = env.get_action_space_size()

# Set the range of action values
action_low = -1
action_high = 1

# Set hyperparameters
discount = 1
epsilon = 0.1
episode = 2000


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


# Initialize Q function
qf = qfunction(dim_obs, dim_act)
model = LinearRegression(fit_intercept=False)
# Initialize the dataset
xdata = np.zeros((dim_obs+dim_act,))
ydata = np.zeros((1,))

action0 = np.zeros(dim_act)
for i in range(episode):
    # Initialize a new simulation
    state = np.array(env.reset())
    reward = 0
    print("Episode ",i)

    # Run the simulation until the framework stop it
    done = False
    while not done:
        # get the action based on Q function and epsilon greedy
        if (rng.rand() < epsilon) :
            # exploration
            action = rng.uniform(action_low, action_high, dim_act)
        else:
            # exploitation
            action_func = lambda x: -qf(state, x)
            bnds = Bounds(action_low, action_high)
            res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
            action = res.x

        # evolve the system to the next time stamp
        state_, reward, done, info = env.step(action)
        state_ = np.array(state_)

        # Obtain {s, a} and [r + gamma * max_a` Q(s`, a`)]
        action_func = lambda x: -qf(state_, x)
        bnds = Bounds(-1,1)
        res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
        max_q = -res.fun
        yy = np.array(reward + discount * max_q)
        xx = np.concatenate((state, action))
        xdata = np.vstack((xdata, xx))
        ydata = np.vstack((ydata, yy))

        # Do linear regression
        model.fit(xdata, ydata)
        qf.update(model.coef_.T)

        # Update state
        state = state_

np.save("model.npy", model.coef_)

