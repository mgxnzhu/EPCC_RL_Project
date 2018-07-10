from osim.env import L2RunEnv
import numpy as np
from scipy.optimize import minimize, Bounds

DEFAULT_SEED = 20180101
rng = np.random.RandomState(DEFAULT_SEED)

env = L2RunEnv(visualize=True)
# Obtain the dimension observation space and action space
dim_obs = env.get_observation_space_size()
dim_act = env.get_action_space_size()

# Set the range of action values
action_low = env.action_space.low
action_high = env.action_space.high # bounds of action space by env
bnds = Bounds(action_low, action_high)

model = np.genfromtxt("model.csv", delimiter=',')
coef_ = model[:-1]
intercept_ = model[-1]

def qfunc(obs, act):
    X = np.concatenate((obs, act))
    res = np.sum(X * coef_) + np.asscalar(intercept_)
    return res

def get_maxq(state_):
    # get maximum of Q(s', a') under given s'
    action_func = lambda x: -qfunc(state_, x)
    action0 = 0.5 * np.ones(dim_act) # the center of action space
    res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
    # note: https://en.wikipedia.org/wiki/Sequential_quadratic_programming
    return (-res.fun, res.x)

# Initialize a new simulation
state = env.reset()
reward = 0

# Run the simulation until the framework stop it
done = False
while not done:
    # get the action based on Q function
    max_q, action = get_maxq(state)

    # evolve the system to the next time stamp
    state, reward, done, info = env.step(action)