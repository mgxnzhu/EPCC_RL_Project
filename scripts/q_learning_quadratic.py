from osim.env import L2RunEnv
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

DEFAULT_SEED = 20180101
rng = np.random.RandomState(DEFAULT_SEED)

env = L2RunEnv(visualize=False)
# Obtain the dimension observation space and action space
dim_obs = env.get_observation_space_size()
dim_act = env.get_action_space_size()

# Set the range of action values
action_low = -0.1
action_high = 0.1

# Set hyperparameters
discount = 0.0001
epsilon = 0.9
episode = 2000

# FIXME: logging
coeff_list = []

class qfunction:
    # random initialization
    def __init__(self, dim_obs, dim_act, rng=None):
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.dim = (dim_obs + dim_act + 1) * (dim_obs + dim_act) // 2
        self.coeff = rng.uniform(-1, 1, self.dim)

    def __call__(self, obs, act):
        con_vec = np.concatenate((obs,act))
        quad_vec = quadartic(con_vec)
        res = np.sum(quad_vec*self.coeff)
        return res

    def update(self, coeff):
        self.coeff = coeff
        
def quadartic(vec):
    # covert (x1, x2, x3, ...) to (x1^2, x1x2, x1x3, ..., x2^2, x2x3, ...)
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    res = []
    for i, x in enumerate(vec):
        res = np.concatenate((res, x*vec[i:]))
    return res

# Initialize Q function
qf = qfunction(dim_obs, dim_act)
model = LinearRegression(fit_intercept=False)
# Initialize the dataset:(xdata, ydata)
xdata = np.zeros((qf.dim,))
ydata = np.zeros((1,))

action0 = np.zeros(dim_act)
for i in range(episode):
    # Initialize a new simulation
    state = np.array(env.reset())
    reward = 0

    # Run the simulation until the framework stop it
    done = False
    while not done:
        # get the action based on Q function and epsilon greedy
        if (rng.rand() < epsilon) :
            # exploration: randomly choose an action
            action = rng.uniform(action_low, action_high, dim_act)
        else:
            # exploitation: choose the action maximumizing Q function
            action_func = lambda x: -qf(state, x)
            bnds = Bounds(action_low, action_high)
            res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
            action = res.x

        # evolve the system to the next time stamp
        state_, reward, done, info = env.step(action)
        state_ = np.array(state_)

        # build the dataset
        action_func = lambda x: -qf(state_, x)
        bnds = Bounds(action_low, action_high)
        res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
        max_q = -res.fun
        # {s, a} and [r + gamma * max_a` Q(s`, a`)]
        xx = quadartic(np.concatenate((state, action)))
        yy = np.array(reward + discount * max_q)
        # put the data point into dataset
        xdata = np.vstack((xdata, xx))
        ydata = np.vstack((ydata, yy))

        # Do linear regression and update Q function
        model.fit(xdata, ydata)
        qf.update(model.coef_.T)
        
        # FIXME: logging 
        coeff2 = np.sum(np.square(model.coef_.T))
        coeff_list.append(coeff2)
        if coeff2 == float('inf'):
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111)
            ax.semilogy(range(len(coeff_list)),coeff_list)
            ax.set_ylim(1e-10, 1e300)
            ax.set_ylabel("$\sum{a_i^2}$")
            ax.set_xlabel("step")
            fig.savefig("coeff2.pdf")
            sys.exit()

        # Update state
        state = state_