from osim.env import L2RunEnv
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.linear_model import LinearRegression, SGDRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_SEED = 20180101
rng = np.random.RandomState(DEFAULT_SEED)

env = L2RunEnv(visualize=False)
# Obtain the dimension observation space and action space
dim_obs = env.get_observation_space_size()
dim_act = env.get_action_space_size()

# Set the range of action values
action_low = env.action_space.low
action_high = env.action_space.high # bounds of action space by env
bnds = Bounds(action_low, action_high)

import argparse

parser = argparse.ArgumentParser(description='Set the evaluation dataset')
parser.add_argument('--gamma', nargs="?", type=float, default=0.6,
    help='discount factor')
parser.add_argument('--alpha', nargs="?", type=float, default=0.01,
    help='learning rate')
parser.add_argument('--episode', nargs="?", type=int, default=100,
    help='Number of simulation episodes')
parser.add_argument('--epsilon', nargs="?", type=float, default=0.3,
    help='epsilon greedy factor')
parser.add_argument('--batch', nargs="?", type=int, default=10,
    help='batch size for GD')
args = parser.parse_args()

# Set hyperparameters
discount = args.gamma
learning_rate = args.alpha
epsilon = args.epsilon
n_episode = args.episode
batch_size = args.batch

reward_list = []
i_act = []
i_reward = []

model_name = "SARSA-gamma%3.2f-epsilon%3.2f" % (discount, epsilon)

class qfunction:
    # A class to store the coefficents of linear function
    
    def __init__(self, dim_obs, dim_act, rng=None):
        # randomly initialize coefficents
        
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim = dim_obs + dim_act
        # Only take one step when fitting
        self.model = SGDRegressor(penalty='none', learning_rate='constant', eta0=learning_rate, random_state=rng, max_iter=1)
        self.model.coef_ = rng.rand(self.dim)
        self.model.intercept_ = rng.rand(1,)
        print("Model Initialized!")
        
    def __call__(self, obs, act):
        # input states and action, return value of q function
        X = np.concatenate((obs, act))
        res = np.sum(X * self.model.coef_) + np.asscalar(self.model.intercept_)
        #res = self.model.predict(X)
        return res
        
    def get_maxq(self, state_):
        # get maximum of Q(s', a') under given s'
        action_func = lambda x: -self(state_, x)
        action0 = 0.5 * np.ones(self.dim_act) # the center of action space
        res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
        # note: https://en.wikipedia.org/wiki/Sequential_quadratic_programming
        max_q = -res.fun # max Q(s', a')
        return max_q
    
# Initialize Q function
qf = qfunction(dim_obs, dim_act)

# Initialize the dataset:(xdata, ydata)
xdata = np.zeros((batch_size, qf.dim))
ydata = np.zeros((batch_size, ))

action0 = 0.5 * np.ones(qf.dim_act)
for i in range(n_episode):
    # Initialize a new simulation
    state = np.array(env.reset())
    reward = 0
    sum_reward = 0
    gamma_n = 1 # gamma^n
    # Run the simulation until the framework stop it
    done = False
    j = 0 # index of data in batch
    while not done:
        
        # choose action a by state s
        # get the action based on Q function and epsilon greedy
        if (rng.rand() < epsilon) :
            # exploration: randomly choose an action
            action = rng.uniform(action_low, action_high, dim_act)
        else:
            # exploitation: choose the action maximumizing Q function
            action_func = lambda x: -qf(state, x)
            res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
            action = res.x
        '''
        # randomly choose an action
        action = rng.uniform(action_low, action_high, dim_act)
        '''

        # take action a, observe r, s′
        # evolve the system to the next time step
        state_, reward, done, info = env.step(action)
        state_ = np.array(state_)
        
        sum_reward = sum_reward + reward * gamma_n
        gamma_n = gamma_n * discount
        
        ave_action = np.average(action)
        i_act.append(ave_action)
        i_reward.append(reward)
        
        # choose action a' by state s'
        # get the action based on Q function and epsilon greedy
        if (rng.rand() < epsilon) :
            # exploration: randomly choose an action
            action_ = rng.uniform(action_low, action_high, dim_act)
        else:
            # exploitation: choose the action maximumizing Q function
            action_func = lambda x: -qf(state_, x)
            res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
            action_ = res.x
        
        # {s, a} and [r + gamma * Q(s`, a`)]
        xx = np.concatenate((state, action))
        yy = np.array(reward + discount * qf(state_, action_))
        
        # put the data point into data batch
        xdata[j] = xx
        ydata[j] = yy
        
        if (j + 1) == batch_size :
            # Do linear fitting and update Q function coefficients
            qf.model.partial_fit(xdata, ydata)
            # reset count for next data batch
            j = -1
        
        # Update state
        state = state_
        j = j + 1
    print("episode %d, sum_reward %f" % (i, sum_reward))
    reward_list.append(sum_reward)

        
model_coeff = np.hstack((qf.model.coef_, qf.model.intercept_))
np.savetxt(model_name+".csv", model_coeff, delimiter=",")

# plot the cummulative rewards
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(range(n_episode), reward_list)
ax.set_title("Q Learning")
ax.set_xlabel("episode")
ax.set_ylabel("Reward of episode")
fig.tight_layout()
fig.savefig(model_name+"-cr.pdf")

np.savez(model_name+".npz", a=i_act, r=i_reward)