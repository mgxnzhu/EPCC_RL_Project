from osim.env import L2RunEnv
import numpy as np
from scipy.optimize import minimize, Bounds
from sklearn.linear_model import LinearRegression, SGDRegressor

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

# Set hyperparameters
discount = 1e-1
learning_rate = 1e-3
#epsilon = 1
episode = 200
batch_size = 10


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
        xdata = rng.rand(batch_size, self.dim)
        ydata = rng.rand(batch_size, )
        self.model.partial_fit(xdata, ydata)
        print("Model Initialized!")
        # Since SGDRegressor does not provide a method to feed the coefficients, which is needed in the beginning to calculate Q function value, used random dataset to fit the model to generate a set of coefficients 
        
    def __call__(self, obs, act):
        # input states and action, return value of q function
        X = np.concatenate((obs, act))
        res = np.sum(X * qf.model.coef_) + np.asscalar(qf.model.intercept_)
        #res = self.model.predict(X)
        return res
        
    def get_maxq(self, state_):
        # get maximum of Q(s', a') under given s'
        action_func = lambda x: -qf(state_, x)
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

for i in range(episode):
    # Initialize a new simulation
    state = np.array(env.reset())
    reward = 0
    # Run the simulation until the framework stop it
    done = False
    j = 0 # index of data in batch
    while not done:
        '''
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
        '''
        # randomly choose an action
        action = rng.uniform(action_low, action_high, dim_act)

        # evolve the system to the next time step
        state_, reward, done, info = env.step(action)
        state_ = np.array(state_)

        max_q = qf.get_maxq(state_)
        
        # {s, a} and [r + gamma * max_a` Q(s`, a`)]
        xx = np.concatenate((state, action))
        yy = np.array(reward + discount * max_q)
        
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
        
model_coeff = np.hstack((qf.model.coef_, qf.model.intercept_))
np.savetxt("model.csv", model_coeff, delimiter=",")