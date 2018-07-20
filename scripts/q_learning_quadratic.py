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
learning_rate = 1e-2
epsilon = 0.1
episode = 1000
batch_size = 10

class qfunction:
    # random initialization
    def __init__(self, dim_obs, dim_act, rng=None):
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim = (dim_obs + dim_act + 1) * (dim_obs + dim_act) // 2
        self.model = SGDRegressor(penalty='none', learning_rate='constant', eta0=learning_rate, random_state=rng, max_iter=1, verbose=0)
        self.model.coef_ = rng.rand(self.dim)
        self.model.intercept_ = rng.rand(1,)

    def __call__(self, obs, act):
        input_vec = np.concatenate((obs,act))
        X = quadartic(input_vec)
        res = np.sum(X * self.model.coef_) + np.asscalar(self.model.intercept_)
        return res

    def get_maxq(self, state_):
        # get maximum of Q(s', a') under given s'
        action_func = lambda x: -self(state_, x)
        action0 = 0.5 * np.ones(self.dim_act) # the center of action space
        res = minimize(action_func, action0, method='SLSQP', bounds=bnds)
        # note: https://en.wikipedia.org/wiki/Sequential_quadratic_programming
        max_q = -res.fun # max Q(s', a')
        return max_q
        
def quadartic(vec):
    # convert (x1, x2, x3, ...) to (x1^2, x1x2, x1x3, ..., x2^2, x2x3, ...)
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    res = []
    for i, x in enumerate(vec):
        res = np.concatenate((res, x*vec[i:]))
    return res

# Initialize Q function
qf = qfunction(dim_obs, dim_act)

# Initialize the dataset:(xdata, ydata)
xdata = np.zeros((batch_size, qf.dim))
ydata = np.zeros((batch_size, ))

action0 = 0.5 * np.ones(qf.dim_act)
for i in range(episode):
    print("episode %d start!" % (i))
    # Initialize a new simulation
    state = np.array(env.reset())
    reward = 0
    # Run the simulation until the framework stop it
    done = False
    j = 0 # index of data in batch
    while not done:
        
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

        # evolve the system to the next time step
        state_, reward, done, info = env.step(action)
        state_ = np.array(state_)

        max_q = qf.get_maxq(state_)
        
        # {s, a} and [r + gamma * max_a` Q(s`, a`)]
        xx = quadartic(np.concatenate((state, action)))
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
np.savetxt("model_qq.csv", model_coeff, delimiter=",")
