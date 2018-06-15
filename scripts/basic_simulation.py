
# coding: utf-8

# Basic Simulation
# ===
# At the very first beginning, we used OpenSim Framework to simulate body motion in the environment `L2RunEnv`, which is the environment for the machine learning problem.
# The framework outputs 41 observation variables in each step, including coordinates and velocities of points on the body, and receives 18 action values associated with 18 muscles for the next step. In this demo, we recorded all observations and actions, as well as the reward value given by the framework.

# In[1]:


from osim.env import L2RunEnv
import numpy as np
import pandas as pd

SEED = 20180101
rng = np.random.RandomState(SEED)

env = L2RunEnv(visualize=True)
# Obtain the dimension observation space and action space
dim_obs = env.get_observation_space_size()
dim_act = env.get_action_space_size()


# In[2]:


obs_desc = ['joint_pos_ground_pelvis_1','joint_pos_ground_pelvis_2','joint_pos_ground_pelvis_3',
            'joint_vel_ground_pelvis_1','joint_vel_ground_pelvis_2','joint_vel_ground_pelvis_3',
            'joint_pos_hip_l', 'joint_vel_hip_l', 'joint_pos_hip_r', 'joint_vel_hip_r',
            'joint_pos_knee_l', 'joint_vel_knee_l', 'joint_pos_knee_r', 'joint_vel_knee_r',
            'joint_pos_ankle_l', 'joint_vel_ankle_l', 'joint_pos_ankle_r', 'joint_vel_ankle_r',
            'body_pos_head_1', 'body_pos_head_2', 
            'body_pos_pelvis_1', 'body_pos_pelvis_2', 
            'body_pos_torso_1', 'body_pos_torso_2', 
            'body_pos_toes_l_1', 'body_pos_toes_l_2','body_pos_toes_r_1', 'body_pos_toes_r_2',
            'body_pos_talus_l_1', 'body_pos_talus_l_2','body_pos_talus_r_1', 'body_pos_talus_r_2',
            'mass_center_pos_1', 'mass_center_pos_2', 'mass_center_vel_1', 'mass_center_vel_2',
            'misc_1', 'misc_2', 'misc_3', 'misc_4', 'misc_5']
act_desc = ['action'+str(i) for i in range(1,dim_act+1)]
columns_label=columns=obs_desc+act_desc+['reward']


# The problem is to obtain a policy to let the body act based on the observation. In the demo, we do not need to get a specific policy but to show how the simulation works. Henceforth, we used an initially randomized matrix representing a transformation from observations to actions:
# $$A_{i}=P_{ij}\cdot O_{j}$$
# where $A_{i}$ is the $i$-th action variable, $P_{ij}$ is the policy matrix component, and $O_{j}$ is the $j$-th observation value. We can obtain each action under a specific set of observations by the policy.
# The framework can give the reward and job-done signal in the simulation and we recorded them for each step.

# In[3]:


# Record the observations and actions for each step
df = pd.DataFrame(data=[], columns=columns_label)
# Start a new simulation
observation = env.reset()
# Initialize the policy matrix
policy_mat = rng.rand(dim_act,dim_obs)
# At the first step, set the reward to zero
reward = 0
# Run the simulation until the framework stop it
done = False
while not done:
    # get the action based on the policy
    action = np.dot(policy_mat, np.array(observation))
    # record current observations and its associated actions
    logging_row = pd.DataFrame(data=[observation+action.tolist()+[reward]], columns=columns_label)
    df = df.append(logging_row, ignore_index=True)
    # evolve the system to the next time stamp
    observation, reward, done, info = env.step(action)
df.to_csv("basic_sim.csv")

