# MSc Project: AI Learning to run

This project is to use OpenSim framework to teach a robot to run. It will use reinforcement learning solution to let the robot walk as far as possible.

Related Pages:
https://www.crowdai.org/challenges/nips-2017-learning-to-run
https://github.com/stanfordnmbl/osim-rl

## Environment Building

Use conda to create a virtual environment:
`conda create -n opensim-rl -c kidzik opensim
conda install -c conda-forge lapack git
pip install git+https://github.com/stanfordnmbl/osim-rl.git`

Reinforcement Learning packages perhaps needed
`conda install keras -c conda-forge
pip install git+https://github.com/matthiasplappert/keras-rl.git`