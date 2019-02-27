import numpy as np
import gym
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from InterfaceGym import GymEnvironment
from InterfaceOutline import Memory, Agent, Simulation
from BrainDDPG import DDPGBrain

################################################################################

ENV_NAME = 'Pendulum-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1

numActions = env.action_space.shape[0]
numStateParams = env.observation_space.shape[0]

# Next, we build a simple model.

print("Input shape : " + str(env.observation_space.shape))
print("Action shape : " + str(env.action_space.shape))

actorInputs = Input(shape = (numStateParams,))
x = Dense(16, activation='relu')(actorInputs)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
actions = Dense(numActions, activation='linear')(x)
actor = Model(inputs=actorInputs, outputs = actions)

stateInput = Input(shape = (numStateParams,))
actionInput = Input(shape = (numActions,))
x = Concatenate()([stateInput, actionInput])
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
values = Dense(1, activation='linear')(x)
critic = Model(inputs=[stateInput, actionInput], outputs = values)

################################################################################

randomProcess = OrnsteinUhlenbeckProcess(size=numActions, theta=.15, mu=0., sigma=.3)
brainMemory = Memory(50000)

brain = DDPGBrain(actorModel=actor, criticModel=critic, memory = brainMemory, randomProcess = randomProcess, gamma = 0.99,
		targetModelUpdate=1e-3, batchSize = 16)
brain.Compile(optimizer=Adam(lr=1e-4, clipnorm=1.))

agent = Agent(brain)
environment = GymEnvironment(env)
environment.AddAgent(agent)

simulation = Simulation()
simulation.AddEnvironment(environment)
simulation.Train(numSteps = 50000, maxStepsPerEpisode = 200)

brain.SaveWeights('custom_pendulum_ddpg_{}_weights.h5f'.format(ENV_NAME))

hist = simulation.Environments[environment]['history']

episodeLengths = []
errors = []
for ep in hist.Episodes :
	epErr = np.sum(ep.Rewards)
	errors.append(epErr)
	episodeLengths.append(ep.NumSteps)

plt.plot(errors)
plt.show()

################################################################################

"""
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
"""