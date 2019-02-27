import numpy as np
import gym
import matplotlib.pyplot as plt

from time import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess

from InterfaceGym import GymEnvironment
from InterfaceOutline import Memory, Agent, Simulation
from BrainDDPG import DDPGBrain

################################################################################

ENV_NAME = 'GridWorldDriving-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

numActions = env.action_space.shape[0]
numStateParams = env.observation_space.shape[0]

actorInputs = Input(shape = (numStateParams,))
x = Dense(32, activation='relu')(actorInputs)
x = Dense(32, activation='relu')(x)
actions = Dense(numActions, activation='tanh')(x)
actor = Model(inputs=actorInputs, outputs = actions)

stateInput = Input(shape = (numStateParams,))
actionInput = Input(shape = (numActions,))
x = Concatenate()([stateInput, actionInput])
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
values = Dense(1, activation='linear')(x)
critic = Model(inputs=[stateInput, actionInput], outputs = values)

randomProcess = OrnsteinUhlenbeckProcess(size=numActions, theta=.15, mu=0., sigma=.3)
brainMemory = Memory(50000)

brain = DDPGBrain(actorModel=actor, criticModel=critic, memory = brainMemory, randomProcess = randomProcess, gamma = 0.99,
		targetModelUpdate=1e-3, batchSize = 32)
brain.Compile(optimizer=Adam(lr=1e-3, clipnorm=1.))

agent = Agent(brain)
environment = GymEnvironment(env)
environment.AddAgent(agent)

simulation = Simulation()
simulation.AddEnvironment(environment)

#brain.LoadWeights('actorWeights.h5f', 'criticWeights.h5f')

#simulation.Train(numSteps = 200000, maxStepsPerEpisode = 200)

#brain.SaveWeights('actorWeights.h5f', 'criticWeights.h5f')

hist = simulation.Environments[environment]['history']

episodeLengths = []
for ep in hist.Episodes :
	episodeLengths.append(ep.NumSteps)

#plt.plot(hist.GetRewardPerEpisode())
#plt.plot(brain.Memory.Actions)
#plt.show()

################################################################################

def VisualizeCritic(brain, speed, theta, goalx, goaly) : 
	xs = np.arange(-10,10,0.1)
	ys = np.arange(-10,10,0.1)
	states = []
	for x in xs : 
		for y in ys :
			states.append([x, y, speed, theta, goalx, goaly])

	stateArray = np.array(states)
	actions = brain.Actor.predict_on_batch(stateArray)
	criticValues = brain.Critic.predict_on_batch([stateArray, actions])
	criticValues = criticValues.reshape([len(xs), len(ys)])

	xmeshgrid, ymeshgrid = np.meshgrid(xs, ys)

	fig = plt.figure()
	cs = plt.contourf(xmeshgrid, ymeshgrid, criticValues)
	fig.colorbar(cs)
	plt.show()

################################################################################

VisualizeCritic(brain, speed = 4.0, theta = 0.0*np.pi, goalx = 0.0, goaly = 0.0)