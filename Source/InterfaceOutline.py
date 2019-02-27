
# PPO
# DDPG
# Dyna-Q

# UVFA

import numpy as np
import random

################################################################################

class Environment :

	def __init__(self) :
		self.Agents = []
		self.Timestep = 1.0 / 60.0

	def Seed(self, seed=None) :
		pass

	# Should return an initial state
	def ResetInternal(self) :
		raise NotImplementedError()

	def GetNextStateAndReward(self, agent, action) :
		raise NotImplementedError()

	# Should return true if state is terminal
	def IsTerminalState(self, state) : 
		raise NotImplementedError()

	def Render(self, mode ='human') :
		raise NotImplementedError()

	def Close(self) :
		raise NotImplementedError()

	def Reset(self) :
		for agent in self.Agents :
			agent.State = self.ResetInternal()

	def AddAgent(self, agent) :
		self.Agents.append(agent)

	def RemoveAgent(self, agent) :
		self.Agents.remove(agent)

	# Returns true if the environment is empty
	def Step(self) :
		
		# (s_t, a_t, r_t, s_(t+1)) for each agent
		log = {}

		# TODO : Doesn't work for multiple agents in the same environment
		# Will terminate when a single agent reaches a terminal state
		inTerminalState = False

		for agent in self.Agents :
			state = agent.State
			action = agent.Brain.SelectAction(state)
			# 0's are filler for reward and nextState (see next loop)
			agentExperience = [state, action, 0, 0]
			log[agent] = agentExperience

		for agent in log :
			experience = log[agent]
			action = experience[1]
			nextState, reward = self.GetNextStateAndReward(agent, action)
			experience[2] = nextState
			experience[3] = reward
			agent.State = nextState
			sars = np.array(experience).flatten()
			agent.ProcessSARS(sars)

			inTerminalState = self.IsTerminalState(agent.State)

		return inTerminalState

################################################################################

class History : 
	class Episode :
		def __init__(self) : 
			self.Rewards = []
			self.Errors = []
			self.NumSteps = 0
	
	def __init__(self) :
		self.Episodes = []
		self.StartNewEpisode()

	def StartNewEpisode(self) :
		self.Episodes.append(History.Episode())

	def StoreReward(self, reward) :
		self.Episodes[-1].Rewards.append(reward)

	def StoreError(self, error) :
		self.Episodes[-1].Errors.append(error)

	def SetNumberOfSteps(self, numSteps) :
		self.Episodes[-1].NumSteps = numSteps

	def GetRewardPerEpisode(self) :
		rewards = []
		for ep in self.Episodes :
			rewards.append(np.sum(ep.Rewards))
		return rewards
		

################################################################################

class Simulation :
	def __init__(self) :
		self.Environments = {}
		self.Testing = False
		self.Training = False

	def Train(self, numSteps, maxStepsPerEpisode) :
		self.Training = True
		for step in range(numSteps) :
			if (step % (numSteps / 10) == 0) or (step == 0):
				print("Step Num : " + str(step))
			self.Step(maxStepsPerEpisode)
		self.Training = False

	def Test(self, numSteps, maxStepsPerEpisode) :
		self.Testing = True
		for step in range(numSteps) :
			self.Step(maxStepsPerEpisode)
		self.Testing = False

	def AddEnvironment(self, environment) :
		environment.Reset()
		self.Environments[environment] = {'stepNum' : 0, 'history' : History()}

	def Step(self, maxStepsPerEpisode) :

		doneEnvironments = set()

		for environment, environmentData in self.Environments.items() :
			done = (environmentData['stepNum'] >= maxStepsPerEpisode)
			
			if not done :
				environmentData['stepNum'] += 1
				done = environment.Step()
			if done : 
				doneEnvironments.add(environment)

		if self.Testing or self.Training :
			for environment in self.Environments :
				environment.Render()

		trainedBrains = set()
		
		for environment in self.Environments :
			for agent in environment.Agents :
				if agent.Brain not in trainedBrains :
					trainedBrains.add(agent.Brain)
					error = agent.Brain.Train()
					self.Environments[environment]['history'].StoreReward(agent.Brain.Memory.GetMostRecentReward())
					self.Environments[environment]['history'].StoreError(error)

		for environment in doneEnvironments :
			self.Environments[environment]['history'].SetNumberOfSteps(self.Environments[environment]['stepNum'])
			self.Environments[environment]['history'].StartNewEpisode()
			self.Environments[environment]['stepNum'] = 0
			environment.Reset()

################################################################################
# Responsible for learning the value of actions in the environment.
#
class Brain :
	def __init__(self) :
		pass
	
	def Train(self) :
		raise NotImplementedError()

	# Maybe put these in a derived class that requires a NN?
	def Compile(self, optimizer) :
		raise NotImplementedError()

	def SaveWeights(self, filepath) :
		raise NotImplementedError()

	def LoadWeights(self, filepath) :
		raise NotImplementedError()

################################################################################
# An object present in the environment controlled by a brain.
# 
class Agent :
	def __init__(self, brain) :
		self.Brain = brain
		self.State = None # Managed by the environment

	def ProcessSARS(self, sars):
		self.Brain.StoreSARS(sars)

################################################################################

class Memory :
	def __init__(self, size) :
		self.Size = size
		self.Initialized = False
		# Numpy arrays
		self.States = None
		self.NextStates = None
		self.Rewards = None
		self.Actions = None
		# While not full, this is the number of Experiences
		# While full, this is the index of the oldest experience + 1
		self.NumExperiences = 0
		self.Full = False

	# Allocates numpy array using firstExperience as the template
	# firstExperience[0] should be a 1d array of state parameters
	# firstExperience[1] should be a 1d array of action values
	# firstExperience[2] should be a 1d array of state values 
	# firstExperience[3] should be a scalar reward
	#
	def Initialize(self, firstExperience) :
		assert(firstExperience[0].shape == firstExperience[2].shape)
		assert(np.isscalar(firstExperience[3]))
		self.States = np.zeros([self.Size, len(firstExperience[0])])
		self.NextStates = np.zeros([self.Size, len(firstExperience[2])])
		self.Rewards = np.zeros([self.Size, 1])
		self.Actions = np.zeros([self.Size, len(firstExperience[1])])
		self.Initialized = True

	# TODO : Make sure we seed np in the correct place so this is reproducible
	def SampleRandomExperience(self, batchSize = 1) :
		assert(self.Initialized)
		
		if self.Full :
			choices = np.random.choice(self.Size, batchSize)
		else :
			choices = np.random.choice(self.NumExperiences, batchSize)

		return [self.States[choices], self.Actions[choices], self.NextStates[choices], self.Rewards[choices]]

	def GetMostRecentReward(self) :
		assert(self.Initialized)
		return self.Rewards[self.NumExperiences-1]

	def AddExperience(self, experience) :
		if not self.Initialized :
			self.Initialize(experience)
		
		self.States[self.NumExperiences] = experience[0]
		self.NextStates[self.NumExperiences] = experience[2]
		self.Actions[self.NumExperiences] = experience[1]
		self.Rewards[self.NumExperiences] = experience[3]
		self.NumExperiences += 1

		if self.NumExperiences == self.Size :
			self.Full = True
			self.NumExperiences = 0