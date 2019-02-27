import gym
from InterfaceOutline import Environment

################################################################################
# The assumption with Gym environments is that there will only 
# ever be 1 agent in any given environment.
#
class GymEnvironment(Environment) :
	def __init__(self, gymEnv) :
		super().__init__()
		self.GymEnv = gymEnv
		self.TerminalState = False

	def GetNextStateAndReward(self, agent, action) :
		observation, reward, done, _ = self.GymEnv.step(action)
		if done : 
			self.TerminalState = True
		return observation, reward

	# To conform to base interface (not a great strategy but should work)
	# - It's not great because if reset returns a terminal state we will still
	#   try to take a step from there (I think the gym environment has this flaw).
	def IsTerminalState(self, state) : 
		return self.TerminalState

	def ResetInternal(self) :
		self.TerminalState = False
		return self.GymEnv.reset()

	def Render(self, mode ='human') :
		self.GymEnv.render(mode)

	def Close(self) : 
		self.GymEnv.close()

################################################################################