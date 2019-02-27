import InterfaceOutline
import tensorflow as tf
import keras.backend as K
import numpy as np 

from InterfaceOutline import Brain
from rl.util import *

################################################################################
# Loss function for actor and critic
# 

# Critic input = actions, state
# Critic output = value
# Critic loss function requires s_i, a_i, targetCritic(s_i+1, a_i+1)
#	- Embed targetCritic(...) into y_true, the rest is y_pred

def CriticLossFunc(y_true, y_pred) :
	return tf.reduce_mean(tf.square(y_true - y_pred))

################################################################################

class DDPGBrain(Brain) :
	def __init__(self, actorModel, criticModel, memory, randomProcess, gamma, targetModelUpdate, batchSize) :
		# Call base class ctor
		self.Actor = actorModel
		self.Critic = criticModel
		self.Memory = memory
		self.RandomProcess = randomProcess
		self.Gamma = gamma
		self.BatchSize = batchSize
		self.TargetActorUpdate = 0.1*targetModelUpdate
		self.TargetCriticUpdate = targetModelUpdate
		self.TargetActor = clone_model(self.Actor)
		self.TargetCritic = clone_model(self.Critic)
		self.ActorTrainFunc = None

	def Train(self) :
		# Don't train if there isn't enough data yet
		if (not self.Memory.Full and self.Memory.NumExperiences < self.BatchSize) :
			return 0.0

		sarsBatch = self.Memory.SampleRandomExperience(batchSize = self.BatchSize)
		actorError = self.UpdateActor(sarsBatch)
		criticError = self.UpdateCritic(sarsBatch)

		self.UpdateTargetNetworkWeights()
		
		# TODO : Actor error
		return criticError

	def StoreSARS(self, sars) :
		self.Memory.AddExperience(sars)

	def UpdateActor(self, sarsBatch) : 
		states, _, _, _ = sarsBatch
		# A little unclear why this needs an extra []
		return self.ActorTrainFunc([states])

	def UpdateCritic(self, sarsBatch) : 
		# TODO !!! make sars actually sars rather than sasr
		states, actions, nextStates, rewards  = sarsBatch
		targetActions = self.TargetActor.predict_on_batch(nextStates)
		targetValues = self.TargetCritic.predict_on_batch([nextStates, targetActions])
		yi = rewards + self.Gamma * targetValues
		return self.Critic.train_on_batch(x = [states, actions], y = yi)

	# Used externally, should return [actions]
	def SelectAction(self, state) :
		if (len(state.shape) == 1) :
			a = np.zeros([1, state.shape[0]])
			a[0] = state
			state = a
		action = self.Actor.predict_on_batch(state)
		if self.RandomProcess != None :
			action += self.RandomProcess.sample()
		return action[0]
	
	def Compile(self, optimizer) :
		# Actor does not update through keras loss function
		self.Actor.compile(optimizer=optimizer, loss='mse')
		self.Critic.compile(optimizer=optimizer, loss=CriticLossFunc)
		# Neither target network will use its optimizer or loss function
		self.TargetActor.compile(optimizer=optimizer, loss='mse')
		self.TargetCritic.compile(optimizer=optimizer, loss=CriticLossFunc)

		# TODO : This is some janky shit
		# Setup Actor loss function
		# Critic's input is [state, action]
		#
		actorInputs = []
		combinedInputs = []
		for i in self.Critic.input :
			if i == self.Critic.input[1]:
				combinedInputs.append([])
			else:
				combinedInputs.append(i)
				actorInputs.append(i)
		print("")
		print("Combined inputs : " + str(combinedInputs))
		print("Critic outputs : " + str(self.Actor(actorInputs)))
		combinedInputs[1] = self.Actor(actorInputs[0])
		print("Combined inputs 2 : " + str(combinedInputs))
		print("")

		criticOutputs = self.Critic(combinedInputs)
		
		actorUpdates = self.Actor.optimizer.get_updates(params=self.Actor.trainable_weights, loss=-tf.reduce_mean(criticOutputs))
		# TODO : actorError? Would prefer to return the error than self.Actor(state)
		self.ActorTrainFunc = K.function(inputs = actorInputs, outputs = [self.Actor(actorInputs)], updates = actorUpdates)

	def UpdateTargetNetworkWeights(self) : 
		self.TargetActor.set_weights(self.TargetActorUpdate * np.array(self.Actor.get_weights()) + (1 - self.TargetActorUpdate) * np.array(self.TargetActor.get_weights()))
		self.TargetCritic.set_weights(self.TargetCriticUpdate * np.array(self.Critic.get_weights()) + (1 - self.TargetCriticUpdate) * np.array(self.TargetCritic.get_weights()))

	# TODO : Might be better to save target model separately
	# Resetting target to model correlates them tightly (unstable?)
	def SaveWeights(self, actorWeights, criticWeights) :
		self.Actor.save_weights(actorWeights, overwrite = True)
		self.Critic.save_weights(criticWeights, overwrite = True)

	def LoadWeights(self, actorWeights, criticWeights) : 
		self.Actor.load_weights(actorWeights)
		self.Critic.load_weights(criticWeights)
		self.TargetActor.set_weights(self.Actor.get_weights())
		self.TargetCritic.set_weights(self.Critic.get_weights())

################################################################################