from random import Random
from disclosuregame.Util import random_expectations

class AgentGenerator(object):
	"""
	A machine that produces new agents to fit some distribution of types.
	"""

	def __init__(self, constructor, agent_args, payoffs, type_distribution=[1/3., 1/3., 1/3.], random=Random()):
		self.type_distribution = type_distribution
		self.random = random
		self.constructor = constructor
		self.agent_args = agent_args
		self.game = payoffs

	def generator(self):
		"""
		A generator that yields a player with random beliefs, according to the type distribution.
		"""
		while True:
			draw = self.random.random()
			bracket = 0.
			for i in range(len(self.type_distribution)):
				bracket += self.type_distribution[i]
				if draw < bracket:
					break
			agent = constructor(player_type=i, seed=self.random.random(), **self.agent_args)
			self.init(agent)
			yield agent

	def init(self, agent):
		"""
		Set up agent's beliefs (does nothing).
		"""
		return None


class SignallerGenerator(AgentGenerator):
	"""
	Generator that produces signallers with uniformly random beliefs.
	"""

	def init(self, agent):
		agent.init_payoffs(self.game.woman_baby_payoff, self.game.woman_social_payoff)
		agent.init_beliefs(random_expectations(random=self.random), [random_expectations(breadth=2, random=self.random) for x in range(3)])


class ResponderGenerator(AgentGenerator):
	"""
	Generator that produces responders with uniformly random beliefs.
	"""

	def init(self, agent):
		agent.init_payoffs(self.game.midwife_payoff)
		agent.init_beliefs(self.game.type_weights)
