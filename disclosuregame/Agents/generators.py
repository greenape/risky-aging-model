from random import Random
from disclosuregame.Util import random_expectations

class AgentGenerator(object):
	"""
	A machine that produces new agents to fit some distribution of types.
	"""

	def __init__(self, constructor, payoffs, agent_args={}, type_distribution=[1/3., 1/3., 1/3.]):
		self.type_distribution = type_distribution
		self.constructor = constructor
		self.agent_args = agent_args
		self.game = payoffs

	def generator(self, random=None):
		"""
		A generator that yields a player with random beliefs, according to the type distribution.
		"""
		if random is None:
			random = Random()
		while True:
			draw = random.random()
			bracket = 0.
			for i in range(len(self.type_distribution)):
				bracket += self.type_distribution[i]
				if draw < bracket:
					break
			agent = constructor(player_type=i, seed=random.random(), **self.agent_args)
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
	Generator that produces responders with uniformly fixed beliefs.
	"""

	def __init__(self, constructor, payoffs, type_weights=[[10., 1., 1.], [1., 10., 1.], [1., 1., 10.]], **kwargs):
		super(ResponderGenerator, self).__init__(constructor, payoffs, **kwargs)
		self.type_weights = type_weights

	def init(self, agent):
		agent.init_payoffs(self.game.midwife_payoff)
		agent.init_beliefs(self.type_weights)
