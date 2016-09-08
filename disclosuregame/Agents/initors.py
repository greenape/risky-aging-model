from random import Random
from disclosuregame.Util import random_expectations, logistic_random, multinomial


def signaller(agent, woman_baby_payoff=None, woman_social_payoff=None):
    """
	Initiate a signaller with uniformly random beliefs.
	"""
    if not woman_social_payoff:
        woman_social_payoff = []
    if not woman_baby_payoff:
        woman_baby_payoff = []
    agent.init_payoffs(woman_baby_payoff, woman_social_payoff,
                       random_expectations(random=agent.random),
                       [random_expectations(breadth=2, random=agent.random) for x in range(3)])


def logistic_stigma(location, scale, random, prior_weight=0.1):
    x = logistic_random(location, scale, random=random)

    type_weights = [prior_weight] * 2  # ,prior_weight+x*-prior_weight]
    if x > 0:
        type_weights[0] += x * prior_weight
    else:
        type_weights[1] += -x * prior_weight
    return type_weights


def ebreferral_logisticstigma(agent, woman_baby_payoff=None, woman_social_payoff=None, referral_beliefs=None,
                              location=0.5025573, scale=0.0630946, prior_weight=0.1):
    """
	Initiate a signaller with Eurobarometer style multinomially distributed belief in referral,
	and logistic distribution beliefs on type distribution drawn from ESS. Both priors are multiplied by a
	constant with a lower bound of of that constant, representing the weight of the prior.

	Agent may have a 3:1, or 2:1 bias in favour of or against their being referred.
	"""
    if not referral_beliefs:
        referral_beliefs = [0.26, 0.491, 0.171, 0.074]
    if not woman_social_payoff:
        woman_social_payoff = []
    if not woman_baby_payoff:
        woman_baby_payoff = []
    type_weights = logistic_stigma(location, scale, agent.random, prior_weight=prior_weight)

    referral_category = multinomial(referral_beliefs, random=agent.random)

    referral_weights = [[prior_weight] * 2, [prior_weight] * 2]
    if referral_category[0] == 1:
        referral_weights[0][1] *= 3
        referral_weights[1][1] *= 3
    elif referral_category[1] == 1:
        referral_weights[0][1] *= 2
        referral_weights[1][1] *= 2
    elif referral_category[2] == 1:
        referral_weights[0][0] *= 2
        referral_weights[1][0] *= 2
    elif referral_category[3] == 1:
        referral_weights[0][0] *= 3
        referral_weights[1][0] *= 3

    agent.init_payoffs(woman_baby_payoff, woman_social_payoff, type_weights, referral_weights)


def onsreferral_logisticstigma(agent, woman_baby_payoff=None, woman_social_payoff=None, referral_beliefs=None,
                               location=0.5025573, scale=0.0630946, prior_weight=0.1):
    """
	Initiate a signaller with an ONS style multinomially distributed belief in referral,
	and logistic distribution beliefs on type distribution drawn from ESS. Both priors are multiplied by a
	constant with a lower bound of of that constant, representing the weight of the prior.
	"""
    if not referral_beliefs:
        referral_beliefs = [0.26357, 0.14288, 0.59355]
    if not woman_social_payoff:
        woman_social_payoff = []
    if not woman_baby_payoff:
        woman_baby_payoff = []
    type_weights = logistic_stigma(location, scale, agent.random, prior_weight=prior_weight)

    referral_category = multinomial(referral_beliefs, random=agent.random)

    referral_weights = [[prior_weight] * 2, [prior_weight] * 2]
    if referral_category[1] == 1:
        referral_weights[0][1] *= 2
        referral_weights[1][1] *= 2
    elif referral_category[2] == 1:
        referral_weights[0][0] *= 2
        referral_weights[1][0] *= 2

    agent.init_payoffs(woman_baby_payoff, woman_social_payoff, type_weights, referral_weights)


def responder(agent, midwife_payoff=None, type_weights=None):
    """
	Set up a responder with fixed payoffs and beliefs.
	"""
    if not type_weights:
        type_weights = [[10., 2., 1.], [1., 10., 1.], [1., 1., 10.]]
    if not midwife_payoff:
        midwife_payoff = []
    agent.init_payoffs(midwife_payoff, type_weights)


def normalresponder(agent, midwife_payoff=None, mean=0.0007080957, sd=0.9116370106, prior_weight=0.1):
    """
	Set up a responder with fixed payoffs and normally distributed beliefs on types,
	drawn from ESS.
	"""
    if not midwife_payoff:
        midwife_payoff = []
    x = -agent.random.normalvariate(mean, sd)
    if x > 0:
        type_weights = [[prior_weight + x * prior_weight, prior_weight],
                        [prior_weight, prior_weight + x * prior_weight]]
    else:
        type_weights = [[prior_weight, prior_weight + x * -prior_weight],
                        [prior_weight + x * -prior_weight, prior_weight]]
    responder(agent, midwife_payoff, type_weights)
