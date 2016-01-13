from bayes import *
from disclosuregame.Util import weighted_random_choice
from random import Random
from math import exp, log
from sharing import *
from heuristic import *


def delta_v(alpha, beta, us, v):
    return alpha * beta * (us - v)


def weighted_choice(choices, weights, random=Random()):
    low = abs(min(weights))
    weights = map(lambda x: x + low, weights)
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices, weights):
        if upto + w > r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def softmax(choices, weights, random=Random(), temp=1000):
    try:
        exped = [exp(act/temp) for act in weights]
    except OverflowError as e:
        LOG.error(e)
        LOG.error(", ".join(map(str, weights)))
        LOG.error(temp)
        raise e
    denom = sum(exped)
    prbs = [x/denom for x in exped]
    return weighted_choice(choices, prbs, random=random)

def linear(choices, weights, random=Random()):
    denom = sum(weights)
    return weighted_choice(choices, [weight/denom for weight in weights], random=random)

class RWSignaller(BayesianSignaller):
    def __init__(self, player_type=1, signals=None, responses=None, signal_alpha=.25, mw_alpha=.3, type_alpha=.3,
                 configural_alpha=.03, beta=.75, seed=None):
        if not responses:
            responses = [0, 1]
        if not signals:
            signals = [0, 1, 2]
        self.signal_alpha = signal_alpha
        self.mw_alpha = mw_alpha
        self.type_alpha = type_alpha
        self.configural_alpha = configural_alpha
        self.beta = beta
        self.v_sig = [0.] * 3
        self.v_type = [0.] * 3
        self.v_mw = {}
        self.v_configural = {}
        self.observed_type = False
        self.low = 0.
        self.diff = 0.
        self.temp = 1000
        super(RWSignaller, self).__init__(player_type, signals, responses, seed=seed)

    def init_payoffs(self, baby_payoffs, social_payoffs, type_weights=None, response_weights=None, num=10):
        """
        An alternative way of generating priors by using the provided weights
        as weightings for random encounters.
        """
        if not response_weights:
            response_weights = [[1., 1.], [1., 1.], [1., 1.]]
        if not type_weights:
            type_weights = [1., 1., 1.]
        self.low = min(min(l) for l in baby_payoffs) + min(min(l) for l in social_payoffs)
        self.diff = float(max(max(l) for l in baby_payoffs) + max(max(l) for l in social_payoffs) - self.low)
        tmp = type(self)()
        for i in xrange(num):
            # Choose a random signal
            signal = self.random.choice(self.signals)
            # A weighted response
            response = weighted_random_choice(self.responses, response_weights[signal], self.random)
            # A weighted choice of type
            player_type = weighted_random_choice(self.signals, type_weights, self.random)
            # Payoffs
            tmp.player_type = player_type
            payoff = baby_payoffs[self.player_type][response] + social_payoffs[signal][player_type]
            self.signal_log.append(signal)
            self.last_v = self.risk(signal, tmp)
            self.update_counts(response, tmp, payoff, player_type)
            self.update_beliefs()
            self.signal_log.pop()
            self.response_log.pop()
            # self.type_log.pop()

    def update_counts(self, response, midwife, payoff, midwife_type=None, weight=1.):
        if response is not None:
            self.response_log.append(response)
        if midwife is not None:
            # Log true type for bookkeeping
            self.type_log.append(midwife.player_type)
            midwife_type = midwife.player_type
        self.last_sig = self.signal_log[-1]
        self.last_mw = hash(midwife)
        self.last_payoff = (payoff - self.low) / self.diff
        # print "Payoff was %d, rescaled it to %f" % (payoff, self.last_payoff)
        self.last_type = midwife_type
        self.update_weight = weight
        self.payoff_log.append(payoff)
        self.last_v = self.risk(self.last_sig, midwife)
        self._update_beliefs()

    def update_beliefs(self):
        pass

    # @profile
    def _update_beliefs(self):
        # Cues
        # Signal
        self.v_sig[self.last_sig] += delta_v(self.signal_alpha * self.update_weight, self.beta, self.last_payoff,
                                             self.last_v)
        # Midwife 
        try:
            self.v_mw[self.last_mw] += delta_v(self.mw_alpha * self.update_weight, self.beta, self.last_payoff,
                                               self.last_v)
            self.v_type[self.last_type] += delta_v(self.type_alpha * self.update_weight, self.beta, self.last_payoff,
                                                   self.last_v)
        except KeyError:
            self.v_mw[self.last_mw] = delta_v(self.type_alpha * self.update_weight, self.beta, self.last_payoff,
                                              self.last_v)
        # Midwife type 
        # Configurals
        try:
            self.v_configural[(self.last_sig, self.last_type)] += delta_v(self.configural_alpha * self.update_weight,
                                                                          self.beta, self.last_payoff, self.last_v)
        except:
            self.v_configural[(self.last_sig, self.last_type)] = delta_v(self.configural_alpha * self.update_weight,
                                                                         self.beta, self.last_payoff, self.last_v)

    def risk(self, signal, opponent=None):
        risk = 0.
        risk += self.v_sig[signal]
        self.observed_type = False
        try:
            self.v_mw[hash(opponent)]
            risk += self.v_type[opponent.player_type]
            self.observed_type = True
            risk += self.v_configural[(signal, opponent.player_type)]
        except:
            pass
        return risk


    def signal_search(self, signals=None, opponent=None):
        if signals is None:
            signals = self.signals
        # print "Type %d woman evaluating signals." % self.player_type
        weights = map(lambda signal: self.risk(signal, opponent), self.signals)
        best = softmax(self.signals, weights, self.random, temp=self.temp)
        # best = (best, weights[best])
        self.last_v = weights[best]
        self.temp = 1/log(1+self.rounds+1e-7)
        return (best, self.last_v)

    def do_signal(self, opponent=None):
       #print "Type %d woman evaluating signals." % self.player_type
        best = self.signal_search(shuffled(self.signals, self.random), opponent=opponent)
        self.rounds += 1
        self.log_signal(best[0])
        return best[0]

class RWResponder(BayesianResponder):
    def __init__(self, player_type=1, signals=None, responses=None, signal_alpha=.3, w_alpha=.3, response_alpha=.3,
                 configural_alpha=.03, beta=.75, seed=None):
        if not responses:
            responses = [0, 1]
        if not signals:
            signals = [0, 1, 2]
        self.signal_alpha = signal_alpha
        self.w_alpha = w_alpha
        self.response_alpha = response_alpha
        self.configural_alpha = configural_alpha
        self.beta = beta
        self.v_sig = [0.] * 3
        self.v_response = [0.] * 2
        self.v_mw = {}
        self.v_configural = {}
        self.observed_type = False
        self.low = 0.
        self.diff = 0.
        self.temp = 1000
        self.last_v = 0.
        super(RWResponder, self).__init__(player_type, signals, responses, seed=seed)

    def init_payoffs(self, payoffs, type_weights=None, num=1000):
        if not type_weights:
            type_weights = [[10., 2., 1.], [1., 10., 1.], [1., 1., 10.]]
        self.type_weights = [[0.] * 3] * 3
        self.low = min(min(l) for l in payoffs)
        self.diff = float(max(max(l) for l in payoffs) - self.low)
        # [map(lambda x: (x - low) / diff, l) for l in payoffs]
        for i in xrange(num):
            signal = self.random.choice(self.signals)
            player_type = weighted_random_choice(self.signals, type_weights[signal])
            # print "Signal is %d, type is %d" % (signal, player_type)
            response = self.random.choice(self.responses)
            self.response_log.append(response)
            payoff = payoffs[player_type][response]
            self.last_v = self.risk(response, signal, None)
            self.update_beliefs(payoff, None, signal, player_type)
            self.response_log.pop()

        # Only interested in payoffs for own type
        self.payoffs = payoffs
        # self.update_beliefs(None, None, None)

    def update_counts(self, response, midwife, payoff, midwife_type=None, weight=1.):
        payoff = (payoff - self.low) / self.diff
        if response is not None:
            self.response_log.append(response)
        if midwife is not None:
            # Log true type for bookkeeping
            self.type_log.append(midwife.player_type)
            midwife_type = midwife.player_type
        self.last_sig = self.signal_log[-1]
        self.last_mw = hash(midwife)
        self.last_payoff = payoff
        self.last_type = midwife_type

    def update_beliefs(self, payoff, signaller, signal, signaller_type=None, weight=1.):
        payoff = (payoff - self.low) / self.diff
        response = self.response_log[-1]
        self.last_v = self.risk(response, signal, opponent=signaller)
        self.v_sig[signal] += delta_v(self.signal_alpha * weight, self.beta, payoff, self.last_v)
        self.v_response[response] += delta_v(self.response_alpha * weight, self.beta, payoff, self.last_v)
        try:
            self.v_configural[(signal, response)] += delta_v(self.configural_alpha * weight, self.beta, payoff,
                                                             self.last_v)
        except:
            self.v_configural[(signal, response)] = delta_v(self.configural_alpha * weight, self.beta, payoff,
                                                            self.last_v)

    def risk(self, act, signal, opponent=None):
        risk = 0.
        risk += self.v_sig[signal]
        risk += self.v_response[act]
        # self.observed_type = False
        try:
            # risk += self.v_mw[hash(opponent)]
            # risk += self.v_type[opponent.player_type]
            # .observed_type = True
            risk += self.v_configural[(signal, act)]
        except:
            pass
        return risk

    def respond(self, signal, opponent=None):
        """
        Make a judgement about somebody based on
        the signal they sent based on expe
        """
        self.signal_log.append(signal)
        # print "Type %d woman evaluating signals." % self.player_type
        weights = map(lambda response: self.risk(response, signal, opponent), self.responses)
        best = softmax(self.responses, weights, self.random, temp=self.temp)
        # best = (best, weights[best])
        self.rounds += 1
        self.last_v = weights[best]
        self.temp = 1/log(1+self.rounds+1e-7)
        # best = (best, weights[best])
        self.last_v = weights[best]
        self.response_log.append(best)
        return best


class SharingRWResponder(SharingResponder, RWResponder):
    """
    A lexicographic reasoner that shares info updates.
    """

    def exogenous_update(self, signal, response, tmp_signaller, payoff, midwife_type=None):
        """
        Update counts from an external source. Counts are weighted according to the agent's
        share_weight attribute.
        """
        self.log_signal(signal, tmp_signaller, self.share_weight)
        self.exogenous.append((tmp_signaller.player_type, signal, response, payoff))
        self.update_counts(response, tmp_signaller, payoff, midwife_type, self.share_weight)
        # Remove from memory, but keep the count
        self.type_log.pop()
        self.signal_log.pop()
        self.response_log.pop()
        self.payoff_log.pop()

class SharingRWSignaller(SharingSignaller, RWSignaller):
    """
    A lexicographic reasoner that shares info updates.
    """

    def exogenous_update(self, payoff, signaller, signal, signaller_type=None):
        """
        Perform a weighted update of the agent's beliefs based on external
        information.
        """
        self.last_v = self.risk(act, signal, opponent=None)
        self.update_beliefs(payoff, signaller, signal, signaller_type, self.share_weight)