from bayes import *
from disclosuregame.Util import weighted_random_choice, rescale
from random import Random
from math import exp, log
from sharing import *
from heuristic import *

try:
    import scoop
    scoop.worker
    scoop_on = True
    LOG = scoop.logger
except:
    scoop_on = False
    import multiprocessing
    LOG = multiprocessing.get_logger()
    pass

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
    try:
        assert False
    except:
        LOG.error("Fell out the bottom of weighted choice.")
        LOG.error(weights)
        raise AssertionError





def softmax(choices, weights, random=Random(), temp=1.):
    exped = []
    K = max(weights)
    for i, act in enumerate(weights):
        try:
            exped.append(exp((act - K)/temp))
        except (ZeroDivisionError, OverflowError) as e:
            LOG.debug(e)
            LOG.debug(", ".join(map(str, weights)))
            LOG.debug(temp)
            return choices[i]
        except Exception as e:
            LOG.error(e)
            raise e
    denom = sum(exped)
    try:
        assert denom > 0
        prbs = [x/denom for x in exped]
        return weighted_choice(choices, prbs, random=random)
    except Exception as e:
        LOG.error(e)
        LOG.error(denom)
        LOG.error(weights)
        LOG.error(temp)
        LOG.error(exped)
        raise e

def linear(choices, weights, random=Random()):
    denom = sum(weights)
    return weighted_choice(choices, [weight/denom for weight in weights], random=random)

class RWSignaller(BayesianSignaller):
    def __init__(self, player_type=1, signals=None, responses=None, signal_alpha=.3, mw_alpha=.3, type_alpha=.3,
                 configural_alpha=.09, beta=.95, seed=None):
        if not responses:
            responses = [0, 1]
        if not signals:
            signals = [0, 1, 2]
        self.signal_alpha = signal_alpha
        self.mw_alpha = mw_alpha
        self.type_alpha = type_alpha
        self.configural_alpha = configural_alpha
        self.beta = beta
        self.v_sig = [0.] * len(signals)
        self.v_type = [0.] * len(signals)
        self.v_mw = {}
        self.v_configural = {}
        self.observed_type = False
        self.low = 0.
        self.diff = 0.
        self.temp = 1/log(2+1e-7)
        super(RWSignaller, self).__init__(player_type, signals, responses, seed=seed)

    def init_payoffs(self, baby_payoffs, social_payoffs, type_weights=None, response_weights=None, num=100):
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
        LOG.debug("Starting init for player {}".format(self.ident))
        for i in xrange(num):
            # Choose a random signal
            signal = self.random.choice(self.signals)
            LOG.debug("Chose signal {}".format(signal))
            # A weighted response
            LOG.debug("Choosing response over {}, weighted by {}".format(self.responses, response_weights[signal]))
            response = weighted_choice(self.responses, response_weights[signal], self.random)
            LOG.debug("Chose response {}".format(response))
            # A weighted choice of type
            LOG.debug("Choosing player over {}, weighted by {}".format(self.signals, type_weights))
            player_type = weighted_choice(self.signals, type_weights, self.random)
            LOG.debug("Chose player type {}".format(player_type))
            # Payoffs
            tmp.player_type = player_type
            LOG.debug("Getting payoff.")
            payoff = baby_payoffs[self.player_type][response] + social_payoffs[signal][player_type]
            LOG.debug("Ready to update init values for run {} of {}".format(i, num))
            self.signal_log.append(signal)
            self.v_mw[hash(tmp)] = 0.
            self.last_v = self.risk(signal, tmp)
            self.update_counts(response, tmp, payoff, player_type)
            self.update_beliefs()
            self.signal_log.pop()
            self.response_log.pop()
            self.v_mw.pop(hash(tmp), None)
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
        self.last_payoff = rescale(-1, 1, self.low, self.diff, payoff)
        LOG.debug("Payoff was {}, rescaled it to {}".format(payoff, self.last_payoff))
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
        return (best, self.last_v)

    def do_signal(self, opponent=None):
       #print "Type %d woman evaluating signals." % self.player_type
        best = self.signal_search(shuffled(self.signals, self.random), opponent=opponent)
        self.rounds += 1
        self.log_signal(best[0])
        self.temp /= 2. #1/log(2+self.rounds+1e-7)
        return best[0]

class RWResponder(BayesianResponder):
    def __init__(self, player_type=1, signals=None, responses=None, signal_alpha=.3, w_alpha=.3, response_alpha=.3,
                 configural_alpha=.3, beta=.75, seed=None):
        if not responses:
            responses = [0, 1]
        if not signals:
            signals = [0, 1, 2]
        self.signal_alpha = signal_alpha
        self.w_alpha = w_alpha
        self.response_alpha = response_alpha
        self.configural_alpha = configural_alpha
        self.beta = beta
        self.v_sig = [0.] * len(signals)
        self.v_response = [0.] * len(responses)
        self.v_mw = {}
        self.v_configural = {}
        self.observed_type = False
        self.low = 0.
        self.diff = 0.
        self.temp = 1/log(2+1e-7)
        self.last_v = 0.
        super(RWResponder, self).__init__(player_type, signals, responses, seed=seed)

    def init_payoffs(self, payoffs, type_weights=None, num=100):
        if not type_weights:
            type_weights = [[10., 2., 1.], [1., 10., 1.], [1., 1., 10.]]
        self.type_weights = [[0.] * len(type_weights[0])] * len(type_weights)
        self.low = min(min(l) for l in payoffs)
        self.diff = float(max(max(l) for l in payoffs) - self.low)
        # [map(lambda x: (x - low) / diff, l) for l in payoffs]
        for i in xrange(num):
            signal = self.random.choice(self.signals)
            player_type = weighted_choice(self.signals, type_weights[signal])
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
        payoff = rescale(-1, 1, self.low, self.diff, payoff)
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
        payoff = rescale(-1, 1, self.low, self.diff, payoff)
        self.last_sig = signal
        self.last_payoff = payoff
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
        """

        :param act:
        :param signal:
        :param opponent:
        :return:
        """
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
        self.temp /= 2.
        # best = (best, weights[best])
        self.last_v = weights[best]
        self.response_log.append(best)
        return best


class SharingRWResponder(SharingResponder, RWResponder):
    """
    A lexicographic reasoner that shares info updates.
    """

class SharingRWSignaller(SharingSignaller, RWSignaller):
    """
    A lexicographic reasoner that shares info updates.
    """

    def exogenous_update(self, signal, response, responder, payoff, responder_type=None):
        """
        Perform a weighted update of the agent's beliefs based on external
        information.
        """
        self.last_v = self.risk(signal, opponent=responder)
        super(SharingRWSignaller, self).exogenous_update(signal, response, responder, payoff, responder_type)