from bayes import *
from recognition import RecognitionResponder
from sharing import *
from disclosuregame.Util import random_expectations
import operator
from random import Random


def weighted_random_choice(choices, weights, random=Random()):
    population = [val for val, cnt in zip(choices, weights) for i in range(int(cnt))]
    return random.choice(population)

class LexicographicSignaller(BayesianSignaller):
    """
    A signaller that uses the Lexicographic heuristic to make decisions.
    """

    def __str__(self):
        return "lexicographic"

    # @profile
    def init_payoffs(self, baby_payoffs, social_payoffs, type_weights=None, response_weights=None):
        # Payoff counter
        if not response_weights:
            response_weights = [[1., 1.], [1., 1.], [1., 2.]]
        if not type_weights:
            type_weights = [1., 1., 1.]
        self.payoff_count = dict([(signal, {}) for signal in self.signals])
        self.payoff_belief = dict([(signal, {}) for signal in self.signals])

        self.baby_payoffs = baby_payoffs[self.player_type]
        try:
            self.baby_payoffs[0][0]
        except:
            self.baby_payoffs = [self.baby_payoffs for x in self.signals]

        for signal in self.signals:
            self.payoff_count[signal] = {}
            for response in self.responses:
                for player_type in self.signals:
                    payoff = self.baby_payoffs[signal][response] + social_payoffs[signal][player_type]
                    self.payoff_count[signal][payoff] = 0.

        self.depth = 0
        for signal, payoffs in self.payoff_count.items():
            self.depth = max(len(payoffs), self.depth)
        # Psuedocounts go in
        # for signal in self.signals:
        #    breadth = len(self.payoff_count[signal])
        #    weights = random_expectations(breadth=breadth, high=self.random.randint(breadth, 10), random=self.random)
        #    i = 0
        #    for payoff in self.payoff_count[signal].keys():
        #        self.payoff_count[signal][payoff] = weights[i]
        #        i += 1
        for signal in self.signals:
            for response in self.responses:
                for player_type in self.signals:
                    payoff = self.baby_payoffs[signal][response] + social_payoffs[signal][player_type]
                    self.payoff_count[signal][payoff] += type_weights[player_type] + response_weights[signal][response]
        super(LexicographicSignaller, self).init_payoffs(baby_payoffs, social_payoffs, type_weights,
                                                         response_weights)

    def set_uninformative_prior(self, weight=1):
        """
        Set uninformative prior (i.e. everything to 1).
        :return:
        """
        for signal in self.signals:
                for payoff, counts in self.payoff_count[signal].iteritems():
                    self.payoff_count[signal][payoff] = 1*weight

    def init_payoffs_(self, baby_payoffs, social_payoffs, type_weights=None, response_weights=None, num=10):
        """
        An alternative way of generating priors by using the provided weights
        as weightings for random encounters.
        """
        if not response_weights:
            response_weights = [[1., 1.], [1., 1.], [1., 1.]]
        if not type_weights:
            type_weights = [1., 1., 1.]
        # Payoff counter
        self.payoff_count = dict([(signal, {}) for signal in self.signals])
        self.payoff_belief = dict([(signal, {}) for signal in self.signals])
        for signal in self.signals:
            for payoff in social_payoffs[signal]:
                for baby_payoff in baby_payoffs[self.player_type]:
                    self.payoff_count[signal][payoff + baby_payoff] = 0
                    self.payoff_belief[signal][payoff + baby_payoff] = 0.
        self.depth = 0
        for signal, payoffs in self.payoff_count.items():
            self.depth = max(len(payoffs), self.depth)
        for i in xrange(num):
            # Choose a random signal
            signal = self.random.choice(self.signals)
            # A weighted response
            response = weighted_random_choice(self.responses, response_weights[signal], self.random)
            # A weighted choice of type
            player_type = weighted_random_choice(self.signals, type_weights, self.random)
            # Payoffs
            payoff = baby_payoffs[self.player_type][response] + social_payoffs[signal][player_type]
            self.payoff_count[signal][payoff] += 1
        super(LexicographicSignaller, self).init_payoffs(baby_payoffs, social_payoffs, type_weights,
                                                         response_weights)

    def freq_it(self, signal):
        """
        Return a generator over outcomes experienced, ordered by how often they've been seen.
        Sticks on the least frequent rather than ending.
        """
        shuffled_pairs = shuffled(self.payoff_count[signal].items(), self.random)
        sorted_dict = sorted(shuffled_pairs, key=operator.itemgetter(1), reverse=True)
        n = 0
        while True:
            yield sorted_dict[min(n, len(sorted_dict) - 1)][0]
            n += 1

    def frequent(self, signal, n):
        """
        Return the nth most frequently experienced outcome from
        this signal.
        """
        sorted_dict = sorted(self.payoff_count[signal].items(), key=operator.itemgetter(1), reverse=True, cmp=self.compare_with_ties)
        return sorted_dict[min(n, len(sorted_dict) - 1)][0]

    def update_counts(self, response, midwife, payoff, midwife_type=None, weight=1.):
        if payoff is not None:
            try:
                self.payoff_count[self.signal_log[len(self.signal_log) - 1]][payoff] += weight
            except KeyError:
                # Must be an impossible payoff add it anyway?
                self.payoff_count[self.signal_log[len(self.signal_log) - 1]][payoff] = weight
            self.payoff_log.append(payoff)
        if response is not None:
            self.response_log.append(response)
        if midwife is not None:
            # Log true type for bookkeeping
            self.type_log.append(midwife.player_type)
        for signal, payoffs in self.payoff_count.items():
            self.depth = max(len(payoffs), self.depth)

    # @profile
    def update_beliefs(self):
        return None
        # self.update_counts(response, midwife, payoff, midwife_type, weight)

    def signal_search(self, signals=None):
        if signals is None:
            signals = self.signals
        n = 0
        sigits = {}
        for signal in signals:
                sigits[signal] = self.freq_it(signal)
        while n < self.depth:
            mappings = {}
            # N most frequent outcome of each signal
            for signal in signals:
                payoff = sigits[signal].next()
                mappings[signal] = payoff
            n += 1
            # Choose the highest unless there's a tie
            sorted_mappings = sorted(shuffled(mappings.items(), self.random), key=operator.itemgetter(1), reverse=True)
            # Is there a best option?
            best = sorted_mappings[0]
            try:
                if sorted_mappings[0][1] > sorted_mappings[1][1]:
                    break
                    # Remove any worse than the tied pair.
                    # if sorted_mappings[1][1] > sorted_mappings[2][1]:
                    #    print "removing", sorted_mappings[2][0]
                    #    signals.remove(sorted_mappings[2][0])
            except IndexError:
                pass
        return best


class LexicographicResponder(BayesianResponder):
    def __str__(self):
        return "lexicographic"

    # @profile
    def init_payoffs(self, payoffs, type_weights=None):
        if not type_weights:
            type_weights = [[10., 2., 1.], [1., 10., 1.], [1., 1., 10.]]
        self.payoff_count = dict([(y, dict([(x, {}) for x in self.responses])) for y in self.signals])
        self.payoff_belief = dict([(y, dict([(x, {}) for x in self.responses])) for y in self.signals])
        # This is a bit more fiddly. Psuedo counts are for meanings..
        for signal in self.signals:
            # total = sum(type_weights[signal])
            for player_type in self.signals:
                # freq = type_weights[signal][player_type] / float(total)
                for response in self.responses:
                    payoff = payoffs[player_type][response]
                    if payoff not in self.payoff_count[signal][response]:
                        self.payoff_count[signal][response][payoff] = type_weights[signal][player_type]
                        self.payoff_belief[signal][response][payoff] = 0.
                    else:
                        # print self.payoff_count
                        self.payoff_count[signal][response][payoff] += type_weights[signal][player_type]
        self.depth = 0
        for signal, responses in self.payoff_count.items():
            for response, payoff in responses.items():
                self.depth = max(len(payoff), self.depth)
        super(LexicographicResponder, self).init_payoffs(payoffs, type_weights)

    def set_uninformative_prior(self):
        for signal, responses in self.payoff_count.items():
            for response, payoffs in responses.iteritems():
                for payoff in payoffs:
                    self.payoff_count[signal][response][payoff] = 1

    def init_payoffs_(self, payoffs, type_weights=None, num=10):
        if not type_weights:
            type_weights = [[10., 2., 1.], [1., 10., 1.], [1., 1., 10.]]
        self.payoff_count = dict([(y, dict([(x, {}) for x in self.responses])) for y in self.signals])
        self.payoff_belief = dict([(y, dict([(x, {}) for x in self.responses])) for y in self.signals])
        # This is a bit more fiddly. Psuedo counts are for meanings..
        for signal in self.signals:
            # total = sum(type_weights[signal])
            for player_type in self.signals:
                # freq = type_weights[signal][player_type] / float(total)
                for response in self.responses:
                    payoff = payoffs[player_type][response]
                    if payoff not in self.payoff_count[signal][response]:
                        self.payoff_count[signal][response][payoff] = 0
                        self.payoff_belief[signal][response][payoff] = 0.
        self.depth = 0
        for signal, responses in self.payoff_count.items():
            for response, payoff in responses.items():
                self.depth = max(len(payoff), self.depth)
        for i in xrange(num):
            # Choose a random signal
            signal = self.random.choice(self.signals)
            # A weighted response
            response = self.random.choice(self.responses)
            # A weighted choice of type
            player_type = weighted_random_choice(self.signals, type_weights[signal], random=self.random)
            # Payoffs
            payoff = payoffs[player_type][response]
            self.payoff_count[signal][response][payoff] += 1
        super(LexicographicResponder, self).init_payoffs(payoffs, type_weights)

    # @profile
    def update_beliefs(self, payoff, signaller, signal, signaller_type=None, weight=1.):
        if payoff is not None:
            # print self.payoff_count, signal, payoff, self.response_log[len(self.response_log) - 1], weight
            self.payoff_count[signal][self.response_log[len(self.response_log) - 1]][payoff] += weight
            # super(LexicographicResponder, self).update_beliefs(payoff, signaller, signal, signaller_type)
            for signal, responses in self.payoff_count.items():
                for response, payoff in responses.items():
                    self.depth = max(len(payoff), self.depth)

    def freq_it(self, signal, response):
        """
        Return a generator over outcomes experienced, ordered by how often they've been seen.
        Sticks on the least frequent rather than ending.
        """
        shuffled_pairs = shuffled(self.payoff_count[signal][response].items())
        sorted_dict = sorted(shuffled_pairs, key=operator.itemgetter(1), reverse=True, cmp=self.compare_with_ties)
        n = 0
        while True:
            yield sorted_dict[min(n, len(sorted_dict) - 1)][0]
            n += 1

    def frequent(self, signal, response, n, signaller=None):
        """
        Return the nth most frequently experienced outcome from
        this response to the signal.
        """
        sorted_dict = sorted(self.payoff_count[signal][response].items(), key=operator.itemgetter(1), reverse=True, cmp=self.compare_with_ties)
        return sorted_dict[min(n, len(sorted_dict) - 1)][0]

    def respond(self, signal, opponent=None):
        """
        Make a judgement about somebody based on
        the signal they sent by minimising bayesian risk.
        """
        # super(LexicographicResponder, self).respond(signal, opponent)
        if opponent is not None:
            self.type_log.append(opponent.player_type)
        self.signal_log.append(signal)
        self.signal_matches[signal] += 1.
        n = 0
        respits = {}
        for response in self.responses:
                respits[response] = self.freq_it(signal, response)
        while n < self.depth:
            mappings = {}
            for response in self.responses:
                payoff = respits[response].next()
                mappings[response] = payoff
            sorted_mappings = sorted(shuffled(mappings.items(), self.random), key=operator.itemgetter(1), reverse=True)
            # Is there a best option?
            best = sorted_mappings[0][0]
            try:
                if sorted_mappings[0][1] > sorted_mappings[1][1]:
                    break
            except IndexError:
                # Only one payoff
                pass
            n += 1
        # self.response_log.pop()
        self.rounds += 1
        self.response_log.append(best)
        return best


class RecognitionLexicographicResponder(RecognitionResponder, LexicographicResponder):
    """
    A class of lexicographic responder that retrospectively updates when it learns a True
    type.
    """


class SharingLexicographicResponder(SharingResponder, LexicographicResponder):
    """
    A lexicographic reasoner that shares info updates.
    """


class SharingLexicographicSignaller(SharingSignaller, LexicographicSignaller):
    """
    A lexicographic reasoner that shares info updates.
    """
