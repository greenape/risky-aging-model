from collections import OrderedDict
import collections
from disclosuregame.results import Result
import itertools
import math

def percentile(N, percent, key=lambda x:x):
    """
    Taken from http://code.activestate.com/recipes/511478/
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1

def median(lst):
    return percentile(sorted(lst), 0.5)

def iqr(lst):
    lst = sorted(lst)
    return percentile(lst, 0.75) - percentile(lst, 0.25)

class Measures(object):
    def __init__(self, measures, dump_after=0, dump_every=25):
        self.measures = measures
        self.dump_after = dump_after
        self.dump_every = dump_every

    def add(self, other):
        """
        Add the measures defined in the other measure object to this one.
        """
        self.measures.update(other.measures)

    def keys(self):
        return self.measures.keys()

    def values(self):
        return self.measures.values()

    def dump(self, women, rounds, game, results=None):
        """
        A results dumper. Takes a tuple of a game and players, and two dictionaries.
        Measures should contain a mapping from a field name to method for getting a result
        given an appointment, set of players, and a game. Params should contain mappings
        from parameter names to values.
        Optionally takes an exist results object to add records to. This should have the same
        measures and params.
        Returns a results object for writing to csv.
        """
        if results is None:
            results = Result(self.measures.keys(), game.parameters, [])
        if women is None:
            return results
        line = map(lambda x: x.measure(rounds, women, game), self.measures.values())
        if rounds >= self.dump_after and (rounds % self.dump_every == 0 or rounds == (game.rounds - 1)):
            results.add_results(Result(self.measures.keys(), game.parameters, [line]))
        return results

# Measures

class Measure(object):
    def __init__(self, player_type=None, midwife_type=None, signal=None):
        self.player_type = player_type
        self.midwife_type = midwife_type
        self.signal = signal

    def filter_present(self, women, roundnum):
        """
        Filter out any women not present on this round.
        """
        women = filter(lambda x: x.started < roundnum, women)
        women = filter(lambda x: x.finished > roundnum, women)
        return women


class NumRounds(Measure):
    """
    Return the average number of rounds a player played before finishing.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = filter(lambda x: x.is_finished, women)
        if len(women) == 0:
            return "NA"
        return sum(map(lambda woman: woman.finished - woman.started, women)) / float(len(women))


class NumRoundsCumulative(Measure):
    def __init__(self, player_type=None, midwife_type=None, signal=None, counted=set()):
        super(NumRoundsCumulative, self).__init__(player_type, midwife_type, signal)
        self.count = 0
        self.rounds = 0.
        self.counted = counted

    """
    Return the cumulative average number of rounds played by a type.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = filter(lambda x: x.is_finished, women)

        women = filter(lambda x: hash(x) not in self.counted, women)
        self.counted.update(map(hash, women))
        self.count += len(women)
        self.rounds += sum(map(lambda woman: woman.finished - woman.started, women))
        if self.count == 0:
            return 0.
        return self.rounds / self.count


class Referred(Measure):
    """
    Return the fraction of players referred this round.
    """

    def measure(self, roundnum, women, game):
        return len(filter(lambda x: 1 in x.get_response_log(), women)) / float(len(women))

class Appointment(Measure):
    def measure(self, roundnum, women, game):
        """
        Return the value passed as roundnum.
        """
        return roundnum

class TypeFinished(Measure):
    """
    Return the fraction of women of a particular type who finished this round.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        num_women = float(len(women))
        num_finished = len(filter(lambda x: x.is_finished, women))
        if num_women == 0:
            return 0.
        return num_finished / num_women


class TypeSignalBreakdown(Measure):
    """
    Return a function that yields the fraction of women of some type signalling
    a particular way who had a midwife of a particular type in the last round.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        if self.midwife_type is not None:
            women = filter(lambda x: x.get_type_log()[len(x.get_type_log()) - 1] == self.midwife_type, women)
        num_women = float(len(women))

        women = filter(lambda x: x.get_signal_log()[len(x.get_signal_log()) - 1] == self.signal, women)
        signalled = len(women)
        if num_women == 0:
            return 0.
        return signalled / num_women
    

class TypeReferralBreakdown(Measure):
    """
    Return a function that yields the fraction of women of some type signalling
    a particular way who had a midwife of a particular type and got referred.
    If signal is None, then this is for all signals.
    If midwife_type is None, then this is for all midwife types.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        if self.midwife_type is not None:
            women = filter(lambda x: x.get_type_log()[len(x.get_type_log()) - 1] == self.midwife_type, women)
        if self.signal is not None:
            women = filter(lambda x: x.get_signal_log()[len(x.get_signal_log()) - 1] == self.signal, women)
        num_women = float(len(women))
        women = filter(lambda x: 1 in x.get_response_log(), women)
        signalled = len(women)
        if num_women == 0:
            return 0.
        return signalled / num_women

class PayoffType(Measure):
    """
    Return a function that gives the average payoff in the last
    round for a given type. If type is None, then the average
    for all types is returned.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        if len(women) == 0:
            return 0.
        try:
            result = sum(map(lambda x: x.get_memory()[1][3][len(x.get_memory()[1][3])], women)) / float(len(women))
        except AttributeError:
            result = sum(map(lambda x: x.payoff_log[len(x.payoff_log)], women)) / float(len(women))
        return result
    

class SignalChange(Measure):
    """
    Return a function that yields average change in signal
    by women this round from last round.
    """
    def measure(self, roundnum, women, game):
        if roundnum == 0:
            return 0.
        women = filter(lambda x: x.player_type == self.player_type, women)
        num_women = len(women)
        if num_women == 0:
            return 0.
        change = map(lambda x: x.get_signal_log()[roundnum - x.started] - x.get_signal_log()[roundnum - 1 - x.started], women)
        return sum(change) / float(num_women)
    

class SignalRisk(Measure):
    """
    Return a function that gives the average risk associated
    with sending signal by players of this type who had that
    midwife type on their last round.
    """
    def measure(self, roundnum, women, game):
        #women = filter(lambda x: len(x.get_type_log()) > roundnum, women)
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        if self.midwife_type is not None:
            women = filter(lambda x: x.get_type_log()[len(x.get_type_log()) - 1] == self.midwife_type, women)
        total = sum(map(lambda x: x.round_signal_risk(roundnum)[self.signal], women))
        if len(women) == 0:
            return 0.
        return total / float(len(women))

class TypeFrequency(Measure):
    """
    Return the frequency of this type in the population at this round.
    """
    def measure(self, roundnum, women, game):
        
        types = map(lambda x: x.player_type, women)
        frequencies = collections.Counter(types)
        total = sum(frequencies.values())
        if total == 0:
            return 0.
        return frequencies[self.player_type] / float(total)

class SignalExperience(Measure):
    """
    A measure that gives the frequency of a signal experienced up
    to that round by some (or all) types of midwife.
    """
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        else:
            group_log = itertools.chain(*map(lambda x: x.get_signal_log(), women))
        frequencies = collections.Counter(group_log)
        total_signals = sum(frequencies.values())
        if total_signals == 0:
            return 0.
        return frequencies[self.signal] / float(total_signals)

class TypeExperience(Measure):
    """
    A measure that gives the frequency of a type experienced up
    to that round by some (or all) types of midwife.
    """
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        else:
            group_log = itertools.chain(*map(lambda x: x.type_log, women))
        frequencies = collections.Counter(group_log)
        total_signals = sum(frequencies.values())
        if total_signals == 0:
            return 0.
        return frequencies[self.player_type] / float(total_signals)


class RightCallUpto(Measure):
    """
    Gives the frequency of right calls given by midwives of
    some (or any) type, up to now.
    """
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        total_calls = 0.
        total_right = 0.
        for midwife in women:
            r_log = midwife.response_log
            t_log = midwife.type_log
            total_calls += len(r_log)
            for i in range(len(r_log)):
                response = r_log[i]
                player = t_log[i]
                if response == 0:
                    if player == 0:
                        total_right += 1
                else:
                    if player != 0:
                        total_right += 1
        if total_calls == 0:
            return 0.
        return total_right / total_calls

class RightCall(Measure):
    """
    Gives the frequency of right calls given by midwives of
    some (or any) type, in this.
    """
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        total_calls = 0.
        total_right = 0.
        for midwife in women:
            try:
                response = midwife.response_log[len(midwife.response_log) - 1]
                player = midwife.type_log[len(midwife.response_log) - 1]
                total_calls += 1
                if response == 0:
                    if player == 0:
                        total_right += 1
                else:
                    if player != 0:
                        total_right += 1
            except IndexError:
                pass
        if total_calls == 0:
            return 0.
        return total_right / total_calls

class FalsePositive(Measure):
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        total_calls = 0.
        total_right = 0.
        for midwife in women:
            try:
                response = midwife.response_log[roundnum]
                player = midwife.type_log[roundnum]
                if response == 1:
                    if player == 0:
                        total_right += 1
                    total_calls += 1
            except IndexError:
                pass
        if total_calls == 0:
            return 0.
        return total_right / total_calls

class FalseNegative(Measure):
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        total_calls = 0.
        total_right = 0.
        for midwife in women:
            try:
                response = midwife.response_log[roundnum]
                player = midwife.type_log[roundnum]
                if response == 0:
                    if player != 0:
                        total_right += 1
                    total_calls += 1
            except IndexError:
                pass
        if total_calls == 0:
            return 0.
        return total_right / total_calls

class TypedFalseNegativeUpto(Measure):
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        total_calls = 0.
        total_right = 0.
        for midwife in women:
            r_log = midwife.response_log[:roundnum]
            t_log = midwife.type_log[:roundnum]
            for i in range(len(r_log)):
                response = r_log[i]
                player = t_log[i]
                if player == self.player_type:
                    total_calls += 1
                    if response == 0:
                        total_right += 1
        if total_calls == 0:
            return 0.
        return total_right / total_calls


class FalsePositiveUpto(Measure):
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        total_calls = 0.
        total_right = 0.
        for midwife in women:
            r_log = midwife.response_log[:roundnum]
            t_log = midwife.type_log[:roundnum]
            for i in range(len(r_log)):
                response = r_log[i]
                player = t_log[i]
                if response == 1:
                    if player == 0:
                        total_right += 1
                    total_calls += 1
        if total_calls == 0:
            return 0.
        return total_right / total_calls

class FalseNegativeUpto(Measure):
    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        total_calls = 0.
        total_right = 0.
        for midwife in women:
            r_log = midwife.response_log[:roundnum]
            t_log = midwife.type_log[:roundnum]
            for i in range(len(r_log)):
                response = r_log[i]
                player = t_log[i]
                if response == 0:
                    if player != 0:
                        total_right += 1
                    total_calls += 1
        if total_calls == 0:
            return 0.
        return total_right / total_calls


class AccruedPayoffs(Measure):
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        total = sum(map(lambda x: x.accrued_payoffs, women))
        if len(women) == 0:
            return 0.
        return total / float(len(women))

class GameSeed(Measure):
    def measure(self, roundnum, women, game):
        return game.seed

class GroupResponse(Measure):
    def measure_one(self, woman):
        signaller = type(woman)()
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        try:
            memory = woman.shareable
        except:
            raise
        r = woman.respond(self.signal, signaller)
        woman.signal_log.pop()
        woman.response_log.pop()
        woman.rounds -= 1
        woman.signal_matches[self.signal] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = memory
        except:
            raise
        return r

    def measure(self, roundnum, women, game):
        if self.midwife_type is not None:
            women = filter(lambda x: x.player_type == self.midwife_type, women)
        if len(women) == 0:
            return "NA"
        return sum(map(self.measure_one, women)) / float(len(women))

class GroupHonesty(Measure):
    """
    Return the average absolute distance of everybody's choice of signal
    if they were to signal right now, from their own type.
    """
    def measure_one(self, woman):
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        state = woman.random.getstate()
        r = woman.do_signal()
        woman.random.setstate(state)
        woman.signal_log.pop()
        woman.rounds -= 1
        woman.signal_matches[r] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = None
        except:
            pass
        return abs(r - woman.player_type)

    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        if len(women) == 0:
            return "NA"
        return sum(map(self.measure_one, women)) / float(len(women))

class GroupSignal(GroupHonesty):
    """
    Return the average of everybody's choice of signal
    if they were to signal right now.
    """
    def measure_one(self, woman):
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        state = woman.random.getstate()
        r = woman.do_signal()
        woman.random.setstate(state)
        woman.signal_log.pop()
        woman.rounds -= 1
        woman.signal_matches[r] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = None
        except:
            pass
        return r

class GroupSignalMedian(GroupSignal):
    """
    Return the median signal of the group.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        if len(women) == 0:
            return "NA"
        return median(map(self.measure_one, women))


class GroupSignalIQR(GroupSignal):
    """
    Return the IQR of the group's signals.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        if len(women) == 0:
            return "NA"
        return iqr(map(self.measure_one, women))


class ExpectedPointMutualInformation(Measure):
    """
    Return the expected pointwise mutual information of this group-signal combination.
    """
    def measure_one(self, woman, signal):
        """
        Return a 1 if this agent would signal to match the signal parameter.
        """
        #
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        state = woman.random.getstate()
        r = woman.do_signal()
        woman.random.setstate(state)
        woman.signal_log.pop()
        woman.rounds -= 1
        woman.signal_matches[r] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = None
        except:
            pass
        return 1. if r == signal else 0.

    def measure(self, roundnum, women, game):
        total_women = float(len(women))
        if total_women == 0:
            return "NA"
        if self.player_type is not None:
            typed_women = filter(lambda x: x.player_type == self.player_type, women)
        total_type = float(len(typed_women))
        # Probability of being this player type
        p_type = total_type / total_women
        if p_type == 0:
            return 0.
        # Probability of this signal
        p_signal = sum(map(lambda x: self.measure_one(x, self.signal), women)) / total_women
        if p_signal == 0:
            return 0.
        # Probabilty of this signal and this type
        p_type_signal = sum(map(lambda x: self.measure_one(x, self.signal), typed_women)) / total_women
        if p_type_signal == 0 :
            return 0.
        return p_type_signal*math.log(p_type_signal / (p_type*p_signal), 2)


class TypeSignalProbability(ExpectedPointMutualInformation):
    """
    Calculate p(signal, type). Can marginalize for individual distributions.
    """

    def measure_one(self, woman, signal):
        """
        Return a 1 if this agent would signal to match the signal parameter.
        There is a fudge here - agents will sometimes choose effectively randomly,
        where there's no reason to prefer one option. So we reset the random state 
        after pulling a signal to avoid messing up the distributions on sucessive measures.
        """
        #
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        state = woman.random.getstate()
        r = woman.do_signal()
        woman.random.setstate(state)
        woman.signal_log.pop()
        woman.rounds -= 1
        woman.signal_matches[r] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = None
        except:
            pass
        return 1. if r == signal else 0.

    def measure(self, roundnum, women, game):
        total_women = float(len(women))
        if total_women == 0:
            return "NA"
        if self.player_type is not None:
            typed_women = filter(lambda x: x.player_type == self.player_type, women)
        # Probabilty of this signal and this type
        p_type_signal = sum(map(lambda x: self.measure_one(x, self.signal), typed_women)) / total_women
        if p_type_signal == 0 :
            return 0.
        return p_type_signal

class BayesTypeSignalProbability(ExpectedPointMutualInformation):
    counts = {0: {0: 100.0, 1: 0.0, 2: 0.0},
                1: {0: 70.0, 1: 30.0, 2: 0.0},
                2: {0: 60.0, 1: 30.0, 2: 10.0}}
    total = 300.
    """
    Calculate p(signal, type) using Bayesian updates on a dirichlet distrbution.
    """

    def measure_one(self, woman, signal):
        """
        Return a 1 if this agent would signal to match the signal parameter.
        There is a fudge here - agents will sometimes choose effectively randomly,
        where there's no reason to prefer one option. So we reset the random state 
        after pulling a signal to avoid messing up the distributions on sucessive measures.
        """
        #
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        #state = woman.random.getstate()
        r = woman.do_signal()
        #woman.random.setstate(state)
        woman.signal_log.pop()
        woman.rounds -= 1
        woman.signal_matches[r] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = None
        except:
            pass
        BayesTypeSignalProbability.total += 1.
        BayesTypeSignalProbability.counts[woman.player_type][r] += 1.
        return

    def measure(self, roundnum, women, game):
        total_women = float(len(women))
        if total_women == 0:
            return "NA"
        BayesTypeSignalProbability.total += total_women
        if self.player_type is None:
            return "NA"
        # Probabilty of this signal and this type
        map(lambda x: self.measure_one(x), women))
        return BayesTypeSignalProbability.counts[self.signal][self.player_type] / BayesTypeSignalProbability.total

class SignalEntropy(ExpectedPointMutualInformation):
    """
    Return the shannon entropy of the signalling distribution.
    """

    def measure(self, roundnum, women, game):
        total_women = float(len(women))
        if total_women == 0:
            return "NA"
        def pointentropy(signal):
            # Probability of this signal
            p_signal = sum(map(lambda x: self.measure_one(x, signal), women)) / total_women
            if p_signal == 0 :
                return 0.
            return p_signal*math.log(p_signal, 2)
        return -sum(map(pointentropy, [0, 1, 2]))

class TypeEntropy(Measure):
    """
    Return the shannon entropy of the type distribution.
    """

    def measure(self, roundnum, women, game):
        total_women = float(len(women))
        if total_women == 0:
            return "NA"
        def pointentropy(player_type):
            # Probability of this type
            typed_women = filter(lambda x: x.player_type == player_type, women)
            total_type = float(len(typed_women))
            # Probability of being this player type
            p_type = total_type / total_women
            if p_type == 0:
                return 0.
            return p_type*math.log(p_type, 2)
        return -sum(map(pointentropy, [0, 1, 2]))


class SquaredGroupHonesty(GroupHonesty):
    """
    Return the average squared distance of everybody's choice of signal
    if they were to signal right now, from their own type.
    """
    def measure_one(self, woman):
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        r = woman.do_signal(self.signal)
        woman.signal_log.pop()
        woman.rounds -= 1
        woman.signal_matches[r] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = None
        except:
            pass
        return (r - woman.player_type)**2

class NormalisedSquaredGroupHonesty(GroupHonesty):
    def scale(self, n, low, high, a=-1., b=1.):
        return (((b - a)*(n - low)) / (high - low) ) + a

    """
    Return the average squared normalised distance of everybody's choice of signal
    if they were to signal right now, from their own type.
    Distances are normalised to between +-1, type 1s are left as is.
    """
    def measure_one(self, woman):
        #print "Hashing by", hash(woman), "hashing", hash(signaller)
        r = woman.do_signal(self.signal)
        woman.signal_log.pop()
        woman.rounds -= 1
        woman.signal_matches[r] -= 1
        try:
            woman.signal_memory.pop(hash(signaller), None)
            woman.shareable = None
        except:
            pass
        diff = (r - woman.player_type)
        if woman.player_type == 0:
            diff = self.scale(diff, 0., 2.)
        elif woman.player_type == 2:
            diff = self.scale(diff, -2., 0., 0.)
        return diff**2

def measures_women():
    measures = OrderedDict()
    measures['game_seed'] = GameSeed()
    measures['appointment'] = Appointment()
    #measures['finished'] = TypeFinished()
    #measures["honesty"] = GroupHonesty()
    #measures["nom_sq_honesty"] = NormalisedSquaredGroupHonesty()
    measures["group_signal"] = GroupSignal()
    measures["median_signal"] = GroupSignalMedian()
    measures["signal_iqr"] = GroupSignalIQR()
    #measures["type_entropy"] = TypeEntropy()
    #measures["signal_entropy"] = SignalEntropy()
    #measures['accrued_payoffs'] = AccruedPayoffs()
    for i in range(3):
        #measures["type_%d_ref" % i] = TypeReferralBreakdown(player_type=i)
        #measures["type_%d_finished" % i] = TypeFinished(player_type=i)
        #measures['accrued_payoffs_type_%d' % i] = AccruedPayoffs(player_type=i)
        #measures['rounds_played_type_%d_upto' % i] = NumRoundsCumulative(player_type=i)
        measures['rounds_played_type_%d' % i] = NumRounds(player_type=i)
        measures['type_%d_frequency' % i] = TypeFrequency(player_type=i)
        #measures["honesty_type_%d" % i] = GroupHonesty(player_type=i)
        measures["group_signal_%d" % i] = GroupSignal(player_type=i)
        measures["median_signal_type_%d" % i] = GroupSignalMedian(player_type=i)
        measures["signal_iqr_type_%d" % i] = GroupSignalIQR(player_type=i)
        for j in range(3):
            #measures["pmi_type_%d_signal_%d" % (i, j)] = ExpectedPointMutualInformation(player_type=i, signal=j)
            measures["p_signal_%d_type_%d" % (i, j)] = BayesTypeSignalProbability(player_type=j, signal=i)
            #measures["type_%d_signal_%d" % (i, j)] = TypeSignalBreakdown(player_type=i, signal=j)
            #measures["type_%d_mw_%d_ref" % (i, j)] = TypeReferralBreakdown(player_type=i, midwife_type=j)
            #measures["type_%d_sig_%d_ref" % (i, j)] = TypeReferralBreakdown(player_type=i, signal=j)
            #for k in range(3):
            #    measures["type_%d_mw_%d_sig_%d" % (i, j, k)] = TypeReferralBreakdown(player_type=i, midwife_type=j, signal=k)
    return Measures(measures, 0, 1)


##@profile
def measures_midwives():
    measures = OrderedDict()
    measures['game_seed'] = GameSeed()
    measures['appointment'] = Appointment()
    measures['all_right_calls_upto'] = RightCallUpto()
    #measures['all_right_calls'] = RightCall()
    measures['false_positives_upto'] = FalsePositiveUpto()
    #measures['false_positives'] = FalsePositive()
    measures['false_negatives_upto'] = FalseNegativeUpto()
    #measures['false_negatives'] = FalseNegative()
    #measures['accrued_payoffs'] = AccruedPayoffs()
    for i in range(3):
        measures['response_signal_%d' % i] = GroupResponse(signal=i)
        measures['response_signal_0_type_%d' % i] = GroupResponse(signal=0,midwife_type=i)
        #measures['signal_%d_frequency' % i] = SignalExperience(signal=i)
        #measures['type_%d_frequency' % i] = TypeExperience(player_type=i)
        #measures['type_%d_right_calls_upto' % i] = RightCallUpto(midwife_type=i)
        #measures['type_%d_right_calls' % i] = RightCall(midwife_type=i)
        #measures['type_%d_false_positives_upto' % i] = FalsePositiveUpto(midwife_type=i)
        #measures['type_%d_false_positives' % i] = FalsePositive(midwife_type=i)
        #measures['type_%d_false_negatives_upto' % i] = FalseNegativeUpto(midwife_type=i)
        #measures['type_%d_false_negatives' % i] = FalseNegative(midwife_type=i)
        #measures['type_%d_misses' % i] = TypedFalseNegativeUpto(player_type=i)
        #measures['accrued_payoffs_type_%d' % i] = AccruedPayoffs(player_type=i)
    return Measures(measures, 0)