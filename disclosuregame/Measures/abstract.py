from measures import *


class PopCount(Measure):
    def __init__(self, **kwargs):
        super(PopCount, self).__init__(**kwargs)
        self.hash_bucket = set()

    """
    Return the count of this type.
    """

    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = map(lambda x: x.ident, women)

        self.hash_bucket.update(women)
        return len(self.hash_bucket)

class PopCountNow(Measure):

    """
    Return the count of this type.
    """

    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = map(lambda x: x.ident, women)
        return len(women)

class PlayedJustNow(Measure):

    """
    Return the count of this type who just played.
    """

    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = [player for player in women if player.just_played]
        women = map(lambda x: x.ident, women)
        return len(women)

class RoundsPlayedSignal(Measure):
    """
    Calculate how many of some type of agent are on a particular appointment,
    and (optionally) have sent a particular signal.
    """
    def _init__(self, appointment=0, **kwargs):
        super(RoundsPlayedSignal, self).__init__(**kwargs)
        self.appointment = appointment

    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        # women = filter(lambda (y, x): x.rounds == self.appointment, women)
        women = [player for player in women if player.rounds == self.appointment]
        if self.signal is not None:
            women = [player for player in women if self.signal in player.get_signal_log()]
        return len(women)

class RoundsPlayedRef(RoundsPlayedSignal):
    """
    Calculate how many of some type of agent are on a particular appointment,
    and have been referred.
    """

    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        # women = filter(lambda (y, x): x.rounds == self.appointment, women)
        women = [player for player in women if player.rounds == self.appointment]
        if self.signal is not None:
            women = [player for player in women if 1 in player.get_response_log()]
        return len(women)

class HonestyMeasure(Measure):
    def __init__(self, counted=None, appointment=0, **kwargs):
        super(HonestyMeasure, self).__init__(**kwargs)
        if counted is None:
            counted = set()
        self.count = 0
        self.counted = counted
        self.appointment = appointment

    """
    Return the number of honest signals sent on an appointment.
    """

    def measure(self, roundnum, women, game):
        #pairs = zip(map(lambda x: x.ident, women), women)
        # women = filter(lambda (x, y): x not in self.counted, pairs)
        women = [player for player in women if player.ident not in self.counted]
        if self.player_type is not None:
            #women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        #women = filter(lambda (y, x): x.rounds == self.appointment, women)
        women = [player for player in women if player.player_type in player.get_signal_log()]
        women = [player.ident for player in women if player.rounds == self.appointment]
        #women = filter(lambda (y, x): x.player_type in x.get_signal_log(), women)
        self.counted.update(women)
        self.count += len(women)
        return self.count


class RefCountAnyRound(Measure):
    def __init__(self, **kwargs):
        super(RefCountAnyRound, self).__init__(**kwargs)
        self.count = 0

    """
    Return the number of women referred.
    """

    def measure(self, roundnum, women, game):
        #pairs = zip(map(lambda x: x.ident, women), women)
        #women = filter(lambda (x, y): x not in self.counted, pairs)
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        #women = filter(lambda (y, x): x.rounds == self.appointment, women)
        #women = filter(lambda (y, x): 1 in x.get_response_log(), women)
        women = [player.ident for player in women if 1 in player.get_response_log()]
        self.count += len(women)
        return self.count


class RefCountThisRound(Measure):

    """
    Return the number of women referred.
    """

    def measure(self, roundnum, women, game):
        #pairs = zip(map(lambda x: x.ident, women), women)
        #women = filter(lambda (x, y): x not in self.counted, pairs)
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        #women = filter(lambda (y, x): x.rounds == self.appointment, women)
        #women = filter(lambda (y, x): 1 in x.get_response_log(), women)
        women = [player.ident for player in women if 1 in player.get_response_log()]
        return len(women)

class FinishedThisRound(Measure):

    """
    Return the number of women referred.
    """

    def measure(self, roundnum, women, game):
        #pairs = zip(map(lambda x: x.ident, women), women)
        #women = filter(lambda (x, y): x not in self.counted, pairs)
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
            startcount = len(women)
        #women = filter(lambda (y, x): x.rounds == self.appointment, women)
        #women = filter(lambda (y, x): 1 in x.get_response_log(), women)
        women = [player.ident for player in women if player.is_finished]
        return len(women)

class FinishedAnyRound(RefCountAnyRound):
    """
    Return the number of women referred.
    """

    def measure(self, roundnum, women, game):
        #pairs = zip(map(lambda x: x.ident, women), women)
        #women = filter(lambda (x, y): x not in self.counted, pairs)
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        #women = filter(lambda (y, x): x.rounds == self.appointment, women)
        #women = filter(lambda (y, x): 1 in x.get_response_log(), women)
        women = [player.ident for player in women if player.is_finished]
        self.count += len(women)
        return self.count


class RefCount(Measure):
    def __init__(self, counted=set(), appointment=0, **kwargs):
        super(RefCount, self).__init__(**kwargs)
        self.count = 0
        self.counted = counted
        self.appointment = appointment

    """
    Return the number of women referred on an appointment.
    """

    def measure(self, roundnum, women, game):
        #pairs = zip(map(lambda x: x.ident, women), women)
        #women = filter(lambda (x, y): x not in self.counted, pairs)
        women = [woman for woman in women if woman.ident not in self.counted]
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        #women = filter(lambda (y, x): x.rounds == self.appointment, women)
        women = [player for player in women if player.rounds == self.appointment]
        #women = filter(lambda (y, x): 1 in x.get_response_log(), women)
        women = [player.ident for player in women if 1 in player.get_response_log()]
        self.counted.update(women)
        self.count += len(women)
        return self.count


class CumulativeRefCount(Measure):
    def __init__(self, appointment=0, **kwargs):
        super(CumulativeRefCount, self).__init__(**kwargs)
        counted = set()
        self.counters = [
            RefCount(counted=counted, appointment=x, player_type=self.player_type, midwife_type=self.midwife_type) for x
            in range(appointment + 1)]

    """
    Return the number of women referred upto an appointment.
    """

    def measure(self, roundnum, women, game):
        return sum(map(lambda x: x.measure(roundnum, women, game), self.counters))


class CumulativeHonestyCount(Measure):
    def __init__(self, appointment=0, **kwargs):
        super(CumulativeHonestyCount, self).__init__(**kwargs)
        counted = set()
        self.counters = [
            HonestyMeasure(counted=counted, appointment=x, player_type=self.player_type, midwife_type=self.midwife_type)
            for x in range(appointment + 1)]

    """
    Return the number of women signalling honestly upto an appointment.
    """

    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            # women = filter(lambda (y, x): x.player_type == self.player_type, women)
            women = [player for player in women if player.player_type == self.player_type]
        women = [player for player in women if player.player_type in player.get_signal_log()]
        return sum(map(lambda x: x.measure(roundnum, women, game), self.counters))


class AppointmentTypeSignalCount(TypeSignalCount):
    """
    Return a cumulative count of type-signal pairs, for players who have had
    a specific number of appointments.
    """

    def __init__(self, appointment=0, **kwargs):
        super(AppointmentTypeSignalCount, self).__init__(**kwargs)
        self.appointment = appointment

    def measure(self, roundnum, women, game):
        women = filter(lambda x: x.rounds == self.appointment, women)
        women = [player for player in women if player.player_type == self.player_type]
        total_women = float(len(women))
        if total_women == 0:
            return "NA"
        if self.player_type is None:
            return "NA"
        # Probabilty of this signal and this type
        map(lambda x: self.measure_one(x), women)
        return self.count



def abstract_measures_women(signals=None, freq=1):
    if not signals:
        signals = [0, 1]
    n_signals = len(signals)
    measures = OrderedDict()
    measures['round'] = Appointment()
    for i in range(n_signals):
        measures["type_%d_pop" % i] = PopCount(player_type=i)
        measures["type_%d_pop_now" % i] = PopCountNow(player_type=i)
        #measures["type_%d_round_12_ref" % i] = CumulativeRefCount(player_type=i, appointment=11)
        measures["ref_overall_%d" % i] = RefCountAnyRound(player_type=i)
        measures["ref_now_%d" % i] = RefCountThisRound(player_type=i)
        measures["finished_now_%d" % i] = FinishedThisRound(player_type=i)
        measures["finished_overall_%d" % i] = FinishedAnyRound(player_type=i)
        measures["just_played_%d" %i] = PlayedJustNow(player_type=i)
        for j in range(12):
            measures["type_%d_round_%d_ref" % (i, j + 1)] = CumulativeRefCount(player_type=i, appointment=j)
            measures["type_%d_round_%d_honesty" % (i, j + 1)] = CumulativeHonestyCount(player_type=i, appointment=j)
        #    for k in range(n_signals):
        #        measures["n_type_%d_sig_%d_round_%d" % (i, k, j + 1)] = AppointmentTypeSignalCount(player_type=i,
        #                                                                                           signal=k,
        #                                                                                           appointment=j,
        #                                                                                           signals=signals)
    base = measures_women(freq=freq, signals=signals)
    base.add(Measures(measures))
    return base


def abstract_measures_mw(signals=None, freq=1):
    if not signals:
        signals = [0, 1]
    measures = OrderedDict()
    measures['appointment'] = Appointment()
    base = measures_midwives(freq=freq, signals=signals)
    base.add(Measures(measures))
    return base
