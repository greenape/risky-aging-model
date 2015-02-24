from measures import *

class PopCount(Measure):
    def __init__(self, **kwargs):
        super(PopCount, self).__init__(**kwargs)
        self.hash_bucket = set()

    """
    Return the count of this type up to roundnum.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = map(lambda x: hash(x), women)
         
        self.hash_bucket.update(women)
        return len(self.hash_bucket)


class HonestyMeasure(Measure):
    def __init__(self, counted=set(), appointment=0,**kwargs):
        super(HonestyMeasure, self).__init__(**kwargs)
        self.count = 0
        self.counted = counted
        self.appointment = appointment

    """
    Return the number of honest signals sent on an appointment.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = filter(lambda x: x.rounds == self.appointment, women)
        women = filter(lambda x: x.player_type in x.get_signal_log(), women)
        women = filter(lambda x: hash(x) not in self.counted, women)
        self.counted.update(map(hash, women))
        self.count += len(women)
        return self.count


class RefCount(Measure):
    def __init__(self, counted=set(), appointment=0,**kwargs):
        super(RefCount, self).__init__(**kwargs)
        self.count = 0
        self.counted = counted
        self.appointment = appointment

    """
    Return the number of women referred on an appointment.
    """
    def measure(self, roundnum, women, game):
        if self.player_type is not None:
            women = filter(lambda x: x.player_type == self.player_type, women)
        women = filter(lambda x: x.rounds == self.appointment, women)
        women = filter(lambda x: 1 in x.get_response_log(), women)
        women = filter(lambda x: hash(x) not in self.counted, women)
        self.counted.update(map(hash, women))
        self.count += len(women)
        return self.count
        
class CumulativeRefCount(Measure):
    def __init__(self, appointment=0, **kwargs):
        super(CumulativeRefCount, self).__init__(**kwargs)
        counted = set()
        self.counters = [RefCount(counted=counted, appointment=x, player_type=self.player_type, midwife_type=self.midwife_type) for x in range(appointment)]

    """
    Return the number of women referred upto an appointment.
    """
    def measure(self, roundnum, women, game):
        return sum(map(lambda x: x.measure(roundnum, women, game), self.counters))
        
class CumulativeHonestyCount(Measure):
    def __init__(self, appointment=0, **kwargs):
        super(CumulativeHonestyCount, self).__init__(**kwargs)
        counted = set()
        self.counters = [HonestyMeasure(counted=counted, appointment=x, player_type=self.player_type, midwife_type=self.midwife_type) for x in range(appointment)]

    """
    Return the number of women signalling honestly upto an appointment.
    """
    def measure(self, roundnum, women, game):
        return sum(map(lambda x: x.measure(roundnum, women, game), self.counters))

class AppointmentTypeSignalCount(TypeSignalCount):
    """
    Return a cumulative count of type-signal pairs, for players who have had
    a specific number of appointments.
    """
    def __init__(self, appointment=0, **kwargs):
        super(AppointmentTypeSignalCount, self).__init__(**kwargs)
        #Uninformed prior
        self.appointment = appointment

    def measure(self, roundnum, women, game):
        women = filter(lambda x: x.rounds == self.appointment, women)
        total_women = float(len(women))
        if total_women == 0:
            return "NA"
        if self.player_type is None:
            return "NA"
        # Probabilty of this signal and this type
        map(lambda x: self.measure_one(x), women)
        total = sum(x for counter in self.counts.values() for x in counter.values())
        result = self.counts[self.player_type][self.signal]
        return result

def abstract_measures_women(signals=[0, 1]):
    n_signals = len(signals)
    measures = OrderedDict()
    measures['round'] = Appointment()
    for i in range(n_signals):
        measures["type_%d_pop" % i] = PopCount(player_type = i)
        for j in range(12):
            measures["type_%d_round_%d_ref" % (i, j+1)] = CumulativeRefCount(player_type=i, appointment=j)
            measures["type_%d_round_%d_honesty" % (i, j+1)] = CumulativeHonestyCount(player_type=i, appointment=j)
            for k in range(n_signals):
                measures["n_type_%d_sig_%d_round_%d" % (i, k, j+1)] = AppointmentTypeSignalCount(player_type=i, signal=k, appointment=j, signals=signals)
    base = measures_women()
    base.add(Measures(measures))
    return base

def abstract_measures_mw():
    measures = OrderedDict()
    measures['appointment'] = Appointment()
    base = measures_midwives()
    base.add(Measures(measures))
    return base