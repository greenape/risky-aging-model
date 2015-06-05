from measures import *

class RiskSpace(Measure):
    """
    Return a signaller's position in 'risk space'. i.e.
    a tuple of values corresponding to the risk associated
    with each signal.
    Returns valuespace for cpt players, and signal space for heuristic
    players.
    """
    def measure_one(self, woman):
        player_type = str(woman)
        if "prospect" in player_type:
            LOG.debug("Measuring valuespace for player %d, type %d, rule %s" % (hash(woman), woman.player_type, str(woman)))
            res = map(lambda signal: woman.cpt_value(woman.collect_prospects(signal)), woman.signals)
        elif "lexic" in player_type:
            LOG.debug("Measuring signalspace for player %d, type %d, rule %s" % (hash(woman), woman.player_type, str(woman)))
            res = woman.signal_search()
        else:
            LOG.debug("Measuring riskspace for player %d, type %d, rule %s" % (hash(woman), woman.player_type, str(woman)))
            res = map(lambda signal: woman.risk(signal), woman.signals)
        return res

    def measure(self, roundnum, women, game):
        res = map(lambda x: (self.measure_one(x), hash(x), x.player_type), women)
        return res

class SignalSpace(RiskSpace):
    """
    Return a signaller's preferred signal at this time.
    """
    def measure_one(self, woman):
        return woman.signal_search()

class ReferralEvents(Measure):
    """
    Log the id number and type of all players referred this round.
    """
    def measure(self, roundnum, women, game):
        res = map(lambda x: (hash(x), x.player_type), filter(lambda x: 1 in x.get_response_log(), women))
        return res

def space_measures_women(signals=None, base=None):
    if not signals:
        signals = [0, 1]
    measures = OrderedDict()
    measures["riskspace"] = RiskSpace()
    measures["referralevents"] = ReferralEvents()
    measures["signalspace"] = SignalSpace()
    if base is None:
        base = measures_women()
    base.add(Measures(measures))
    return base

def space_measures_mw(base=None):
    measures = OrderedDict()
    if base is None:
        base = measures_midwives()
    base.add(Measures(measures))
    return base