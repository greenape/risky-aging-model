import game
from disclosuregame.Util import random_expectations
from referral import *
import collections
from random import Random
try:
    import scoop
    scoop.worker
    scoop_on = True
except:
    scoop_on = False
    pass

class CarryingGame(game.Game):
    def __unicode__(self):
        return "carrying_%s" % super(CarryingGame, self).__unicode__()

    def __init__(self, **kwargs):
        super(CarryingGame, self).__init__(**kwargs)
        self.player_random = Random(self.random.random())


    """
    A game type that maintains the size of the population of
    women, by adding in a new one of the same type as any that finish.
    """
    def play_game(self, players, file_name=""):
        women, midwives = players
        women_generator = self.signaller_fn.generator(random=self.player_random, type_distribution=self.women_weights, 
            agent_args=self.signaller_args, initor=self.signaller_initor,init_args=self.signaller_init_args)
        rounds = self.rounds
        birthed = []
        self.random.shuffle(women)
        num_midwives = len(midwives)
        women_res = self.measures_women.dump(None, self.rounds, self, None)
        mw_res = self.measures_midwives.dump(None, self.rounds, self, None)
        for i in range(rounds):
            players = [women.pop() for j in range(num_midwives)]
            self.random.shuffle(midwives)
            map(self.play_round, players, midwives)
            for x in midwives:
                x.finished += 1
            women_res = self.measures_women.dump(players, i, self, women_res)
            mw_res = self.measures_midwives.dump(midwives, i, self, mw_res)
            for woman in players:
                if self.all_played([woman], self.num_appointments):
                    woman.is_finished = True
                    # Add a new naive women back into the mix
                    new_woman = signaller_generator.next()
                    new_woman.started = i
                    new_woman.finished = i
                    women.insert(0, new_woman)
                    women_res.add_results(self.measures_women.dump([woman, new_woman], self.rounds, self))
                    del woman
                else:
                    women.insert(0, woman)
                    woman.finished += 1
            #if scoop_on:
            #    scoop.logger.info("Worker %s played %d rounds." % (scoop.worker[0], i))

        del women
        del midwives

        if scoop_on:
            scoop.logger.info("Worker %s completed a game." % (scoop.worker[0]))
        return women_res, mw_res

class CaseloadCarryingGame(CarryingGame, game.CaseloadGame):
    def __unicode__(self):
        return "caseload_carrying_%s" % super(CarryingGame, self).__unicode__()

    def play_game(self, players, file_name=""):
        women, midwives = players
        women_generator = self.signaller_fn.generator(random=self.player_random, type_distribution=self.women_weights, 
            agent_args=self.signaller_args, initor=self.signaller_initor,init_args=self.signaller_init_args)
        rounds = self.rounds
        birthed = []
        self.random.shuffle(women)
        num_midwives = len(midwives)
        women_res = self.measures_women.dump(None, self.rounds, self, None)
        mw_res = self.measures_midwives.dump(None, self.rounds, self, None)

        caseloads = {}
        num_women = len(women)
        num_midwives = len(midwives)
        load = num_women / num_midwives
        self.random.shuffle(women)
        for midwife in midwives:
            caseloads[midwife] = []
            for i in range(load):
                caseloads[midwife].append(women.pop())

        # Assign leftovers at random
        while len(women) > 0:
            caseloads[self.random.choice(midwives)].append(women.pop())

        for i in range(rounds):
            players = [caseloads[midwife].pop() for midwife in midwives]
            map(self.play_round, players, midwives)
            for x in midwives:
                x.finished += 1
            women_res = self.measures_women.dump(players, i, self, women_res)
            mw_res = self.measures_midwives.dump(midwives, i, self, mw_res)
            for j in range(len(players)):
                woman = players[j]
                women = caseloads[midwives[j]]
                if self.all_played([woman], self.num_appointments):
                    woman.is_finished = True
                    # Add a new naive women back into the mix
                    new_woman = signaller_generator.next()
                    new_woman.started = i
                    new_woman.finished = i
                    women.insert(0, new_woman)
                    women_res.add_results(self.measures_women.dump([woman, new_woman], self.rounds, self))
                    del woman
                else:
                    women.insert(0, woman)
                    woman.finished += 1
            if scoop_on:
                scoop.logger.info("Worker %s played %d rounds." % (scoop.worker[0], i))
        del women
        del midwives
        
        if scoop_on:
            scoop.logger.info("Worker %s completed a game." % (scoop.worker[0]))
        return women_res, mw_res


class CarryingReferralGame(CarryingGame, ReferralGame):
        """
        Just like the referral game, but maintains a carrying capacity.
        """

class CarryingCaseloadReferralGame(CaseloadCarryingGame, ReferralGame):
    """
    Ditto, but caseloaded.
    """