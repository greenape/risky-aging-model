from carrying import CarryingReferralGame
from disclosuregame.Measures import measures_midwives, measures_women
from disclosuregame.Util import random_expectations
from disclosuregame.Agents.bayes import Agent
from random import Random
from math import copysign
import operator
from collections import OrderedDict
from copy import deepcopy
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

class CarryingInformationGame(CarryingReferralGame):
    """
    A game type identical with the carrying referral game, but where some information
    is shared on each round. Sharing is controlled by the share_prob parameters which
    are the probability that any given memory is shared.
    """
    def __init__(self, mw_share_prob=0, mw_share_bias=-.99, women_share_prob=0, women_share_bias=0.99,
                 signaller_args=None, responder_args=None, **kwargs):
        if not responder_args:
            responder_args = {}
        if not signaller_args:
            signaller_args = {}
        super(CarryingInformationGame, self).__init__(**kwargs)
        self.parameters['mw_share_prob'] = mw_share_prob
        self.parameters['mw_share_bias'] = mw_share_bias
        self.parameters['women_share_prob'] = women_share_prob
        self.parameters['women_share_bias'] = women_share_bias
        self.mw_share_bias = mw_share_bias
        self.mw_share_prob = mw_share_prob
        self.women_share_bias = women_share_bias
        self.women_share_prob = women_share_prob
        self.signaller_args = signaller_args
        self.responder_args = responder_args

    def __str__(self):
        return "sharing_%s" % super(CarryingInformationGame, self).__unicode__()

    #@profile
    def play_game(self, players, file_name=""):
        #self.random.seed(1)
        try:
            worker = scoop.worker[0]
        except:
            worker = multiprocessing.current_process()
        LOG.debug("Worker %s playing a game." % worker)
        women, midwives = players

        women_res, mw_res = self.pre_game(women, midwives)
        LOG.debug("Starting play.")
        for self.current_round in range(self.rounds):
            women_res, mw_res, players = self.run_round(women, midwives, women_res, mw_res)
            self.post_round(players, women, midwives)
        del women
        del midwives
        del self.women_memories
        LOG.debug("Worker %s completed a game." % worker)
        return women_res, mw_res

    def pre_game(self, women, midwives):
        """
        Setup function for running a game.
        """
        self.signaller_generator = self.signaller_fn.generator(random=self.player_random, type_distribution=self.women_weights, 
            agent_args=self.signaller_args, initor=self.signaller_initor,init_args=self.signaller_init_args)
        while self.signaller_fn.id_generator.next() != self.gen_reset:
            LOG.debug("Spinning id generator.")
        LOG.debug("Made player generator.")
        self.random.shuffle(women)
        self.num_midwives = len(midwives)
        women_res = self.measures_women.dump(None, self.rounds, self)
        mw_res = self.measures_midwives.dump(None, self.rounds, self)
        self.women_memories = []
        return women_res, mw_res

    def run_round(self, women, midwives, women_res, mw_res):
        """
        Run one round of simulation.
        """
        finished_count = 0
        players = [women.pop() for j in range(self.num_midwives)]
        self.random.shuffle(midwives)
        map(self.play_round, players, midwives)
        for x in midwives:
            x.finished += 1
        for woman in players + women:
            if self.all_played([woman], self.num_appointments):
                woman.is_finished = True
                finished_count += 1
        LOG.debug("Counted %d finished of %d." % (finished_count, len(players + women)))
        women_res = self.measures_women.dump(women + players, self.current_round, self, results=women_res)
        mw_res = self.measures_midwives.dump(midwives, self.current_round, self, results=mw_res)
        return women_res, mw_res, players
        

    def post_round(self, players, signallers, responders, **kwargs):
        """
        Actions to perform after playing a round.
        :param **kwargs:
        """
        try:
            worker = scoop.worker[0]
        except:
            worker = multiprocessing.current_process()

        for woman in players:
            if woman.is_finished:
                # Add a new naive women back into the mix
                new_woman = self.signaller_generator.next()
                new_woman.started = self.current_round
                new_woman.finished = self.current_round
                signallers.insert(0, new_woman)
                LOG.debug("Generated a new player.")
                if self.women_share_prob > 0 and abs(self.women_share_bias) < 1:
                    self.women_memories.append(woman.get_memory())
                for midwife in responders:
                    midwife.signal_memory.pop(hash(woman), None)
                del woman
            else:
                signallers.insert(0, woman)
                woman.finished += 1
        # Share information
        LOG.debug("Worker %s prepping share." % worker)
        #Midwives
        try:
            self.share_midwives(responders)
        except Exception as e:
            LOG.debug("Sharing to midwives failed.")
            LOG.debug(e)

        #Women
        try:
            self.share_women(signallers)
        except Exception as e:
            LOG.debug("Sharing to women failed.")
            LOG.debug(e)

    def weighted_prob(self, high, low, weight, bias):
        if bias > 0:
            high, low = low, high
        diff = high - low
        return (weight - low) / float(diff)
    
    def threshold(self, weight, bias, base):
        """
        Bias of the coin deciding if this memory is shared
        """
        return base*(1-weight*abs(bias))

    #@profile
    def share_midwives(self, midwives):
        """
        Share the recent experiences of each midwife with some probability,
        or erase them if not shared.
        """
        #Worst outcome for a responder
        for midwife in midwives:
            memory = midwife.shareable
            midwife.shareable = None
            p = self.random.random()
            LOG.debug("p=%f, threshold is %f" % (p, self.mw_share_prob))
            if p < self.mw_share_prob and memory is not None:
                LOG.debug("Sharing %s" % str(memory))
                possibles = filter(lambda x: hash(x) != hash(midwife), midwives)
                self.disseminate_midwives(memory[1][1], possibles)


    def share_women(self, women):
            """
            Go through all the experiences of those who were referred this round,
            and for each one share with probability self.women_share_prob.
            """
            while len(self.women_memories) > 0:
                memory = self.women_memories.pop()
                if self.random.random() < self.women_share_prob:
                    self.disseminate_women(memory[1], women)
            map(lambda x: x.update_beliefs(), women)

    ##@profile
    def disseminate_midwives(self, memory, recepients):
        LOG.debug("Sharing a memory to midwives.")
        if memory is None or len(recepients) == 0:
            return
        player_type, signals = memory
        LOG.debug("Sharing to midwives: %s" % str(memory))
        if len(memory[1]) > 1:
            LOG.debug("Memory chain length %d" % len(memory[1]))
        tmp_signaller = type(recepients[0])(player_type=player_type)
        
        for recepient in recepients:

            for signal, response in signals:
                recepient.remember(tmp_signaller, signal, response, False)
                recepient.exogenous_update(None, tmp_signaller, signal, signaller_type=player_type)

   #@profile
    def disseminate_women(self, memory, recepients):
        """
        Trigger an exogenous update of this memory into the mind of the
        recepients.
        """
        LOG.debug("Sharing %s to %d women." % (str(memory), len(recepients)))
        if memory is None:
            return
        for mem in memory:
            pt, signal, response, payoff = mem
            tmp_signaller = type(recepients[0])(player_type=pt)
            map(lambda x: x.exogenous_update(signal, response, tmp_signaller, payoff, midwife_type=pt), recepients)


    def share_to(self, pop, fraction):
        """
        Choose some percentage of the population to share to at self.random.
        """
        size = len(pop)
        k = int(round(size * fraction))
        return self.random.sample(pop, k)

    def n_most(self, threshold, memories):
        num = int(round(len(memories)*(1 - abs(threshold))))
        return memories[0:num]


class ShuffledSharingGame(CarryingInformationGame):
    def __str__(self):
        return "shuffled_%s" % super(ShuffledSharingGame, self).__unicode__()

    def play_game(self, players, file_name=""):
        """
        Minor variant, where women are chosen at random rather than in a rotating queue.
        """
        try:
            worker = scoop.worker[0]
        except:
            worker = multiprocessing.current_process()
        LOG.debug("Worker %s playing a game." % worker)
        women, midwives = players

        signaller_generator = self.signaller_fn.generator(random=self.player_random, type_distribution=self.women_weights, 
            agent_args=self.signaller_args, initor=self.signaller_initor,init_args=self.signaller_init_args)
        LOG.debug("Made player generator.")
        rounds = self.rounds
        num_midwives = len(midwives)
        women_res = self.measures_women.dump(None, self.rounds, self)
        mw_res = self.measures_midwives.dump(None, self.rounds, self)
        women_memories = []
        LOG.debug("Starting play.")
        for i in range(rounds):
            self.random.shuffle(women)
            LOG.debug("Shuffled women.")
            players = [women.pop() for j in range(num_midwives)]
            self.random.shuffle(midwives)
            map(self.play_round, players, midwives)
            for x in midwives:
                x.finished += 1
            women_res = self.measures_women.dump(women + players, i, self)
            mw_res = self.measures_midwives.dump(midwives, i, self)
            for woman in players:
                if self.all_played([woman], self.num_appointments):
                    woman.is_finished = True
                    # Add a new naive women back into the mix
                    new_woman = signaller_generator.next()
                    new_woman.started = i
                    new_woman.finished = i
                    women.insert(0, new_woman)
                    LOG.debug("Generated a new player.")
                    if self.women_share_prob > 0 and abs(self.women_share_bias) < 1:
                        women_memories.append(woman.get_memory())
                    for midwife in midwives:
                        midwife.signal_memory.pop(hash(woman), None)
                    del woman
                else:
                    women.insert(0, woman)
                    woman.finished += 1
            # Share information
            LOG.debug("Worker %s prepping share." % worker)
            #Midwives
            try:
                self.share_midwives(midwives)
            except e:
                LOG.debug("Sharing to midwives failed.")
                LOG.debug(e)

            #Women
            try:
                self.share_women(women, women_memories)
            except Exception as e:
                LOG.debug("Sharing to women failed.")
                LOG.debug(e)

            #if scoop_on:
            #    scoop.logger.debug("Worker %s played %d rounds." % (worker, i))
        del women
        del midwives
        del women_memories
        LOG.debug("Worker %s completed a game." % worker)
        return women_res, mw_res

class CaseloadSharingGame(CarryingInformationGame):
    def __str__(self):
        return "caseload_%s" % super(CaseloadSharingGame, self).__unicode__()

    
    def play_game(self, players, file_name=""):
        try:
            worker = scoop.worker[0]
        except:
            worker = multiprocessing.current_process()
        if scoop_on:
            scoop.logger.debug("Worker %s playing a game." % worker)
        else:
            LOG.debug("Playing a game.")
        women, midwives = players
        signaller_generator = self.signaller_fn.generator(random=self.player_random, type_distribution=self.women_weights, 
            agent_args=self.signaller_args, initor=self.signaller_initor,init_args=self.signaller_init_args)
        rounds = self.rounds
        self.random.shuffle(women)
        women_res = self.measures_women.dump(None, self.rounds, self)
        mw_res = self.measures_midwives.dump(None, self.rounds, self)
        women_memories = []
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
        LOG.debug("Assigned caseloads.")
        for i in range(rounds):
            players = [caseloads[midwife].pop() for midwife in midwives]
            #self.random.shuffle(midwives)
            LOG.debug("Playing a round.")
            map(self.play_round, players, midwives)
            for x in midwives:
                x.finished += 1
            LOG.debug("Played.")
            women_for_measure = players + [item for sublist in caseloads.values() for item in sublist]
            women_res = self.measures_women.dump(women_for_measure, i, self)
            mw_res = self.measures_midwives.dump(midwives, i, self)
            #print("Wrote results.")
            for j in range(len(players)):
                woman = players[j]
                women = caseloads[midwives[j]]
                LOG.debug("Working on player %d" % j)
                if self.all_played([woman], self.num_appointments):
                    LOG.debug("Adding a new player")
                    woman.is_finished = True
                    # Add a new naive women back into the mix
                    new_woman = signaller_generator.next()
                    new_woman.started = i
                    new_woman.finished = i
                    women.insert(0, new_woman)
                    LOG.debug("Inserted them.")
                    if self.women_share_prob > 0 and abs(self.women_share_bias) < 1:
                        women_memories.append(woman.get_memory())
                    LOG.debug("Collected memories.")
                    for midwife in midwives:
                        try:
                            midwife.signal_memory.pop(hash(woman), None)
                        except AttributeError:
                            LOG.debug("Not a recognising midwife.")
                    LOG.debug("Pruned from midwives.")
                    del woman
                    LOG.debug("Added a new player.")
                else:
                    women.insert(0, woman)
                    woman.finished += 1
            # Share information
            if scoop_on:
                LOG.debug("Worker %s prepping share." % worker)
            #Midwives
            try:
                self.share_midwives(midwives)
            except Exception as e:
                LOG.debug("Sharing to midwives failed.")
                LOG.debug(e)

            #Women
            try:
                self.share_women(reduce(lambda x, y: x + y, caseloads.values()), women_memories)
            except Exception as e:
                LOG.debug("Sharing to women failed.")
                LOG.debug(e)
            LOG.debug("Played %d rounds." % i)

            #if scoop_on:
            #    scoop.logger.debug("Worker %s played %d rounds." % (worker, i))
        del women
        del midwives
        del women_memories
        if scoop_on:
            scoop.logger.debug("Worker %s completed a game." % worker)
        else:
            LOG.debug("Completed a game.")
        return women_res, mw_res

class SubgroupSharingGame(CarryingInformationGame):
    """
    In this game, rather than share to everybody with some probability,
    everybody shares but people are selected to receive with some 
    probability.
    """

    def __str__(self):
        return "subgroup_%s" % super(SubgroupSharingGame, self).__unicode__()

        #@profile
    def share_midwives(self, midwives):
        """
        Share the recent experiences of each midwife with some probability,
        or erase them if not shared.
        """
        #Worst outcome for a responder
        for midwife in midwives:
            memory = midwife.shareable
            midwife.shareable = None
            if memory is not None:
                LOG.debug("Sharing %s" % str(memory))
                possibles = filter(lambda x: hash(x) != hash(midwife), midwives)
                possibles = filter(lambda x: self.random.random() < self.mw_share_prob, possibles)
                LOG.debug("Selected %d to share to with probability %f" % (len(possibles),  self.mw_share_prob))
                self.disseminate_midwives(memory[1][1], possibles)


    def share_women(self, women, women_memories):
            """
            Go through all the experiences of those who were referred this round,
            and for each one share with probability self.women_share_prob.
            """
            while len(women_memories) > 0:
                memory = women_memories.pop()
                possibles = filter(lambda x: self.random.random() < self.women_share_prob, women)
                self.disseminate_women(memory[1], possibles)
                #And null it
                #women_memories.remove(memory)
            map(lambda x: x.update_beliefs(), women)


class CombinedSharingGame(CarryingInformationGame):
    """
    In this game, rather than share to everybody with some probability,
    everybody shares but people are selected to receive with some 
    probability.
    """

    def __str__(self):
        return "combined_%s" % super(CombinedSharingGame, self).__unicode__()

        #@profile
    def share_midwives(self, midwives):
        """
        Share the recent experiences of each midwife with some probability,
        or erase them if not shared.
        """
        #Worst outcome for a responder
        for midwife in midwives:
            memory = midwife.shareable
            midwife.shareable = None
            if memory is not None and self.random.random() < self.mw_share_prob:
                LOG.debug("Sharing %s" % str(memory))
                possibles = filter(lambda x: hash(x) != hash(midwife), midwives)
                possibles = filter(lambda x: self.random.random() < self.mw_share_prob, possibles)
                LOG.debug("Selected %d to share to with probability %f" % (len(possibles),  self.mw_share_prob))
                self.disseminate_midwives(memory[1][1], possibles)


    def share_women(self, women, women_memories):
            """
            Go through all the experiences of those who were referred this round,
            and for each one share with probability self.women_share_prob.
            """
            while len(women_memories) > 0:
                memory = women_memories.pop()
                if self.random.random() < self.women_share_prob:
                    possibles = filter(lambda x: self.random.random() < self.women_share_prob, women)
                    self.disseminate_women(memory[1], possibles)
                #And null it
                #women_memories.remove(memory)
            map(lambda x: x.update_beliefs(), women)

class ShuffledSubgroupSharingGame(SubgroupSharingGame, ShuffledSharingGame):
    """
    In this game, rather than share to everybody with some probability,
    everybody shares but people are selected to receive with some 
    probability. And players are shuffled.
    """
