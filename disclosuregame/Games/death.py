from sharing import CarryingInformationGame

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

class DeathGame(CarryingInformationGame):
    """
    A variation on the regular game where agents play ends with probability
    proportionate to their age.
    """
    def all_played(self, women, rounds=12):
        for woman in women:
            age = self.current_round - woman.started
            livep = 1./(age + 1)
            livep = 1/10.
            if(woman.random.random() > livep) and not woman.is_finished:
                return False
            LOG.debug("Player %d finished after %d rounds." % (hash(woman), woman.rounds))
        return True

    def post_round(self, players, signallers, responders, **kwargs):
        players += signallers
        signallers[:] = []
        super(DeathGame, self).post_round(players, signallers, responders, **kwargs)

    def __str__(self):
        return "death_%s" % super(DeathGame, self).__unicode__()

class DeathAndSharingGame(DeathGame):
    def post_round(self, players, signallers, responders, **kwargs):
        """
        Actions to perform after playing a round.
        :param **kwargs:
        """
        players += signallers
        signallers[:] = []
        new_count = 0
        total_count = 0
        try:
            worker = scoop.worker[0]
        except:
            worker = multiprocessing.current_process()

        for woman in players:
            total_count += 1
            #Now anybody can share..
            if self.random.random() < self.women_share_prob:
                self.women_memories.append((woman.ident, woman.get_memory()))

            if woman.is_finished:
                # Add a new naive women back into the mix
                new_count += 1
                new_woman = self.signaller_generator.next()
                try:
                    assert new_woman.ident not in self.measures_women.measures["type_0_pop"].hash_bucket
                    assert new_woman.ident not in self.measures_women.measures["type_1_pop"].hash_bucket
                except:
                    print new_woman.ident
                    raise

                new_woman.started = self.current_round
                new_woman.finished = self.current_round
                signallers.insert(0, new_woman)
                LOG.debug("Generated a new player.")
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
        LOG.debug("Made %d new players on round %d. Started with %d." % (new_count, self.current_round, total_count))

    def share_women(self, women):
            """
            Go through all the experiences of those who were referred this round,
            and for each one share with probability self.women_share_prob.
            """
            while len(self.women_memories) > 0:
                ident, memory = self.women_memories.pop()
                targets = filter(lambda woman: woman.ident != ident, women)
                self.disseminate_women(memory[1], targets)
            map(lambda x: x.update_beliefs(), women)

    def __str__(self):
        return "sharinganddeath_%s" % super(DeathAndSharingGame, self).__unicode__()