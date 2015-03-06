from disclosuregame.Games.game import *
from disclosuregame.Games.referral import *
from disclosuregame.Games.recognition import *
from disclosuregame.Games.carrying import *
from disclosuregame.Games.sharing import *

from disclosuregame.Agents.cpt import *
from disclosuregame.Agents.recognition import *
from disclosuregame.Agents.heuristic import *
from disclosuregame.Agents.payoff import *
from disclosuregame.Agents.rl import *

from disclosuregame.Measures import *
from disclosuregame.Measures.abstract import *

from disclosuregame.experiments import *

from disclosuregame.Util import *

import disclosuregame.Agents.initors

import disclosuregame

import multiprocessing
import itertools
from collections import OrderedDict
import argparse
from os.path import expanduser
import cPickle
import pickle
import gzip
from copy import deepcopy
import sqlite3
import sys
import logging
from random import Random
import time
import os.path

formatter = logging.Formatter('%(asctime)s - [%(levelname)s/%(processName)s] %(message)s')
logger = multiprocessing.log_to_stderr()
logger.handlers[0].setFormatter(formatter)


version = disclosuregame.__version__

def load_kwargs(file_name):
    try:
        kwargs = cPickle.loads(file_name)
    except:    
        with open(file_name, "rb") as f:
            kwargs = cPickle.load(f)
    assert type(kwargs) is list, "%s does not contain a pickled list." % file_name
    #Check this is a valid list of dicts
    valid_args = decision_fn_compare.func_code.co_varnames[:decision_fn_compare.func_code.co_argcount]
    line = 0
    for kwarg in kwargs:
        assert type(kwarg) is dict, "Kwargs %d is not a valid dictionary." % line
        for arg, value in kwarg.items():
            if arg != "game_args":
                assert arg in valid_args, "Kwargs %d, argument: %s is not valid." % (line, arg)

        line +=1

    return kwargs

def arguments():
    parser = argparse.ArgumentParser(
        description='Run some variations of the disclosure game with all combinations of games, signallers and responders provided.')
    parser.add_argument('-g', '--games', type=str, nargs='*',
                   help='A game type to play.', default=['Game', 'CaseloadGame'],
                   choices=['Game', 'CaseloadGame', 'RecognitionGame', 'ReferralGame',
                   'CaseloadRecognitionGame', 'CaseloadReferralGame', 'CarryingGame',
                   'CarryingReferralGame', 'CarryingCaseloadReferralGame', 'CaseloadSharingGame',
                   'CarryingInformationGame', 'ShuffledSharingGame'],
                   dest="games")
    parser.add_argument('-s','--signallers', type=str, nargs='*',
        help='A signaller type.', default=["SharingSignaller"],
        choices=['BayesianSignaller', 'RecognitionSignaller',
        'ProspectTheorySignaller', 'LexicographicSignaller', 'BayesianPayoffSignaller',
        'PayoffProspectSignaller', 'SharingBayesianPayoffSignaller', 'SharingLexicographicSignaller',
        'SharingPayoffProspectSignaller', 'SharingSignaller', 'SharingProspectSignaller',
        'RWSignaller'],
        dest="signallers")
    parser.add_argument('-r','--responders', type=str, nargs='*',
        help='A responder type.', default=["SharingResponder"],
        choices=['BayesianResponder', 'RecognitionResponder', 'ProspectTheoryResponder',
        'LexicographicResponder', 'BayesianPayoffResponder',
        'SharingBayesianPayoffResponder', 'SharingLexicographicResponder',
        'PayoffProspectResponder', 'SharingPayoffProspectResponder',
        'RecognitionResponder', 'RecognitionBayesianPayoffResponder', 'RecognitionLexicographicResponder',
        'PayoffProspectResponder', 'RecognitionPayoffProspectResponder',
        'SharingResponder', 'SharingProspectResponder', 'RWResponder'], dest="responders")
    parser.add_argument('-R','--runs', dest='runs', type=int,
        help="Number of runs for each combination of players and games.",
        default=100)
    parser.add_argument('-i','--rounds', dest='rounds', type=int,
        help="Number of rounds each woman plays for.",
        default=100)
    parser.add_argument('-f','--file', dest='file_name', default="", type=str,
        help="File name prefix for output.")
    parser.add_argument('-t', '--test', dest='test_only', action="store_true", 
        help="Sets test mode on, and doesn't actually run the simulations.")
    parser.add_argument('-p', '--prop_women', dest='women', nargs=3, type=float,
        help="Proportions of type 0, 1, 2 women as decimals.")
    parser.add_argument('-c', '--combinations', dest='combinations', action="store_true",
        help="Run all possible combinations of signallers & responders.")
    parser.add_argument('-d', '--directory', dest='dir', type=str,
        help="Optional directory to store results in. Defaults to user home.",
        default=expanduser("~"), nargs="?")
    parser.add_argument('--pickled-arguments', dest='kwargs', type=str, nargs='?',
        default=None, help="A file containing a pickled list of kwarg dictionaries to be run.")
    #parser.add_argument('--individual-measures', dest='indiv', action="store_true",
    #    help="Take individual outcome measures instead of group level.", default=False)
    parser.add_argument('--abstract-measures', dest='abstract', action="store_true",
        help="Take measures intended for the extended abstract.", default=False)
    parser.add_argument('--log-level', dest='log_level', type=str, choices=['debug',
        'info', 'warniing', 'error'], default='info', nargs="?")
    parser.add_argument('--log-file', dest='log_file', type=str, default='')
    parser.add_argument('--tag', dest='tag', type=str, default='')
    parser.add_argument('--measure-every', dest='measure_freq', type=int,
        help="Measure every x rounds.",
        default=1)
    parser.add_argument('--procs', dest='procs', type=int,
        help="Number of cpus to utilize.",
        default=multiprocessing.cpu_count())

    args, extras = parser.parse_known_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    logger.setLevel(numeric_level)
    if args.log_file != "":
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    file_name = os.path.join(args.dir, args.file_name)
    games = map(eval, args.games)
    if args.combinations:
        players = list(itertools.product(map(eval, set(args.signallers)), map(eval, set(args.responders))))
    else:
        players = zip(map(eval, args.signallers), map(eval, args.responders))
    kwargs = {'runs':args.runs, 'rounds':args.rounds, 'nested':False, 'file_name':file_name, 'tag':args.tag}
    if args.women is not None:
        kwargs['women_weights'] = args.women
    #if args.indiv:
    #    kwargs['measures_midwives'] = indiv_measures_mw()
    #    kwargs['measures_women'] = indiv_measures_women()
    if args.abstract:
        logger.debug("Using abstract measures.")
        kwargs['measures_midwives'] = abstract_measures_mw()
        kwargs['measures_women'] = abstract_measures_women()
    kwargs = [kwargs]
    if args.kwargs is not None:
        try:
            new_args = load_kwargs(args.kwargs)
            old = kwargs[0]
            kwargs = []
            for arg in new_args:
                tmp = old.copy()
                tmp.update(arg)
                kwargs.append(tmp)
        except IOError:
            logger.info("Couldn't open %s." % args.kwargs)
            raise
        except cPickle.UnpicklingError:
            logger.info("Not a valid pickle file.")
            raise
    return games, players, kwargs, args.runs, args.test_only, file_name, args.kwargs, args.procs


def make_players(constructor, num=100, weights=[1/3., 1/3., 1/3.], nested=False,
    signaller=True, player_args={}, random=Random()):
    women = []
    player_type = 0
    player_args = deepcopy(player_args)
    player_args['seed'] = random.random()
    for weight in weights:
        for i in range(int(round(weight*num))):
            if len(women) == num: break
            if nested:
                if signaller:
                    women.append(DollSignaller(player_type=player_type, child_fn=constructor))
                else:
                    women.append(constructor(player_type=player_type, **player_args))    
            else:
                women.append(constructor(player_type=player_type, **player_args))
        player_type += 1
    while len(women) < num:
        player_type = 0
        if nested:
            if signaller:
                women.append(DollSignaller(player_type=player_type, child_fn=constructor))
            else:
                women.append(constructor(player_type=player_type, **player_args))    
        else:
            women.append(constructor(player_type=player_type, **player_args))
    return women

def params_dict(signaller_rule, responder_rule, mw_weights, women_weights, game, rounds,
    signaller_args, responder_args, tag, signaller_init_args, responder_init_args, 
    signaller_initor, responder_initor):
    params = OrderedDict()
    params['game'] = str(game)
    params['decision_rule_responder'] = responder_rule
    params['decision_rule_signaller'] = signaller_rule
    params['signaller_generator'] = signaller_initor.__name__
    params['responder_generator'] = responder_initor.__name__
    params['caseload'] = game.is_caseloaded()
    for i in range(len(mw_weights)):
        params['mw_%d' % i] = mw_weights[i]

    for i in range(len(women_weights)):
        params['women_%d' % i] = women_weights[i]
    params['max_rounds'] = rounds
    params['tag'] = tag
    for k, v in signaller_args.items():
        params['signaller_%s' % k] = v
    for k, v in responder_args.items():
        params['responder_%s' % k] = v
    for k, v in signaller_init_args.items():
        params['signaller_init_%s' % k] = v
    for k, v in responder_init_args.items():
        params['responder_init_%s' % k] = v
    for i in range(len(game.type_weights)):
        for j in range(len(game.type_weights[i])):
            params['weight_%d_%d' % (i, j)] = game.type_weights[i][j]
    return params

def decision_fn_compare(signaller_fn=BayesianSignaller, responder_fn=BayesianResponder,
    num_midwives=100, num_women=1000, 
    runs=1, game=None, rounds=100,
    mw_weights=[80/100., 15/100., 5/100.], women_weights=[1/3., 1/3., 1/3.], women_priors=None, seeds=None,
    women_modifier=None, measures_women=measures_women(), measures_midwives=measures_midwives(),
    nested=False, mw_priors=None, file_name="", responder_args={}, signaller_args={}, tag="", measure_freq=1,
    responder_initor=initors.responder, signaller_initor=initors.signaller, signaller_init_args={},
    responder_init_args={}):

    if game is None:
        game = Game()
    if mw_priors is not None:
        game.type_weights = mw_priors
    measures_midwives.dump_every = measure_freq
    measures_women.dump_every = measure_freq
    game.measures_midwives = measures_midwives
    game.measures_women = measures_women
    game.signaller_args = signaller_args
    game.responder_args = responder_args
    game.women_weights = women_weights
    game.signaller_initor = signaller_initor
    game.signaller_init_args = signaller_init_args
    game.signaller_fn = signaller_fn

    params = params_dict(str(signaller_fn()), str(responder_fn()), mw_weights, women_weights, game, rounds,
        signaller_args, responder_args, tag, signaller_init_args, responder_init_args, signaller_initor, responder_initor)
    signaller_init_args['woman_baby_payoff'] = game.woman_baby_payoff
    signaller_init_args['woman_social_payoff'] = game.woman_social_payoff
    responder_init_args['midwife_payoff'] = game.midwife_payoff
    for key, value in params.items():
        game.parameters[key] = value
    logger.debug("Parameters")
    logger.debug("----------")
    logger.debug(game.parameters)
    game.rounds = rounds
    random = Random()
    if seeds is None:
        #seeds = [random.random() for x in range(runs)]
        seeds = range(runs)
    player_pairs = []
    #for i in range(runs):
    i =  0
    while i < runs:
        logger.debug("Generated run %d of %d" % (i, runs))
        #game.parameters['seed'] = i
        # Parity across different conditions but random between runs.
        random = Random(seeds[i])
        game.seed = seeds[i]
        game.random = Random(seeds[i])
        try:
          game.player_random = Random(game.random.random())
        except AttributeError:
          pass
          
        #random.seed(1)
        #logger.info "Making run %d/%d on %s" % (i + 1, runs, file_name)

        #Make players and initialise beliefs

        women_generator = signaller_fn.generator(random=game.player_random, type_distribution=women_weights, agent_args=signaller_args, initor=signaller_initor,init_args=signaller_init_args)
        women = [women_generator.next() for x in range(num_women)]

        if women_modifier is not None:
            women_modifier(women)
        #logger.info("Set priors.")
        #print responder_args
        mw_generator = responder_fn.generator(random=game.player_random, type_distribution=mw_weights, agent_args=responder_args, initor=responder_initor,init_args=responder_init_args)
        mw = [mw_generator.next() for x in range(num_midwives)]
        #logger.info("Set priors.")
        #player_pairs.append((deepcopy(game), women, mw))
        yield (deepcopy(game), women, mw)
        i += 1

        #pair = game.play_game(women, mw, rounds=rounds)
    #played = map(lambda x: game.play_game(x, "%s_%s" % (file_name, str(game))), player_pairs)
    #logger.info("Ran a set of parameters.")
    #return player_pairs

def play_game(config):
    """
    Play a game.
    """
    game, women, midwives = config
    return game.play_game((women, midwives))


def make_work(queue, kwargs, num_consumers, kill_queue):
    logger.info("Starting make work process.")
    i = 1
    while len(kwargs) > 0:
        if not kill_queue.empty():
            logger.info("Poison pill in the kill queue. Not making more jobs.")
            break
        exps = decision_fn_compare(**kwargs.pop())
        for exp in exps:
            if not kill_queue.empty():
                logger.info("Poison pill in the kill queue. Not making more jobs.")
                break
            logger.info("Enqueing experiment %d" %  i)
            queue.put((i, exp))
            i += 1
    for i in range(num_consumers):
        logger.info("Sending finished signal to queue.")
        queue.put(None)
    queue.put(None)
    logger.info("Ending make work process.")


def do_work(queueIn, queueOut, kill_queue):
    """
    Consume games, play them, then put their results in the output queue.
    """
    logger.info("Starting do work process.")
    while True:
        try:
            if not kill_queue.empty():
                logger.info("Poison pill in the kill queue. Stopping.")
                break
            number, config = queueIn.get()
            logger.info("Running game %d." % number)
            res = (number, play_game(config))
            queueOut.put(res)
            del config
        except MemoryError:
            raise
            break
        except AssertionError:
            kill_queue.put(None)
            raise
            break
        except TypeError:
            break
        except:
            raise
            break
    logger.info("Ending do work process.")

def write(queue, db_name, kill_queue):
    logger.info("Starting write process.")
    while True:
        try:
            number, res = queue.get()
            #print res
            women_res, mw_res = res
            logger.info("Writing game %d." % number)
            women_res.write_db("%s_women" % db_name)
            mw_res.write_db("%s_mw" % db_name)
            del women_res
            del mw_res
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error("SQLite failure.")
            logger.error(e)
            kill_queue.put(None)
            raise
            break
        except:
            kill_queue.put(None)
            raise
            break
    logger.info("Ending write process.")


def experiment(file_name, game_fns=[Game, CaseloadGame], 
    agents=[(ProspectTheorySignaller, ProspectTheoryResponder), (BayesianSignaller, BayesianResponder)],
    kwargs=[{}], procs=1):
    run_params = []
    for pair in agents:
        for game_fn in game_fns:
            for kwarg in kwargs:
                arg = kwarg.copy()
                try:
                    game = game_fn(**arg.pop('game_args', {}))
                except TypeError as e:
                    logger.error("Wrong arguments for this game type.")
                    logger.error(e)
                    raise
                #kwarg.update({'measures_midwives': measures_midwives, 'measures_women': measures_women})
                arg['game'] = game
                arg['signaller_fn'] = pair[0]
                arg['responder_fn'] = pair[1]
                run_params.append(arg)
    kw_experiment(run_params, file_name, procs)

def kw_experiment(kwargs, file_name, procs):
    """
    Run a bunch of experiments in parallel. Experiments are
    defined by a list of keyword argument dictionaries.
    """
    num_consumers = procs
    #Make tasks
    jobs = multiprocessing.Queue(num_consumers)
    kill_queue = multiprocessing.Queue(1)
    results = multiprocessing.Queue()
    producer = multiprocessing.Process(target = make_work, args = (jobs, kwargs, num_consumers, kill_queue))
    producer.start()
    calcProc = [multiprocessing.Process(target = do_work , args = (jobs, results, kill_queue)) for i in range(num_consumers)]
    writProc = multiprocessing.Process(target = write, args = (results, file_name, kill_queue))
    writProc.start()

    for p in calcProc:
        p.start()
    for p in calcProc:
        try:
            if not kill_queue.empty():
                logger.info("Poison pill in the kill queue. Terminating.")
                p.terminate()
            p.join()
        except KeyboardInterrupt:
            for p in calcProc:
                p.terminate()
            producer.terminate()
            kill_queue.put(None)
    results.put(None)
    writProc.join()
    while True:
        if jobs.get() is None:
            break
        logger.info("Flushing jobs queue..")
    producer.join()


def main():
    games, players, kwargs, runs, test, file_name, args_path, procs = arguments()
    logger.info("Version %s" % version)
    logger.info("Running %d game type%s, with %d player pair%s, and %d run%s of each." % (
        len(games), "s"[len(games)==1:], len(players), "s"[len(players)==1:], runs, "s"[runs==1:]))
    logger.info("Total simulations runs is %d" % (len(games) * len(players) * runs * len(kwargs)))
    logger.info("File is %s" % file_name)
    logger.info("Using %d processors." % procs)
    if test:
        logger.info("This is a test of the emergency broadcast system. This is only a test.")
    else:
        start = time.clock()
        experiment(file_name, games, players, kwargs=kwargs, procs=procs)
        print "Ran in %f" % (time.clock() - start)

if __name__ == "__main__":
    main()