import multiprocessing
import itertools
from collections import OrderedDict
import argparse
from os.path import expanduser
import cPickle
from copy import deepcopy
import sqlite3
import logging
from random import Random
import time
import os, sys
import gc
import platform
from Queue import Full, Empty

from disclosuregame.Measures.space_measures import *
from disclosuregame.experiments import *
import disclosuregame

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
    try:
        assert type(kwargs) is list, "%s does not contain a pickled list." % file_name
    except:
        kwargs = [kwargs]
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
                   help='A game type to play.', default=['SimpleGame', 'CaseloadGame'],
                   choices=['SimpleGame', 'CaseloadGame', 'RecognitionGame', 'ReferralGame',
                   'CaseloadRecognitionGame', 'CaseloadReferralGame', 'CarryingGame',
                   'CarryingReferralGame', 'CarryingCaseloadReferralGame', 'CaseloadSharingGame',
                   'CarryingInformationGame', 'ShuffledSharingGame', 'SubgroupSharingGame', 'CombinedSharingGame',
                            'DeathGame', 'DeathAndSharingGame'],
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
    parser.add_argument('--space-measures', dest='space', action="store_true",
        help="Take measures for risk, value, signal space.", default=False)
    parser.add_argument('--player-types', dest='measure_sigs', type=int, nargs="*",
        help="Types of players for measurement.", default=[0, 1])
    parser.add_argument('--log-level', dest='log_level', type=str, choices=['debug',
        'info', 'warning', 'error'], default='info', nargs="?")
    parser.add_argument('--log-file', dest='log_file', type=str, default='')
    parser.add_argument('--tag', dest='tag', type=str, default='')
    parser.add_argument('--measure-every', dest='measure_freq', type=int,
        help="Measure every x rounds.",
        default=1)
    parser.add_argument('--procs', dest='procs', type=int,
        help="Number of cpus to utilize.",
        default=multiprocessing.cpu_count())
    parser.add_argument('--save-state-file', dest='state_file', type=str,
        help="File to record state in if the simulation needs to be resumed.",
        default=None)

    parser.add_argument('--load-state-file', dest='load_state_file', type=str,
        help="File to reload state from, will discard all other options except --save-state-file.", default=None)

    args, extras = parser.parse_known_args()

    savestate = args.state_file

    it = set()

    try:
        if args.load_state_file:
            with open(args.load_state_file, "rb") as state:
                args, it = cPickle.load(state)
    except IOError as e:
        logger.debug("Failed to load state file.")
        logger.debug(e)

    if savestate:
        args.state_file = savestate

    if args.state_file:
        with open(args.state_file, "wb") as state:
            cPickle.dump((args, it), state)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)

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
    kwargs = {'runs':args.runs, 'rounds':args.rounds, 'tag':args.tag}
    if args.women is not None:
        kwargs['women_weights'] = args.women
    #if args.indiv:
    #    kwargs['measures_midwives'] = indiv_measures_mw()
    #    kwargs['measures_women'] = indiv_measures_women()
    if args.abstract:
        logger.debug("Using abstract measures.")
        kwargs['measures_midwives'] = abstract_measures_mw(freq=args.measure_freq, signals=args.measure_sigs)
        kwargs['measures_women'] = abstract_measures_women(freq=args.measure_freq, signals=args.measure_sigs)
    else:
        kwargs['measures_midwives'] = measures_midwives(freq=args.measure_freq, signals=args.measure_sigs)
        kwargs['measures_women'] = measures_women(freq=args.measure_freq, signals=args.measure_sigs)
    if args.space:
        logger.debug("Using space measures.")
        kwargs['measures_midwives'] = space_measures_mw(base=kwargs['measures_midwives'], signals=args.measure_sigs)
        kwargs['measures_women'] = space_measures_women(base=kwargs['measures_women'], signals=args.measure_sigs)
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
    return games, players, kwargs, args.runs, args.test_only, file_name, args.kwargs, args.procs, args.state_file, it


def make_players(constructor, num=100, weights=None, nested=False, signaller=True, player_args=None, random=Random()):
    if not player_args:
        player_args = {}
    if not weights:
        weights = [1 / 3., 1 / 3., 1 / 3.]
    women = []
    player_type = 0
    player_args = deepcopy(player_args)
    player_args['seed'] = random.random()
    for weight in weights:
        for i in range(int(round(weight*num))):
            if len(women) == num: break
            women.append(constructor(player_type=player_type, **player_args))
        player_type += 1
    while len(women) < num:
        player_type = 0
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

def decision_fn_compare(signaller_fn=BayesianSignaller, responder_fn=BayesianResponder, num_midwives=100,
                        num_women=1000, runs=1, game=None, rounds=100, mw_weights=None, women_weights=None,
                        seeds=None, women_modifier=None, measures_women=measures_women(),
                        measures_midwives=measures_midwives(), mw_priors=None, responder_args=None, signaller_args=None, tag="",
                        responder_initor=initors.responder, signaller_initor=initors.signaller,
                        signaller_init_args=None, responder_init_args=None):
    
    if not women_weights:
        women_weights = [1 / 3., 1 / 3., 1 / 3.]
    if not mw_weights:
        mw_weights = [80 / 100., 15 / 100., 5 / 100.]
    if responder_args is None:
        responder_args = {}
    else:
        responder_args = deepcopy(responder_args)
    if signaller_args is None:
        signaller_args = {}
    else:
        signaller_args = deepcopy(signaller_args)
    if signaller_init_args is None:
        signaller_init_args = {}
    else:
        responder_init_args = deepcopy(responder_init_args)
    if responder_init_args is None:
        responder_init_args = {}
    else:
        signaller_init_args = deepcopy(signaller_init_args)

    if game is None:
        game = SimpleGame()
    if mw_priors is not None:
        game.type_weights = mw_priors
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
    if seeds is None:
        #seeds = [random.random() for x in range(runs)]
        seeds = range(runs)
    #for i in range(runs):
    i =  0
    while i < runs:
        logger.debug("Generated run %d of %d" % (i, runs))
        #game.parameters['seed'] = i
        # Parity across different conditions but random between runs.
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


def make_work(queue, kwargs, kill_queue):
    logger.info("Starting make work process.")
    i = 1
    queue.cancel_join_thread()
    try:
        while len(kwargs) > 0:
            assert kill_queue.empty()
            exps = decision_fn_compare(**kwargs.pop())
            for exp in exps:
                logger.info("Enqueing experiment %d" %  i)
                while True:
                    try:
                        assert kill_queue.empty()
                        queue.put_nowait((i, exp))
                        #with open("/Users/jono/game", "w") as fout:
                        #    cPickle.dump(exp, fout)
                        break
                    except Full:
                        logger.debug("Waiting for space in the jobs queue.")
                        pass
                logger.info("Enqueued experiment %d" %  i)
                i += 1
    except Exception as e:
        logger.info("Poison pill in the kill queue. Not making more jobs.")
        try:
            queue.put_nowait(None)
        except Full:
            pass
        raise e
    finally:
        logger.info("Closing the jobs queue.")
        queue.close()
        logger.info("Ending make work process.")


def workit(kwargs, skiplist=None):
    logger.info("Starting make work process.")
    i = 0
    if skiplist is None:
        skiplist = set()
    while len(kwargs) > 0:
        exps = decision_fn_compare(**kwargs.pop())
        for exp in exps:
            i += 1
            logger.info("Enqueing experiment %d" %  i)
            if i not in skiplist:
                yield (i, exp)
    logger.info("Ending make work process.")


def doplay(config):
    try:
        number, config = config
        logger.info("Playing game %d" % number)
        res = (number, play_game(config))
    except:
        raise KeyboardInterruptError
    return res


def writer(results, db_name, state_file=None):
    for number, result in results:
        logger.info("Writing game %d." % number)
        women_res, mw_res = result
        women_res.write_db("%s_women" % db_name)
        mw_res.write_db("%s_mw" % db_name)
        del women_res
        del mw_res
        gc.collect()
        logger.info("Wrote game %d." % number)
        if state_file:
            try:
                with open(state_file, "rb") as state:
                    savestate, skiplist = cPickle.load(state)
                    skiplist.add(number)
                    savestate = (savestate, skiplist)
                with open(state_file, "wb") as state:
                    cPickle.dump(savestate, state)
            except Exception as e:
                logger.error("Failed to save state!")
                raise(e)

class KeyboardInterruptError(Exception): pass

def run(kwargs, file_name, procs, state_file=None, start_point=None):
    if start_point is None:
        start_point = set()
    pool = multiprocessing.Pool(procs)
    try:
        jobqueue = workit(kwargs, start_point)
        writer(pool.imap_unordered(doplay, jobqueue), file_name, state_file=state_file)
    except KeyboardInterruptError:
        pool.terminate()
        sys.exit(1)

def do_work(queuein, queueout, kill_queue):
    """
    Consume games, play them, then put their results in the output queue.
    """
    queuein.cancel_join_thread()
    
    logger.info("Starting do work process.")
    while True:
        try:
            assert kill_queue.empty()
            number, config = queuein.get(timeout=10) #Shouldn't ever need to wait
            logger.info("Running game %d." % number)
            res = (number, play_game(config))
            queueout.put(res, timeout=120) #Wait at most 2 mins for writes
            del config
        except MemoryError as e:
            logger.error(e)
            queuein.cancel_join_thread()
            try:
                logger.info("Dropping poison.")
                kill_queue.put_nowait(None)
            except Full:
                pass
            raise
        except AssertionError as e:
            logger.error(e)
            try:
                logger.info("Dropping poison.")
                kill_queue.put_nowait(None)
            except Full:
                pass
            raise
        except Empty:
            logger.info("No more work.")
            break
        except Exception as e:
            logger.error(e)
            raise
        gc.collect()
    logger.info("Ending do work process.")

def write(queue, db_name, kill_queue):
    logger.info("Starting write process.")
    while True:
        try:
            assert kill_queue.empty()
            number, res = queue.get_nowait()
            #print res
            women_res, mw_res = res
            logger.info("Writing game %d." % number)
            try:
                logger.info("Queue length is %d" % queue.qsize())
            except NotImplementedError:
                #Don't bother if not implemented
                pass
            women_res.write_db("%s_women" % db_name)
            mw_res.write_db("%s_mw" % db_name)
            del women_res
            del mw_res
            gc.collect()
            logger.info("Wrote game %d." % number)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error("SQLite failure.")
            logger.error(e)
            try:
                logger.info("Dropping poison.")
                kill_queue.put_nowait(None)
            except Full:
                pass
            raise
        except TypeError as e:
            raise e
        except Empty:
            pass
        except AssertionError as e:
            logger.error(e)
            break
        except Exception as e:
            logger.error(e)
            kill_queue.put_nowait(None)
            raise
    logger.info("Ending write process.")
    logger.info("Results queue empty: %s" % str(queue.empty()))


def experiment(file_name, game_fns=None, agents=None, kwargs=None, procs=1, state_file=None, start_point=0):
    if not agents:
        agents = [(ProspectTheorySignaller, ProspectTheoryResponder), (BayesianSignaller, BayesianResponder)]
    if not game_fns:
        game_fns = [SimpleGame, CaseloadGame]
    if not kwargs:
        kwargs = [{}]
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
    #kw_experiment(run_params, file_name, procs)
    run(run_params, file_name, procs, state_file=state_file, start_point=start_point)

def kw_experiment(kwargs, file_name, procs):
    """
    Run a bunch of experiments in parallel. Experiments are
    defined by a list of keyword argument dictionaries.
    """
    host = platform.uname()[1]
    num_consumers = procs
    #Make tasks
    jobs = multiprocessing.Queue(num_consumers)
    kill_queue = multiprocessing.Queue(1)
    results = multiprocessing.Queue()
    producer = multiprocessing.Process(target = make_work, name="%s: Producer" % host, args = (jobs, kwargs, kill_queue))
    producer.start()
    calcproc = [multiprocessing.Process(target = do_work, name="%s: Simulation process %d" % (host, i),
                                        args = (jobs, results, kill_queue)) for i in range(num_consumers)]
    writproc = multiprocessing.Process(target = write, name="%s Writer" % host, args = (results, file_name, kill_queue))
    writproc.start()

    for p in calcproc:
        p.start()
    while any(map(lambda p: p.is_alive(), calcproc)):
            try:
                assert kill_queue.empty()
            except (KeyboardInterrupt, AssertionError, MemoryError):
                for proc in calcproc:
                    logger.info("Terminating %s" % str(proc))
                    proc.terminate()
                logger.info("Poison pill")
                try:
                    kill_queue.put_nowait(None)
                except Full:
                    logger.info("Poison already in place.")
                break
    try:
        kill_queue.put_nowait(None)
    except Full:
        logger.info("Poison already in place.")
    logger.info("Closing results.")
    try:
        results.put(None, block=False)
    except Full:
        pass
    logger.info("Flushing jobs.")
    try:
        while not jobs.empty():
            logger.info("Flushing a job.")
            jobs.get_nowait()
            logger.info("Flushed.")
    except Empty:
        logger.info("Jobs flushed.")
    if writproc.is_alive():
        logger.info("Joining writer.")
        writproc.join(60)
    if producer.is_alive():
        logger.info("Joining producer.")
        producer.join(60)
        producer.terminate()
    logger.info("Done.")


def main():
    games, players, kwargs, runs, test_flag, file_name, args_path, procs, state_file, it = arguments()
    logger.info("Version %s" % version)
    logger.info("Running %d game type%s, with %d player pair%s, and %d run%s of each." % (
        len(games), "s"[len(games)==1:], len(players), "s"[len(players)==1:], runs, "s"[runs==1:]))
    logger.info("Total simulations runs is %d" % (len(games) * len(players) * runs * len(kwargs)))
    logger.info("File is %s" % file_name)
    logger.info("Using %d processors." % procs)
    if test_flag:
        logger.info("This is a test of the emergency broadcast system. This is only a test.")
    else:
        start = time.time()
        experiment(file_name, games, players, kwargs=kwargs, procs=procs, state_file=state_file, start_point=it)
        print "Ran in %f" % (time.time() - start)

if __name__ == "__main__":
    main()
