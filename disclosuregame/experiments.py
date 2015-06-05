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
import disclosuregame.Agents.initors

from disclosuregame.Measures import *
from disclosuregame.Measures.abstract import *

from disclosuregame.experiments import *

from disclosuregame.Util import random_expectations


from multiprocessing import Pool, Queue
import multiprocessing
import itertools
from collections import OrderedDict
import argparse
from os.path import expanduser
import os.path

import cPickle
import gzip, csv


def test():
    game_args = {"baby_payoff":7900, "mid_baby_payoff":5000,"referral_cost":7800,
     "mid_mid":500, "mid_low":0, "low_mid":0,"low_low":0, "mw_share_prob":0.25}
    args = [{'game_args': game_args, 
            'signaller_args':{'share_weight':0., "signals":[0, 1]},
            'responder_args':{'share_weight':0.25, "signals":[0, 1]},
            'mw_weights':[1., 0.], 
            'women_weights':[.75, .25],
            'signaller_initor':initors.ebreferral_logisticstigma,
            'responder_initor':initors.normalresponder}]
    args.append({'game_args': {"baby_payoff":7900, "mid_baby_payoff":5000,"referral_cost":7800,
     "mid_mid":500, "mid_low":0, "low_mid":0,"low_low":0, "women_share_prob":0.25}, 
            'signaller_args':{'share_weight':0.25, "signals":[0, 1]},
            'responder_args':{'share_weight':0., "signals":[0, 1]},
            'mw_weights':[1., 0.], 
            'women_weights':[.75, .25],
            'signaller_initor':initors.ebreferral_logisticstigma,
            'responder_initor':initors.normalresponder})
    args.append({'game_args': {"baby_payoff":7900, "mid_baby_payoff":5000,"referral_cost":7800,
     "mid_mid":500, "mid_low":0, "low_mid":0,"low_low":0, "women_share_prob":0.25,  "mw_share_prob":0.25}, 
            'signaller_args':{'share_weight':0.25, "signals":[0, 1]},
            'responder_args':{'share_weight':0.25, "signals":[0, 1]},
            'mw_weights':[1., 0.], 
            'women_weights':[.75, .25],
            'signaller_initor':initors.ebreferral_logisticstigma,
            'responder_initor':initors.normalresponder})

    return args

def scale_weights(weights, top):
    scaling = top / float(sum(weights))
    for i in range(len(weights)):
        weights[i] *= scaling
    return weights


def make_random_patients(signaller, num=1000, weights=None):
    if not weights:
        weights = [1 / 3., 1 / 3., 1 / 3.]
    women = []
    player_type = 0
    for weight in weights:
        for i in range(weight*100):
            women.append(signaller(player_type=player_type))
        player_type += 1
    return women


def make_random_weights(num=1000):
    weights = []
    for i in range(num):
        weights.append((random_expectations(), [random_expectations(breadth=2) for x in range(3)]))
    return weights


def inject_type_belief(weights, num):
    """
    Return a function that will modify the type distribution priors of num 
    random women to be those in weights and restarts faith.
    """
    def f(women):
        target = random.sample(women, num)
        signals = [0, 1, 2]
        for agent in target:
            agent.response_belief = dict([(signal, dict([(response, []) for response in [0, 1]])) for signal in signals])
            agent.type_distribution = dict([(signal, []) for signal in signals])
            agent.type_weights = weights
            agent.update_beliefs(None, None, None)
    return f


def make_random_midwives(responder, num=100, weights=None):
    if not weights:
        weights = [80 / 100., 15 / 100., 5 / 100.]
    midwives = []
    for i in range(num):
        midwives.append(responder(player_type=weighted_choice(zip([0, 1, 2], weights))))
    return midwives

def proportions(num):
    """
    Generate some number of combinations of 3
    random numbers that sum to 1.
    """
    proportions = []
    for i in range(num):
        initial = 100
        result = []
        for j in range(2):
            n = random.randint(0, initial)
            initial -= n
            result.append(n)
        result.append(initial)

        proportions.append([x/100. for x in result])
    return proportions

def proportions_experiment():
    """
    Sample space for type proportions.
    """
    mw = proportions(10000)
    w = proportions(10000)
    kwargs = []
    for i in range(10000):
        kwarg = {'women_weights':w[i], 'mw_weights':mw[i]}
        kwargs.append(kwarg)
    return kwargs

def naive_partition(resolution=0.1):
    """
    Generator that yields combinations of three numbers that sum to
    one.
    """
    parts = []
    for x in itertools.product(xrange(0, 101, int(100 * resolution)), repeat=3):
        if sum(x) == 100:
            yield map(lambda y: y / 100., x)
            #parts.append(map(lambda y: y / 100., x))
    #return parts

def qualitative_trends():
    args = {'game_args': {'mw_share_prob':0.,
            'women_share_prob':0.,
            'type_weights':[[10., 1., 1.], [1., 10., 1.], [1., 1., 10.]],
            'baby_payoff':10, 'referral_cost':9}, 
            'signaller_args':{'share_weight':0.},
            'responder_args':{'share_weight':0.},
            'mw_weights':[0.85, 0.1, 0.05], 
            'women_weights':[1/3., 1/3., 1/3.]}
    return [args]

def naive_proportions(key='women_weights', resolution=0.1, chunksize=None):
    """
    Produces a list of lists of kwargs for type proportions split into
    chunks.
    """
    w = naive_partition(resolution)
    kwargs = []
    results = []
    for x in w:
        kwarg = {key:x}
        kwargs.append(kwarg)
        if chunksize is not None and len(kwargs) >= chunksize:
            results.append(kwargs)
            kwargs = []
    results.append(kwargs)
    return results

def priors_experiment():
    """
    Sample space for midwives' priors.
    """
    mw = [random_expectations(1, 3, 1, 50) for x in range(10000)]
    kwargs = []
    for i in range(10000):
        kwarg = {'mw_priors':mw[i]}
        kwargs.append(kwarg)
    return kwargs

def synthetic_caseload():
    """
    Synthetic caseload with a harsh midwife.
    """
    kwargs = []
    for i in range(3):
        mw_weights = [0]*3
        mw_weights[i] = 1

        kwargs.append({'num_midwives':1, 'num_women':10, 'mw_weights':mw_weights})
    return kwargs

def abstract_experiment(chunksize=None):
    kwargs = []
    results = []
    for x in itertools.product((y*resolution for y in range(0, int(1/resolution) + 1) ), repeat=2):
        kwarg = {'game_args': {'mw_share_prob':x[0]}, 'responder_args':{'share_weight':x[1]}}
        kwargs.append(kwarg)
        if chunksize is not None and len(kwargs) >= chunksize:
            results.append(kwargs)
            kwargs = []
    results.append(kwargs)
    return results

def mw_sharing_experiment(resolution=0.1, chunksize=None):
    kwargs = []
    results = []
    for x in itertools.product((y*resolution for y in range(0, int(1/resolution) + 1) ), repeat=2):
        kwarg = {'game_args': {'mw_share_prob':x[0]}, 'responder_args':{'share_weight':x[1]}}
        kwargs.append(kwarg)
        if chunksize is not None and len(kwargs) >= chunksize:
            results.append(kwargs)
            kwargs = []
    results.append(kwargs)
    return results

def w_sharing_experiment(resolution=0.1, chunksize=None):
    kwargs = []
    results = []
    for x in itertools.product((y*resolution for y in range(0, int(1/resolution) + 1) ), repeat=2):
        kwarg = {'game_args': {'mw_share_prob':0., 'mw_share_bias':1.,
            'women_share_prob':x[0]}, 'signaller_args':{'share_weight':x[1]}}
        kwargs.append(kwarg)
        if chunksize is not None and len(kwargs) >= chunksize:
            results.append(kwargs)
            kwargs = []
    results.append(kwargs)
    return results

def w_simple_sharing_experiment(chunksize=None):
    kwargs = []
    results = []
    for x in [0., 0.25, 0.5, 0.75, 1.]:
        kwarg = {'game_args': {'mw_share_prob':0., 'mw_share_bias':1.,
            'women_share_prob':x}, 'signaller_args':{'share_weight':x}}
        kwargs.append(kwarg)
        if chunksize is not None and len(kwargs) >= chunksize:
            results.append(kwargs)
            kwargs = []
    results.append(kwargs)
    return results

def args_write(args, directory, name):
    files = []
    for i in xrange(len(args)):
        target = os.path.join(directory, "%s_%d.args" % (name, i))
        files.append(target)
        print "Writing %s" % target
        f = open(target, "wb")
        cPickle.dump(args[i], f, cPickle.HIGHEST_PROTOCOL)
        f.close()
    return files



def midwife_priors():
    priors =  [[[x, 1., 1.], [1., x, 1.], [1., 1., x]] for x in xrange(5, 51, 5)]
    for i in range(4):
        priors.append([[i + 1., 1., 1.], [1., i + 1., 1.], [1., 1., i + 1.]])
    run_params = []
    for prior in priors:
        args = {'mw_priors':prior}
        run_params.append(args)
    return run_params


def lhs_sampling(samples, chunksize=None):
    """
    Read in a csv containing samples generated by R and transform it into a
    set of argument dictionaries.
    Expects the following fields:
    'mw_0', 'mw_1', 'mw_2', 'women_0', 'women_1', 'women_2' value between 0 & 1 
    'mw_share_prob', 'women_share_prob', 'responder_share_weight', 'signaller_share_weight' between 0 & 1

    'payoff_distinction' 0 - 100
    'honesty_bias' value between 1 & 100

    These two are used indirectly to set the value of baby_payoff & referral_cost, and
    midwife priors, respectively.
    """
    kwargs = []
    results = []

    with open(samples, 'rb') as csvfile:
        samples_reader = csv.DictReader(csvfile, delimiter=',')
        for row in samples_reader:
            row = dict((k, float(v)) for k, v in row.iteritems())
            bias = row['honesty_bias']
            type_weights = [[bias, 1., 1.], [1., bias, 1.], [1., 1., bias]]

            referral_cost = row['payoff_distinction']
            baby_payoff = 1 + row['payoff_distinction']

            kwarg = {'game_args': {'mw_share_prob':row['mw_share_prob'],
            'women_share_prob':row['women_share_prob'],
            'type_weights':type_weights, 'baby_payoff':baby_payoff, 'referral_cost':referral_cost}, 
            'signaller_args':{'share_weight':row['signaller_share_weight']},
            'responder_args':{'share_weight':row['responder_share_weight']},
            'mw_weights':[row['mw_0'], row['mw_1'], row['mw_2']], 
            'women_weights':[row['women_0'], row['women_1'], row['women_2']]}
            kwargs.append(kwarg)
            if chunksize is not None and len(kwargs) >= chunksize:
                results.append(kwargs)
                kwargs = []
    results.append(kwargs)
    return results
