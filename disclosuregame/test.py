import cPickle
import time
import os.path
with open(os.path.join(os.path.expanduser("~"), "game"), "rb") as fin:
    game, women, midwives = cPickle.load(fin)
    start = time.time()
    game.play_game((women, midwives))
    print "Ran in %f" % (time.time() - start)