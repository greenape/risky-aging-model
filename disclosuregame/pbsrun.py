from disclosuregame.Util import make_pbs_script
import disclosuregame

from subprocess import check_output, CalledProcessError
import tempfile
import os
import argparse
from os.path import expanduser
import logging
import multiprocessing

logger = multiprocessing.log_to_stderr()
version = disclosuregame.__version__

def arguments():
    parser = argparse.ArgumentParser(
        description='Run some variations of the disclosure game with all combinations of games, signallers and responders provided.')
    parser.add_argument('-g', '--games', type=str, nargs='*',
                   help='A game type to play.', default=['Game', 'CaseloadGame'],
                   choices=['Game', 'CaseloadGame', 'RecognitionGame', 'ReferralGame',
                   'CaseloadRecognitionGame', 'CaseloadReferralGame', 'CarryingGame',
                   'CarryingReferralGame', 'CarryingCaseloadReferralGame', 'CaseloadSharingGame',
                   'CarryingInformationGame'],
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
    parser.add_argument('--pickled-arguments', dest='kwargs', type=str, nargs=1,
        help="A fuzzy match for kwargs files.")
    #parser.add_argument('--individual-measures', dest='indiv', action="store_true",
    #    help="Take individual outcome measures instead of group level.", default=False)
    parser.add_argument('--abstract-measures', dest='abstract', action="store_true",
        help="Take measures intended for the extended abstract.", default=False)
    parser.add_argument('--log-level', dest='log_level', type=str, choices=['debug',
        'info', 'warniing', 'error'], default='info', nargs="?")
    parser.add_argument('--log-file', dest='log_file', type=str, default='')
    parser.add_argument('--tag', dest='tag', type=str, default='')

    parser.add_argument('--hours', dest='hours', type=int, default=60, nargs='?')
    parser.add_argument('--mins', dest='mins', type=int, default=60, nargs='?')
    parser.add_argument('--ppn', dest='ppn', type=int, default=16, nargs='?')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    logger.setLevel(numeric_level)
    if args.log_file != "":
        fh = logging.FileHandler(log_file)
        fh.setLevel(numeric_level)
        formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return args


def main():
    args = arguments()
    logger.info("Version %s" % version)
    fd, filename = tempfile.mkstemp()
    pbs_job, pbs_count =  make_pbs_script(args, hours=args.hours, mins=args.mins, ppn=args.ppn, script_name=filename)
    logger.info("Found %d pickled argument." % pbs_count)
    logger.info("Generated PBS script:\n%s" % pbs_job)

    logger.info("Writing PBS script to temporary file %s" % filename)
    try:
        tfile = os.fdopen(fd, "w")
        tfile.write(pbs_job)
        tfile.close()
        if args.test_only:
            logger.info("This is a test of the emergency broadcast system. This is only a test.")
        else:
            logger.info("Submitting to PBS.")
            response = check_output(['qsub', '-t', '0-%d' % pbs_count, filename])
            logger.info("PBS response was: %s" % response)
    except (CalledProcessError, OSError) as e:
        logger.info("qsub failed with message: %s" % e.strerror)
        logger.info("Removing temp file.")
        os.remove(filename)
    except Exception as e:
        logger.info("Failed to write PBS script.")
        logger.info("Message was: %s" % e.strerror)
       

if __name__ == "__main__":
    main()