from disclosuregame.run import *
from disclosuregame.Util import make_pbs_script
from subprocess import call
import tempfile
import os

logger = multiprocessing.log_to_stderr()

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

    return args


def main():
    args = arguments()
    logger.info("Version %s" % version)
    fd, filename = tempfile.mkstemp()
    pbs_job, pbs_count =  make_pbs_script(args, hours=args.hours, mins=args.mins, ppn=args.ppn, script_name=filename)
    logger.info("Found %d pickled argument." % pbs_count)
    logger.info("Generated PBS script:\n%s" % pbs_job)

    print("Writing PBS script to temporary file %s" % filename)
    try:
        tfile = os.fdopen(fd, "w")
        tfile.write(pbs_job)
        tfile.close()
        if args.test_only:
            logger.info("This is a test of the emergency broadcast system. This is only a test.")
        else:
            logger.info("Submitting to PBS.")
            call(['qsub', '-t', '0-%d' % pbs_count, filename])
    except:
        logger.info("Failed to write PBS script.")
       

if __name__ == "__main__":
    main()