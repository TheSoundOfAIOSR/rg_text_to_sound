from ner_tests.orchestrator import *
import sys


if __name__ == "__main__":
    args = sys.argv[1:]
    if not '--local-scheduler' in args:
        args.append('--local-scheduler')
    luigi.run(args)