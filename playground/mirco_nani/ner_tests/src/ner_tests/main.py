from ner_tests.orchestrator import *
from pathlib import Path
import time
import sys, argparse


def main(
    model_path: Path,
    testset_csv: Path,
    result_json: Path,
    workers: int = 1
):
    luigi.run([
        'Test', 
        '--workers', str(workers), 
        '--local-scheduler',
        '--csv-file', testset_csv,
        '--model-path', model_path,
        '--output-file', result_json
    ])


def parse_cli(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--model_path', type=str, required=True)
  parser.add_argument('--testset_csv', type=str, required=True)
  parser.add_argument('--result_json', type=str, required=True)
  parser.add_argument('--workers', type=int, default=1)

  return parser.parse_known_args(argv)


if __name__ == '__main__':
    known_args, unknown_args = parse_cli(sys.argv[1:])
    now=time.time()
    main(**vars(known_args))
    print("run took "+str(time.time()-now)+" seconds")