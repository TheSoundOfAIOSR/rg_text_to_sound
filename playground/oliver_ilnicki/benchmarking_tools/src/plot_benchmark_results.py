from plotting_tools.plot_benchmark import barh_on_benchmark_results
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
import sys
import os

def run(input_csv, color_column, output_folder):
  df = pd.read_csv(input_csv)
  for col in tqdm(["build_seconds", "first_prediction_seconds", "second_prediction_seconds","embedding_size"]):
    fig, ax = barh_on_benchmark_results(df, col, color_col=color_column)
    plt.savefig(os.path.join(output_folder,f"{col}.png"), bbox_inches="tight")
    plt.close()
    fig, ax = barh_on_benchmark_results(df, col, color_col=color_column, logx=True)
    plt.savefig(os.path.join(output_folder,f"{col}-logx.png"), bbox_inches="tight")
    plt.close()

  
def parse_cli(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--input_csv', type=str, required=True, help="the csv of a df prouced by a benchmark")
  parser.add_argument('--color_column', type=str, required=False, default=None, help="the color column of the plots")
  parser.add_argument('--output_folder', type=str, required=True, help="folder destination of the plots")
  
  return parser.parse_known_args(argv)
  
if __name__ == "__main__":
  known_args, unknown_args = parse_cli(sys.argv[1:])
  run(**vars(known_args))