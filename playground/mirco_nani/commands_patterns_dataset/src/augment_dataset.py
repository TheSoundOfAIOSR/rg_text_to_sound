from commands_patterns_dataset import CommandsPatternsDataset
from tqdm import tqdm
import sys, argparse
import pandas as pd
import numpy as np

def trasform_results(pattern, ordered_values, tokens):
  r=[]
  for w in pattern.strip().split(" "):
    if w in tokens:
      r.append((ordered_values[0],w))
      ordered_values = ordered_values[1:]
    else:
      r.append((w,'NONE'))
  return r

def transform_df(df):
  return df.sort_values("start").groupby("sentence").apply(lambda x: pd.Series({
      "tokens":trasform_results(x["pattern"].values[0], x["keyword"].values, x["token"].values)
  })).reset_index()


def augment_ds(patterns, tokens_to_keywords, dest):
  header=True
  for i,p in enumerate(tqdm(patterns)):
    for q in tqdm(tokens_to_keywords["QLTY"], leave=False):
      for i in tqdm(tokens_to_keywords["INSTR"], leave=False):# using a loop to avoid filling the ram
        try:
          transform_df(
              CommandsPatternsDataset([p], {"INSTR":[i], "QLTY":[q]}).get_as_df()
              ).to_csv(dest, mode='w' if header else 'a', header=header, index=False)
          header=False
        except:
          pass


def augment(source, dest_train, dest_test, test_fraction):
  df_original=pd.read_csv(source)
  #df_simple=df_original[df_original["EDGE"]=="[]"].drop(columns=["EDGE"])
  df = df_original
  patterns = df["pattern"].unique().tolist()
  tokens_to_keywords = {col: list(set([x for s in df[col].unique() for x in eval(s)])) for col in df.columns if col != 'pattern'} 
  test_len = int(len(patterns)*0.1)
  test_patterns  = patterns[:test_len]
  train_patterns = patterns[test_len:]
  augment_ds(train_patterns, tokens_to_keywords, dest_train)
  augment_ds(test_patterns,  tokens_to_keywords, dest_test)


def parse_cli(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=str, required=True)
  parser.add_argument('--dest_train', type=str, required=True)
  parser.add_argument('--dest_test', type=str, required=True)
  parser.add_argument('--test_fraction', type=float, required=False, default=0.1)
  return parser.parse_known_args(argv)


if __name__ == "__main__":
  known_args, unknown_args = parse_cli(sys.argv[1:])
  augment(**vars(known_args))


#python augment_dataset.py \
#   --source="/mnt/e/dev/mirco/SoundOfAI/datasets/02/cureted_pattern_lists.csv" \
#   --dest_train="/mnt/e/dev/mirco/SoundOfAI/datasets/02/cureted_augmented_train.csv" \
#   --dest_test="/mnt/e/dev/mirco/SoundOfAI/datasets/02/cureted_augmented_test.csv"