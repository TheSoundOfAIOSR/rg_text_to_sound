from commands_patterns_dataset import CommandsPatternsDataset
from tqdm import tqdm
import sys, argparse, os
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

def get_token_ids(pattern, ordered_values, tokens, tokens_to_keywords_ids):
  r=[]
  for w in pattern.strip().split(" "):
    if w in tokens:
      r.append((tokens_to_keywords_ids[w][ordered_values[0]],w))
      ordered_values = ordered_values[1:]
  return r


def transform_df(df, pattern_ids, tokens_to_keywords_ids):
  return df.sort_values("start").groupby("sentence").apply(lambda x: pd.Series({
      "tokens":trasform_results(x["pattern"].values[0], x["keyword"].values, x["token"].values),
      "pattern_id": pattern_ids[x["pattern"].values[0]],
      "tokens_id":get_token_ids(x["pattern"].values[0], x["keyword"].values, x["token"].values, tokens_to_keywords_ids)
  })).reset_index()


def augment_ds(patterns, tokens_to_keywords, dest, pattern_ids, tokens_to_keywords_ids):
  header=True
  for i,p in enumerate(tqdm(patterns)):
    for q in tqdm(tokens_to_keywords["QLTY"], leave=False):
      for i in tqdm(tokens_to_keywords["INSTR"], leave=False):# using a loop to avoid filling the ram
        try:
          transform_df(
              CommandsPatternsDataset([p], {"INSTR":[i], "QLTY":[q]}).get_as_df(),
              pattern_ids, tokens_to_keywords_ids
              ).to_csv(dest, mode='w' if header else 'a', header=header, index=False)
          header=False
        except:
          pass


def augment(source_file, dest_train, dest_test_context, dest_test_content, dest_test_unseen, test_context_fraction, test_content_fraction, dest_pattern_ids, dest_keyword_ids):
  df_original=pd.read_csv(source_file)
  #df_simple=df_original[df_original["EDGE"]=="[]"].drop(columns=["EDGE"])
  df = df_original
  patterns = df["pattern"].unique().tolist()
  tokens_to_keywords = {col: list(set([x for s in df[col].unique() for x in eval(s)])) for col in df.columns if col != 'pattern'} 

  pattern_ids={p:i for i,p in enumerate(patterns)}
  tokens_to_keywords_ids={token:{keyword:i for i,keyword in enumerate(tokens_to_keywords[token])} for token in tokens_to_keywords}

  # separating train/test context
  test_len = int(len(patterns)*test_context_fraction)
  patterns_test  = patterns[:test_len]
  patterns_train = patterns[test_len:]

  # separating train/test content
  tokens_to_keywords_test = {k:v[:int(len(v)*test_content_fraction)] for k,v in tokens_to_keywords.items()}
  tokens_to_keywords_train = {k:v[int(len(v)*test_content_fraction):] for k,v in tokens_to_keywords.items()}

  # saving tokens ids and keywords ids to file
  pd.DataFrame(pattern_ids.items(), columns=["pattern","id"])[["id","pattern"]].to_csv(dest_pattern_ids, index=False)
  pd.DataFrame(
      [[token,id,keyword] for token in tokens_to_keywords_ids for keyword,id in tokens_to_keywords_ids[token].items()],
      columns=["token","id","keyword"]
  ).to_csv(dest_keyword_ids, index=False)

  augment_ds(patterns_train,  tokens_to_keywords_train, dest_train, pattern_ids, tokens_to_keywords_ids)
  augment_ds(patterns_test,   tokens_to_keywords_train, dest_test_context, pattern_ids, tokens_to_keywords_ids)
  augment_ds(patterns_train,  tokens_to_keywords_test,  dest_test_content, pattern_ids, tokens_to_keywords_ids)
  augment_ds(patterns_test,   tokens_to_keywords_test,  dest_test_unseen, pattern_ids, tokens_to_keywords_ids)


def run(source_file, dest_folder, test_context_fraction, test_content_fraction):
  augment(
    source_file           = source_file, 
    dest_train            = os.path.join(dest_folder, "train.csv"), 
    dest_test_context     = os.path.join(dest_folder, "test_context.csv"), 
    dest_test_content     = os.path.join(dest_folder, "test_content.csv"), 
    dest_test_unseen      = os.path.join(dest_folder, "test_unseen.csv"), 
    test_context_fraction = test_context_fraction, 
    test_content_fraction = test_content_fraction,
    dest_pattern_ids      = os.path.join(dest_folder, "pattern_ids.csv"),
    dest_keyword_ids      = os.path.join(dest_folder, "keyword_ids.csv")
  )


def parse_cli(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_file', type=str, required=True)
  parser.add_argument('--dest_folder', type=str, required=True)
  parser.add_argument('--test_context_fraction', type=float, required=False, default=0.1)
  parser.add_argument('--test_content_fraction', type=float, required=False, default=0.1)
  return parser.parse_known_args(argv)


if __name__ == "__main__":
  known_args, unknown_args = parse_cli(sys.argv[1:])
  run(**vars(known_args))


#python augment_dataset.py \
#   --source="/mnt/e/dev/mirco/SoundOfAI/datasets/03/original/cureted_pattern_lists.csv" \
#   --dest_folder="/mnt/e/dev/mirco/SoundOfAI/datasets/03/augmented/" \
#   --test_context_fraction=0.1 \
#   --test_content_fraction=0.1