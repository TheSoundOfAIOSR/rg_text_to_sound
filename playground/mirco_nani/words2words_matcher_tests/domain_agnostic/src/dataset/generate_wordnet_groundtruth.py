from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import typer
import nltk
import json
import os

def get_synonyms(target_word):
    syns = wordnet.synsets(target_word)
    mys = set()
    for syn in syns:
        #print(syn.name())
        mys |=set([m.name() for m in syn.lemmas()])
    return mys

def tolist(myset,length):
    l = list(myset)
    if len(l)<length:
        l+=['' for i in range(length-len(myset))]
        return l
    else:
        return l[:length]


assets_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets')
default_output_file = os.path.join(assets_folder, "wordnet20.json")

def main(output_file = default_output_file):
    k=20 #maximum number of synonyms we might need
    nltk.download('wordnet')
    target_word_pairs = [('bright', 'dark'), ('full', 'hollow'),( 'smooth', 'rough'), ('warm', 'metallic'), ('clear', 'muddy'), ('thin', 'thick'), ('pure', 'noisy'), ('rich', 'sparse'), ('soft', 'hard')]
    target_words = np.array(target_word_pairs).ravel().tolist()
    df = pd.DataFrame([pd.Series(tolist(get_synonyms(targetword),k) ,name = targetword) for targetword in target_words]).T
    df_sorted = df.where(~df.isin(df.columns),'').apply(lambda x:sorted(x,reverse=True))
    word_to_words = df_sorted.to_dict(orient="list")
    word_to_words = {k:[w for w in v if len(w) > 0] for k,v in word_to_words.items()} # remove empty strings
    json.dump(word_to_words, open(output_file, "w"))


if __name__ == "__main__":
    typer.run(main)