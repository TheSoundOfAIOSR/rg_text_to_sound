from ner_tests.models import NERKeywordExtractor
import pandas as pd
from tqdm import tqdm
import logging

tqdm.pandas()

def load_ground_truth(csv_path, keyword_ids_path):
    df = pd.read_csv(csv_path, usecols=["sentence","tokens_id"])
    mapping_df = pd.read_csv(keyword_ids_path, usecols=["token","id","keyword"])
    logging.info("generating ground truth")
    
    #df["ground_truth"]=df["tokens_id"].map
