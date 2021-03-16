import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.

import time
import multiprocessing
from tqdm import tqdm

import json
import pandas as pd


def benchmark_prediction_model(model, sentences, results=None):
  if results is None:
    results={}
  
  model_name=model.additional_infos().get("name",type(model).__name__)
  results["model_name"]=model_name
  
  print(f"{model_name} - building...")
  now=time.time()
  model.build()
  results["build_seconds"]=time.time()-now
  
  print(f"{model_name} - first prediction...")
  now=time.time()
  prediction = model.predict(sentences)
  results["first_prediction_seconds"]=time.time()-now
  
  print(f"{model_name} - second prediction...")
  now=time.time()
  prediction = model.predict(sentences)
  results["second_prediction_seconds"]=time.time()-now

  results["embedding_size"]=prediction.shape[1]
  results.update(model.additional_infos())

  return results


def safe_benchmark_prediction_model(model, sentences, results=None):
  if results is None:
    results={}

  try:
    benchmark_prediction_model(model, sentences, results)
    results["success"]=True
  except:
    results["success"]=False

  return results


class BenchmarkingTools():
  def __init__(self):
    self.manager = multiprocessing.Manager()

  def benchmark_and_cleanup(self, model, sentences):
    """
      tests model performances in a separate process
      when the process dies, python should purge 
      occupied resources such as RAM and GPU memory
      Source:
      https://github.com/tensorflow/tensorflow/issues/36465
    """
    return_dict = self.manager.dict() # source: https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce
    process_eval = multiprocessing.Process(
        target=benchmark_prediction_model, args=(model, sentences, return_dict))
    process_eval.start()
    process_eval.join()
    return dict(return_dict)



def benchmark(models, sentences):
    """
    tests models performances on the given sentences
    :param: models
        a list of PredictionModel instances
    
    :param: sentences
        a list of strings, every string is a sentence

    :returns:
        a pandas DataFrame with models on rows and performance metrics and additional models informations on columns

    """
    tools=BenchmarkingTools()
    results = []
    for model in tqdm(models):
        r=tools.benchmark_and_cleanup(model, sentences)
        results.append(r)
    df=pd.DataFrame(results)
    return df