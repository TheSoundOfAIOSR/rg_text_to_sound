# Benchmarking tools
Simple tools to benchmark embedding models preformances

## Setup
* Clone this repository
* Run ```pip install -r rg_text_to_sound/playground/mirco_nani/benchmarking_tools/requirements.txt```

## How to benchmark your model in 2 steps
### Step 1: extend the PredictionModel
In order to benchmark your model you will need to define a wrapper class which extends ```benchmarking_tools.model.prediction_model.PredictionModel```.  
This class must redefine three simple methods:  
* **build**: builds the model
* **predict**: given the parameter *sentences*, which is a list of strings, uses the model to return a batch of embeddings as a numpy array
* **additional_infos**: (optional) returns a dictionary containing additional informations about the wrapped model such as the framework used to build it.
  
The class definition can be found [here](src/benchmarking_tools/model/prediction_model.py) and a sample usage can be found [here](src/benchmarking_tools/model/tf_hub_models.py)

### Step 2: use the benchmark function
The benchmark function can be found in ```benchmarking_tools.benchmarking.benchmark```  
Given a list of PredictionModel and a list of sentences, it returns a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) containing the performances of the model.  
To use the function  
* import it: ```from benchmarking_tools.benchmarking import benchmark```  
* Instantiate your model: ```mymodel = MyPredictionModel()```  
* Invoke the function: ```df=benchmark([mymodel],["a sentence","another sentence"])```  
The function definition can be found [here](src/benchmarking_tools/benchmarking.py), a sample usage can be found [here](src/benchmark_tfhub.py)
### Bonus: how to compare models
* Invoke the ```benchmark``` function with a list of multiple models instead of a list of a single model, the function will return a dataframe with a row for each model
* Use the ```plotting_tools.plot_benchmark.barh_on_benchmark_results``` function to obtain a comparison plot on a specific performance metric.  

The plotting function definition can be found [here](src/benchmarking_tools/plotting_tools/plot_benchmark.py), a sample usage can be found [here](src/plot_benchmark_results.py)

## Currently implemented metrics
* **build_seconds**: How much time it takes to build the model
* **first_prediction_seconds**: How much time it takes to make the very first prediction after the build phase
* **second_prediction_seconds**: How much time it takes to make anothe prediction after the first (to account for possible lazy loading mechanics)
* **embedding_size**: The number of dimensions of the output embedding

## Complete sample usage
A complete sample usage can be found [here](notebooks/benchmark_sample_usage.ipynb)  
It can be run on colab [here](https://colab.research.google.com/github/TheSoundOfAIOSR/rg_text_to_sound/blob/main/playground/mirco_nani/benchmarking_tools/notebooks/benchmark_sample_usage.ipynb) or [here](https://colab.research.google.com/github/Mirco-Nani/rg_text_to_sound/blob/main/playground/mirco_nani/benchmarking_tools/notebooks/benchmark_sample_usage.ipynb)