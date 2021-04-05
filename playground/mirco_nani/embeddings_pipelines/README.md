# Embeddings Pipelines
This repository defines the skeleton of the pipelines used at inference time.  
All the pipelines take a sentence as input and output an embedding.

## Logical entities
Two simple logical entities are defined
* **EmbeddingPipeline**: an object that is able to take a sentence as input and to output an embedding.  
every pipeline is made of a series of steps.  
every step is a PredictionModel  
the pipeline uses PredictionModels connecting their I/O in order to get the final embedding
* **PredictionModel**: a class that contains a model which performs the internal operations of one step of a pipeline

## Setup
**Using pip**  
``` 
pip install git+https://git@github.com/TheSoundOfAIOSR/rg_text_to_sound.git#"subdirectory=playground/mirco_nani/embeddings_pipelines" 
```

**Using git**  
``` 
git clone https://github.com/TheSoundOfAIOSR/rg_text_to_sound
cd playground/mirco_nani/embeddings_pipelines

# to develop the package
pip install -r requirements/dev.txt

# to use the package outside its development environment
pip install .
```


## How to use
First, choose which pipeline you want to use from the [available pipelines](src/embeddings_pipelines/pipelines) and import it
```python
from embeddings_pipelines.pipelines.three_stages_pipelines import MergeAtDimensionalityReductionStepPipeline
```

Check the signature and the docstring to see which are the compatible models for the steps of [this pipeline](src/embeddings_pipelines/pipelines/three_stages_pipeline.py)
```python
...
class MergeAtDimensionalityReductionStepPipeline(EmbeddingPipeline):
    """This pipeline contains the following steps:
    
     - keyword extraction: text sentence to sequence of N words
     - word embedding: sequence of N words to batch of N highly-dimensional embeddings
     - dimensionality reduction: batch of N highly-dimensional embeddings to single low-dimensional embedding

    """
    def __init__(
        self, 
        keyword_extraction_model: KeywordExtractionModel,
        word_embedding_model: WordEmbeddingModel,
        dimensionality_reduction_model: MultipleEmbeddingsDimensionalityReductionModel
    ):
...
```

Select and instantiate the pipeline steps from [models](src/embeddings_pipelines/models/) that extend the step's classes indicated by the signature and use them to instantiate the pipeline
```python
from embeddings_pipelines.models.keyword_extraction_model import DummyKeywordExtractionModel
from embeddings_pipelines.models.word_embedding_models import DummyWordEmbeddingModel
from embeddings_pipelines.models.multiple_embeddings_dimensionality_reduction_model import DummyMultipleEmbeddingsDimensionalityReductionModel

step1 = DummyKeywordExtractionModel(separator = " ")
step2 = DummyWordEmbeddingModel(embedding_size = 256)
step3 = DummyMultipleEmbeddingsDimensionalityReductionModel(reduced_embedding_size=16)
    
pipeline = MergeAtDimensionalityReductionStepPipeline(
    keyword_extraction_model = step1,
    word_embedding_model = step2,
    dimensionality_reduction_model = step3
)
```
Build the pipeline and use it on input sentences
```python
pipeline.build()
sentence1 = "this is a sentence"
sentence2 = "this is another sentence, longer than the first one"
emb1 = pipeline.embed(sentence1)
emb2 = pipeline.embed(sentence2)
print(f""" "{sentence1}" -> {emb1} """)
print(f""" "{sentence2}" -> {emb2} """)
```

## code structure
* embeddings_pipelines.model: contains the abstract classes definition
    + embeddings_pipelines.model.pipelines: contains abstract classes that define the pipelines
    + embeddings_pipelines.model.models: contains abstract classes that define the pipelines steps, AKA the models
* embeddings_pipelines.pipelines: contains the concrete implementations of the pipelines
* embeddings_pipelines.models: contains the concrete implementations of the models, every module contains models that implement a specific abstract class definition

## how to extend
### defining new pipelines
In order to define a new pipeline, create a class that extends [embeddings_pipelines.model.pipelines.EmbeddingPipeline](src/embeddings_pipelines/model/pipelines.py) and implements its abstract methods complying with their signatures.  
Implementation examples can be found [here](src/embeddings_pipelines/pipelines/three_stages_pipelines.py).  
If you are adding this pipeline class inside the embeddings_pipelines package, put it in **embeddings_pipelines.pipelines**.  
  
If the pipeline needs models that are not already defined, new model definitions can be added by extending the [embeddings_pipelines.model.models.PredictionModel](src/embeddings_pipelines/model/models.py) class.  
If you are adding these new model definitions inside the embeddings_pipelines package, put them in the **embeddings_pipelines.model.models** module  
  
### defining new models  
In order to define a new model, create a class that extends a model definition found in the **embeddings_pipelines.model.models** module and implement its abstract methods while complying with their signatures.  
Here are some implementation examples:
* [DummyKeywordExtractionModel](src/embeddings_pipelines/models/keyword_extraction_models.py)
* [DummyWordEmbeddingModel](src/embeddings_pipelines/models/word_embedding_models.py)
* [DummyMultipleWordsEmbeddingModel](src/embeddings_pipelines/models/multiple_words_embedding_models.py)
* [DummyEmbeddingDimensionalityReductionModel](src/embeddings_pipelines/models/multiple_embeddings_dimensionality_reduction_model.py)
* [DummyMultipleEmbeddingsDimensionalityReductionModel](src/embeddings_pipelines/models/multiple_embeddings_dimensionality_reduction_model.py)

If you are adding these model classes inside the embeddings_pipelines package, put them in **embeddings_pipelines.models** in a module (.py file) that has the same name of the implemented abstract class definition (create the module if it does not exist).  

## usage example
A very simple usage example can be found [here](src/embeddings_pipelines/embeddings_pipelines_sample_usage.py). 

## UML of the classes
NOTE: 
* you can edit the UML [here](https://lucid.app/lucidchart/invitations/accept/10b794c9-037d-4437-ac1a-96103aaf9037). Once edited, save it on the online tool, export it as PNG and replace the [assets/uml.png](assets/uml.png) file
* yellow classes are not yet implemented, they are represented for demonstration purposes  
  
![UML](assets/uml.png)

## Tests
Tests are based on [pytest](https://docs.pytest.org/) and are located on the [tests](tests) folder.  
To execute the tests, launch ```pytest tests```.    
When adding new tests please follow the same folder structure as src.  
Test classes follow the same inheritance relations shown in the UML.  
When defining new test classes please try to respect the same inheritance structure as the classes tested in them.  
For an example please refer to the contents of  
* [tests/embeddings_pipelines/model/test_models.py](tests/embeddings_pipelines/model/test_models.py)  
* [tests/embeddings_pipelines/models/test_keyword_extraction_models.py](tests/embeddings_pipelines/models/test_keyword_extraction_models.py)  