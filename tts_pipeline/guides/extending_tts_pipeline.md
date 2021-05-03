If you are here by mistake, read the [README.md](../README.md) first.

# Extending TTS Pipeline
This guide explains how to use models that are not currently included in the tts_pipeline package an how to add them to it

## Using custom models in the pipelines
Every pipeline defines the interfaces of its internal models in its **`pipeline.py`** file.  
As an example, let's have a look at [tts_pipeline/pipelines/waterfall/pipeline.py](../src/tts_pipeline/pipelines/waterfall/pipeline.py)  
  
This module defines a pipeline called ``` WaterfallPipeline ``` which uses three different models:  
* ``` WaterfallKeywordExtractor ```
* ``` WaterfallEmbedder ```
* ``` WaterfallDimensionalityReducer ```  
  
  
In order to use a custom model, one of these three classes needs to be extended and its abstract methods need to be implemented with correct input/output signatures.   
  
Let's implement a custom WaterfallEmbedder that will use Google's "Universal Sentence Encoder" from tf.hub to embed text informations

```python
from tts_pipeline.pipelines.waterfall.pipeline import WaterfallEmbedder

# tensorflow dependencies
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


class MyWaterfallEmbedder(WaterfallEmbedder):
    def __init__(self, tf_hub_url): 
        # init arguments can be whatever this class needs
        self.tf_hub_url = tf_hub_url

    def build(self): 
        # mandatory method: this builds the internal model
        self.model = hub.load(self.tf_hub_url)

    def predict(self, sentence, keyword_extraction_results):
        # mandatory method: as long as input and output are as
        # defined in WaterfallEmbedder, this class will work when 
        # used by WaterfallPipeline
        sentences        = [sentence, keyword_extraction_results["instrument"]] 
        sentences       += keyword_extraction_results["soundquality"]
        sentences_tensor = tf.constant(sentences)
        output_tensor    = self.model(sentences_tensor).numpy()
        return {
            "sentence"    : output_tensor[0].tolist(),
            "instrument"  : output_tensor[1].tolist(),
            "soundquality": output_tensor[2:].tolist()
        }

    def dispose(self):
        del self.model
```
Now let's use our new model in the Pipeline
```python
from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.example import (
    DummyWaterfallKeywordExtractor,
    #BERTWaterfallEmbedder,
    DummyWaterfallDimensionalityReducer
)

pipeline = WaterfallPipeline(
    keyword_extractor = DummyWaterfallKeywordExtractor(),
    embedder = MyWaterfallEmbedder( # our new embedder
        tf_hub_url="https://tfhub.dev/google/universal-sentence-encoder/4"),
    dimensionality_reducer = DummyWaterfallDimensionalityReducer())

pipeline.build()                             # builds the pipeline
pred1 = pipeline.predict("a sentence")       # makes a prediction
pred2 = pipeline.predict("a bright and percussive acoustic guitar")
pipeline.dispose()                           # frees resources

print(pred2)
```
``` sample output: ```
```
{
    "source"        : "acoustic",
    "pitch"         : 60,
    "velocity"      : 75,
    "qualities"     : ['bright', 'percussive'],
    "latent_sample" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
}
```

## Adding new models to the TTS Pipelines package

Now that we developed our new model MyWaterfallEmbedder, we can put the this code in a python module under tts_pipeline/pipelines/waterfall/models.

### Setup  
``` 
git clone https://github.com/TheSoundOfAIOSR/rg_text_to_sound

cd playground/mirco_nani/tts_pipeline

pip install -r requirements/dev.txt
```

### Implementation
We need to add a new python module under src/tts_pipeline/pipelines/waterfall/models, let's name it my_example_models.py and put the code of ```MyWaterfallEmbedder``` in it.  
  
NOTE:  
* This file can contain any number of model classes as long as they extend the abstract definitions contained in tts_pipeline.pipelines.waterfall.pipeline, hence DummyWaterfallKeywordExtractor, BERTWaterfallEmbedder or DummyWaterfallDimensionalityReducer   
* A model that implements abstact definitions of another pipeline needs to be placed in the corresponding package: tts_pipeline.pipelines.\<PIPELINE_NAME\>.models   
(which corresponds to the folder: src/tts_pipeline/pipelines/\<PIPELINE_NAME\>/models)

At this point the model is part of the tts_pipeline package and can be used with normal imports  
```python
from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.example import (
    DummyWaterfallKeywordExtractor,
    DummyWaterfallDimensionalityReducer
)
from tts_pipeline.pipelines.waterfall.models.my_example_models import MyWaterfallEmbedder

pipeline = WaterfallPipeline(
    keyword_extractor = DummyWaterfallKeywordExtractor(),
    embedder = MyWaterfallEmbedder(
        tf_hub_url="https://tfhub.dev/google/universal-sentence-encoder/4"),
    dimensionality_reducer = DummyWaterfallDimensionalityReducer())

pipeline.build()                             
pred2 = pipeline.predict("a bright and percussive acoustic guitar")
pipeline.dispose()       
```

Now your new model is part of the TTS Pipeline module. Pushing the new my_example_models.py file to the TTS repository will make it available to all other research group members.

### BOUNS: Test the model
If you're interested in testing your model with pytest, you can read **[Testing TTS Pipeline Models](testing_tts_pipeline_models.md)**

