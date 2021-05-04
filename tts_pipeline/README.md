# TTS Pipeline
tts_pipeline contains the definition and implementation of **inference models** and **inference pipelines**.

## What is an inference model?
An inference model is simply an object with three methods
```python
inference_model = MyInferenceModel(**args)
inference_model.build()                      # builds the model and all necessary resources
prediction = inference_model.predict(**args) # makes a prediction. I/O changes depending on the model
inference_model.dispose()                    # frees memory and resources
```

## What is an inference pipeline?
An inference pipeline is an object with the same three methods as the inference model.
An inference pipeline uses multiple models to make a prediction.
```python
inference_pipeline = MyInferencePipeline(model1=MyFirstModel(**args1), model2=MySecondModel(**args2))
inference_pipeline.build()                            # builds the pipeline
prediction = inference_pipeline.predict("a sentence") # uses the models to make a prediction. 
                                                      # Input is always a sentence and output 
                                                      # is always a dictionary with fixed fields
inference_pipeline.dispose()                          # frees memory and resources
```

## Where are the available inference pipelines and models?
### Module structure

    tts_pipeline  
     │
     ├── pipelines                <- contains all the available pipelines    
     │   └── waterfall            <- This is a pipeline module, more could be added in the future    
     │   │   ├── pipeline.py      <- Contains the pipeline implementation and models definitions
     │   │   └── models           <- Contains implementations of the models that can be used by this pipeline    
     │   │       ├── examples.py  <- Some example implementations    
     │   │       └── ...          <- More implementations can be added in the future    
     │   │    
     │   └── another_pipeline     <- this is not actually in the module, it is here only for demonstration purposes    
     │       ├── pipeline.py      
     │       └── models           <- every pipeline module always contains a "models" folder and a "pipeline" folder    
     └── core.py                  <- Definition of abstract classes InferenceModel and InferencePipeline    



## how do I use them?

### setup
``` 
pip install git+https://git@github.com/TheSoundOfAIOSR/rg_text_to_sound.git#"subdirectory=playground/mirco_nani/tts_pipeline" 
```

### Usage example: **WaterfallPipeline**
```python
from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.examples import (
    DummyWaterfallKeywordExtractor,
    BERTWaterfallEmbedder,
    DummyWaterfallDimensionalityReducer
)


pipeline = WaterfallPipeline(
    keyword_extractor = DummyWaterfallKeywordExtractor(),
    embedder = BERTWaterfallEmbedder(),
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

You can use other pre-made models in the WaterfallPipeline as long as they are located in the module:  
 ```tts_pipeline.pipelines.waterfall.models```  
You can use other implementations of the WaterfallPipeline as long as they are located in the following module:   
 ```tts_pipeline.pipelines.waterfall.pipeline```


## Can I use my own models?
Yes, have a look at **[Extending TTS Pipeline](guides/extending_tts_pipeline.md)**

## I's still not very clear, can I have more resources?
### Video tutorials and more resources
* Registration of hands-on demo, plus custom model implementation demo: https://www.youtube.com/watch?v=qNQhxMjrxuI  
* Registration of custom model implementation demo, but with more readable text: https://www.youtube.com/watch?v=WS5StYHLAPY   
* Python notebook used during the demo: [tts_pipeline_hands_on.ipynb](notebooks/tts_pipeline_hands_on.ipynb)
* Files added during the demos:
    * [gnews_models.py](src/tts_pipeline/pipelines/waterfall/models/gnews_models.py)