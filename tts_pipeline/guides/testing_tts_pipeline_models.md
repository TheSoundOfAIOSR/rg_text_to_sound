If you are here by mistake, read **[Extending TTS Pipeline](extending_tts_pipeline.md)** first.

# Testing TTS Pipeline Models
NOTE:
* The tts_pipeline project uses pytest as test suite tool, so it is adviced to read its [documentation](https://docs.pytest.org/), expecially beacuse of [fixtures](https://docs.pytest.org/en/stable/fixture.html) which are heavily used   

## Testing your model
To test your model implementation, add the following file:  
```tests/tts_pipeline/pipelines/waterfall/models/test_my_example_models.py```  
Notice how this file has the same path of the module my_example_models under ```src``` and its name has the string ```test_``` prepended to the module's name.   
Then, create your test classes in ```test_my_example_models.py``` as follows:
```python
import pytest
import numpy as np
import os, sys

# import from inside test/
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../../..')
from tests.tts_pipeline.pipelines.waterfall.test_pipeline import AbstractTestWaterfallEmbedder

# import from inside src/
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../../../src') )
from tts_pipeline.pipelines.waterfall.models.my_example_models import MyWaterfallEmbedder

class TestMyWaterfallEmbedder(AbstractTestWaterfallEmbedder):
    @pytest.fixture(params=[
        MyWaterfallEmbedder(tf_hub_url="https://tfhub.dev/google/universal-sentence-encoder/4")
    ])
    def model(self, request):
        return request.param
```

Then, run ```pytest -v tests``` to run all the tests, including your brand new test class.  
This could take a while, so if you want to just test your new class, run   
```pytest -v tests/tts_pipeline/pipelines/waterfall/models/test_my_example_models.py```   
  
NOTE:  
* If ```my_example_models.py``` contains multiple models, test them in ```test_my_example_models.py```   
* notice the naming conventions ```TestMyWaterfallEmbedder```, ```AbstractTestWaterfallEmbedder``` and the file's name ```test_my_example_models.py```.
* ```MyWaterfallEmbedder``` will be tested against all the tests defined in ```AbstractTestWaterfallEmbedder```, if you're curious about what they are, have a look at [waterfall/test_pipeline.py](../tests/tts_pipeline/pipelines/waterfall/test_pipeline.py), which also contains tests for the other two models used by WaterfallPipeline: ```AbstractTestWaterfallKeywordExtractor``` and ```AbstractTestWaterfallDimensionalityReducer```  

## Adding custom tests
You can add custom tests by defining new methods in your test class, their names need to begin with ```test_```.  
Here is an example:

```python
class TestMyWaterfallEmbedder(AbstractTestWaterfallEmbedder):
    @pytest.fixture(params=[
        MyWaterfallEmbedder(tf_hub_url="https://tfhub.dev/google/universal-sentence-encoder/4")
    ])
    def model(self, request):
        return request.param

    def test_output_is_returned(self, simple_prediction): #Custom test!
        assert simple_prediction is not None
```

### **how does this work?**
You may notice that ```simple_prediction``` is not defined anywhere, but if this test class is run, ```test_output_is_returned``` will be evaluated with a correctly valued ```simple_prediction```.  
This is because ```simple_prediction``` comes from a fixture that is defined in the superclass ```AbstractTestWaterfallEmbedder```, so it is built and made available ready to be used.  
```model``` on the other hand, is a fixture defined on ```TestMyWaterfallEmbedder``` that is used by the ```AbstractTestWaterfallEmbedder``` superclass in order to execute tests that are common to every ```WaterfallEmbedder```.
As ```simple_prediction```, there are other fixture provided by the superclass that are ready to use, here is the full list:
* ```built_model```: the model coming from the fixture ```model```, but with its ```build()``` methid already invoked. Pytest will make sure tu invoke its ```dispose()``` method when all the tests on that model are done.
* ```simple_prediction```: a prediction coming from the model. In this case the ```predict()``` method is invoked with the parameter ```**self.predict_input```. If you want to invoke your model with a custo input, you can
    * override the default ```predict_input```
    * direcly use ```built_model``` to make your own predictions with your own inputs  
* ```same_simple_prediction```: This comes from another infocation of the model with the same input as ```simple_prediction```, it is used to test model invariance