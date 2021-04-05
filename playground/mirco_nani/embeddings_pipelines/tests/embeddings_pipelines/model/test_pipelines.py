import pytest
import numpy as np
import os, sys
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../src') )
from embeddings_pipelines.model.pipelines import EmbeddingPipeline

class ExampleEmbeddingPipeline2(EmbeddingPipeline):
    def __init__(self):
        pass

    def build(self):
        pass

    def embed(self, a=1):
        return np.array([1,2,3])

    def dispose(self):
        pass


class AbstractTestEmbeddingPipeline: #since this doesn't start with "Test" it won't get executed
    predict_input = "a simple sentence"

    @pytest.fixture
    def built_pipeline(self, pipeline):
        pipeline.build()
        yield pipeline
        pipeline.dispose()


    @pytest.fixture
    def simple_prediction(self, built_pipeline):
        return built_pipeline.embed(self.predict_input)


    def test_embedding_is_1d(self, simple_prediction):
        assert len(simple_prediction.shape) == 1


class TestEmbeddingPipelineSubclass(AbstractTestEmbeddingPipeline): #this WILL get executed, and since it inherits AbstractTestEmbeddingPipeline it will execute its fixtures and tests too!
    @pytest.fixture(params=[ExampleEmbeddingPipeline2()])
    def pipeline(self, request):
        return request.param


    def test_first_element_is_1(self, built_pipeline):
        result = built_pipeline.embed()
        assert result[0] == 1
