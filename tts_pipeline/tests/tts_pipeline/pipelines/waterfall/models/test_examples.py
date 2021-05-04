import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../../..')
from tests.tts_pipeline.pipelines.waterfall.test_pipeline import (
    AbstractTestWaterfallKeywordExtractor,
    AbstractTestWaterfallEmbedder,
    AbstractTestWaterfallDimensionalityReducer
)
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../../../src') )
from tts_pipeline.pipelines.waterfall.models.examples import (
    DummyWaterfallKeywordExtractor,
    BERTWaterfallEmbedder,
    DummyWaterfallDimensionalityReducer
)

class TestDummyWaterfallKeywordExtractor(AbstractTestWaterfallKeywordExtractor):
    @pytest.fixture(params=[
        DummyWaterfallKeywordExtractor()
    ])
    def model(self, request):
        return request.param

        
class TestBERTWaterfallEmbedder(AbstractTestWaterfallEmbedder):
    @pytest.fixture(params=[
        BERTWaterfallEmbedder(),
        BERTWaterfallEmbedder(tf_hub_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1")
    ])
    def model(self, request):
        return request.param

    def test_output_field_sentence(self, simple_prediction):
        assert "sentence" in simple_prediction

    def test_output_field_instrument(self, simple_prediction):
        assert "instrument" in simple_prediction

    def test_output_field_soundquality(self, simple_prediction):
        assert "soundquality" in simple_prediction

        
class TestDummyWaterfallDimensionalityReducer(AbstractTestWaterfallDimensionalityReducer):
    @pytest.fixture(params=[
        DummyWaterfallDimensionalityReducer()
    ])
    def model(self, request):
        return request.param

        