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
from tts_pipeline.pipelines.waterfall.models.ner_model import NERKeywordExtractor


class TestNERKeywordExtractor(AbstractTestWaterfallKeywordExtractor):
    @pytest.fixture(params=[
        NERKeywordExtractor()
    ])
    def model(self, request):
        return request.param