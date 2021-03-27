import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../..')
from tests.tts_pipeline.test_core import AbstractTestInferencePipeline, AbstractTestInferenceModel
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../../src') )
from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.examples import (
    DummyWaterfallKeywordExtractor,
    BERTWaterfallEmbedder,
    DummyWaterfallDimensionalityReducer
)
from tts_pipeline.pipelines.waterfall.models.gnews_models import GNewsWaterfallEmbedder

PIPELINES_TO_TEST = [
    WaterfallPipeline(
        DummyWaterfallKeywordExtractor(),
        BERTWaterfallEmbedder(tf_hub_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1"),
        DummyWaterfallDimensionalityReducer()
    ),
    WaterfallPipeline(
        DummyWaterfallKeywordExtractor(),
        GNewsWaterfallEmbedder(),
        DummyWaterfallDimensionalityReducer()
    )
]


class TestWaterfallPipeline(AbstractTestInferencePipeline):
    @pytest.fixture(params=PIPELINES_TO_TEST)
    def pipeline(self, request):
        return request.param


class AbstractTestWaterfallKeywordExtractor(AbstractTestInferenceModel):
    predict_input = dict(sentence="a simple sentence")

class AbstractTestWaterfallEmbedder(AbstractTestInferenceModel):
    predict_input = dict(
        sentence="a simple sentence",
        keyword_extraction_results={
            "soundquality": ['bright', 'percussive'],
            "instrument": "acoustic",
            "pitch": 60,
            "velocity": 75
        }
    )

class AbstractTestWaterfallDimensionalityReducer(AbstractTestInferenceModel):
    predict_input = dict(
        sentence="a simple sentence",
        keyword_extraction_results={
            "soundquality": ['bright', 'percussive'],
            "instrument": "acoustic",
            "pitch": 60,
            "velocity": 75
        },
        embedding_results={
            "sentence" : np.zeros((128,)),
            "soundquality": [np.zeros((128,))]*2,
            "instrument" : np.zeros((128,))
        }
    )
