import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','..','..')
from tests.tts_pipeline.pipelines.waterfall.test_pipeline import (
    AbstractTestWaterfallKeywordExtractor,
    AbstractTestWaterfallEmbedder,
    AbstractTestWaterfallDimensionalityReducer
)
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','..','..','src') )
from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import UnifiedKeywordExtractor,UnifiedKeywordPairsExtractor


class TestUnifiedKeywordExtractor(AbstractTestWaterfallKeywordExtractor):
    @pytest.fixture(params=[
        UnifiedKeywordExtractor(),
        UnifiedKeywordPairsExtractor()
    ])
    def model(self, request):
        return request.param