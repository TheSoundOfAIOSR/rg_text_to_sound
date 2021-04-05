import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../..')
from tests.embeddings_pipelines.model.test_models import AbstractTestKeywordExtractionModel
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../src') )
from embeddings_pipelines.models.keyword_extraction_models import (
    DummyKeywordExtractionModel
)


class TestDummyKeywordExtractionModel(AbstractTestKeywordExtractionModel):
    @pytest.fixture(params=[DummyKeywordExtractionModel(), 
                            DummyKeywordExtractionModel(" ")])
    def model(self, request):
        return request.param