import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../..')
from tests.embeddings_pipelines.model.test_models import AbstractTestWordEmbeddingModel
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../src') )
from embeddings_pipelines.models.word_embedding_models import (
    DummyWordEmbeddingModel
)


class TestDummyWordEmbeddingModel(AbstractTestWordEmbeddingModel):
    embedding_size = 32
    @pytest.fixture
    def model(self):
        return DummyWordEmbeddingModel(self.embedding_size)

    def test_output_dimensionality(self, simple_prediction):
        assert simple_prediction.shape[-1] == self.embedding_size