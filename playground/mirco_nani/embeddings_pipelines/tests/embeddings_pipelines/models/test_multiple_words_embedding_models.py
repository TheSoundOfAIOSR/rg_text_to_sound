import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../..')
from tests.embeddings_pipelines.model.test_models import AbstractTestMultipleWordsEmbeddingModel
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../src') )
from embeddings_pipelines.models.multiple_words_embedding_models import (
    DummyMultipleWordsEmbeddingModel,
    TFHubPreTrainedBERTMultipleWordsEmbeddingModel
)


class TestDummyMultipleWordsEmbeddingModel(AbstractTestMultipleWordsEmbeddingModel):
    embedding_size = 32
    @pytest.fixture
    def model(self):
        return DummyMultipleWordsEmbeddingModel(self.embedding_size)

    def test_output_dimensionality(self, simple_prediction):
        assert simple_prediction.shape[0] == self.embedding_size


class TestTFHubPreTrainedBERTMultipleWordsEmbeddingModel(AbstractTestMultipleWordsEmbeddingModel):
    @pytest.fixture
    def model(self):
        return TFHubPreTrainedBERTMultipleWordsEmbeddingModel()