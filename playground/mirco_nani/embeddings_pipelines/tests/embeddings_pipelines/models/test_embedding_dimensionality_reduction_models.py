import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../..')
from tests.embeddings_pipelines.model.test_models import AbstractTestEmbeddingDimensionalityReductionModel
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../src') )
from embeddings_pipelines.models.embedding_dimensionality_reduction_models import (
    DummyEmbeddingDimensionalityReductionModel,
    IdentityEmbeddingDimensionalityReductionModel
)


class TestDummyEmbeddingDimensionalityReductionModel(AbstractTestEmbeddingDimensionalityReductionModel):
    embedding_size = 32
    @pytest.fixture
    def model(self):
        return DummyEmbeddingDimensionalityReductionModel(self.embedding_size)

    def test_output_dimensionality(self, simple_prediction):
        assert simple_prediction.shape[0] == self.embedding_size


class TestIdentityEmbeddingDimensionalityReductionModel(AbstractTestEmbeddingDimensionalityReductionModel):
    @pytest.fixture
    def model(self):
        return IdentityEmbeddingDimensionalityReductionModel()