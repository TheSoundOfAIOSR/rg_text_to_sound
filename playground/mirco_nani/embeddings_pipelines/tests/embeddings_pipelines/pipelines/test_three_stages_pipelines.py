import pytest
import numpy as np
import os, sys
os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../..')
from tests.embeddings_pipelines.model.test_pipelines import AbstractTestEmbeddingPipeline
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../src') )
from embeddings_pipelines.pipelines.three_stages_pipelines import (
    MergeAtWordEmbeddingStepPipeline
)
from embeddings_pipelines.models.keyword_extraction_models import DummyKeywordExtractionModel
from embeddings_pipelines.models.multiple_words_embedding_models import (
    DummyMultipleWordsEmbeddingModel,
    TFHubPreTrainedBERTMultipleWordsEmbeddingModel
)
from embeddings_pipelines.models.embedding_dimensionality_reduction_models import (
    DummyEmbeddingDimensionalityReductionModel,
    IdentityEmbeddingDimensionalityReductionModel)
from embeddings_pipelines.models.word_embedding_models import DummyWordEmbeddingModel
from embeddings_pipelines.models.multiple_embeddings_dimensionality_reduction_models import DummyMultipleEmbeddingsDimensionalityReductionModel



class TestMergeAtWordEmbeddingStepPipeline(AbstractTestEmbeddingPipeline):
    high_dimensionality_embedding_size = 512
    low_dimensionality_embedding_size = 32
    @pytest.fixture(params=[
        MergeAtWordEmbeddingStepPipeline(
            keyword_extraction_model = DummyKeywordExtractionModel(),
            word_embedding_model = DummyMultipleWordsEmbeddingModel(high_dimensionality_embedding_size),
            dimensionality_reduction_model = DummyEmbeddingDimensionalityReductionModel(low_dimensionality_embedding_size)),
        MergeAtWordEmbeddingStepPipeline(
            keyword_extraction_model = DummyKeywordExtractionModel(),
            word_embedding_model = TFHubPreTrainedBERTMultipleWordsEmbeddingModel(),
            dimensionality_reduction_model = IdentityEmbeddingDimensionalityReductionModel())
    ])
    def pipeline(self, request):
        return request.param




class MergeAtDimensionalityReductionStepPipeline(AbstractTestEmbeddingPipeline):
    high_dimensionality_embedding_size = 512
    low_dimensionality_embedding_size = 32
    @pytest.fixture(params=[
        MergeAtWordEmbeddingStepPipeline(
            keyword_extraction_model = DummyKeywordExtractionModel(),
            word_embedding_model = DummyWordEmbeddingModel(high_dimensionality_embedding_size),
            dimensionality_reduction_model = DummyMultipleEmbeddingsDimensionalityReductionModel(low_dimensionality_embedding_size))
    ])
    def pipeline(self, request):
        return request.param
