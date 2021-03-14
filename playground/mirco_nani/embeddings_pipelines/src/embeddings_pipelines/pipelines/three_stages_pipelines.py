from abc import ABC, abstractmethod
from typing import Any, List, Dict
import numpy as np
from embeddings_pipelines.model.pipelines import EmbeddingPipeline
from embeddings_pipelines.model.models import (
    PredictionModel,
    KeywordExtractionModel,
    WordEmbeddingModel,
    MultipleWordsEmbeddingModel,
    EmbeddingDimensionalityReductionModel,
    MultipleEmbeddingsDimensionalityReductionModel
)



class MergeAtWordEmbeddingStepPipeline(EmbeddingPipeline):
    """This pipeline contains the following steps:
    
     - keyword extraction: text sentence to sequence of words
     - word embedding: sequence of words to single highly-dimensional embedding
     - dimensionality reduction: single highly-dimensional embedding ot single low-dimensional embedding

    """

    def __init__(
        self, 
        keyword_extraction_model: KeywordExtractionModel,
        word_embedding_model: MultipleWordsEmbeddingModel,
        dimensionality_reduction_model: EmbeddingDimensionalityReductionModel
    ):
        """The constructor takes the definition of the models for each step of the pipeline

        Args:
            keyword_extraction_model (KeywordExtractionModel): 
                an embeddings_pipelines.model.KeywordExtractionModel
            word_embedding_model (MultipleWordsEmbeddingModel): 
                an embeddings_pipelines.model.MultipleWordsEmbeddingModel
            dimensionality_reduction_model (EmbeddingDimensionalityReductionModel): 
                an embeddings_pipelines.model.EmbeddingDimensionalityReductionModel
        """
        self.keyword_extraction_model=keyword_extraction_model
        self.word_embedding_model=word_embedding_model
        self.dimensionality_reduction_model=dimensionality_reduction_model

    def build(self):
        """builds the models
        """
        self.keyword_extraction_model.build()
        self.word_embedding_model.build()
        self.dimensionality_reduction_model.build()

    def embed(self, sentence: str) -> np.array:
        """uses the built models for each step of the pipeline

        Args:
            sentence (str): a string sentence

        Returns:
            np.array: a low-dimensional 1-D embedding
        """
        sequence_of_words = self.keyword_extraction_model.predict(sentence)
        big_embedding = self.word_embedding_model.predict(sequence_of_words)
        small_embedding = self.dimensionality_reduction_model.predict(big_embedding)
        return small_embedding

    def dispose(self):
        self.keyword_extraction_model.dispose()
        self.word_embedding_model.dispose()
        self.dimensionality_reduction_model.dispose()


class MergeAtDimensionalityReductionStepPipeline(EmbeddingPipeline):
    """This pipeline contains the following steps:
    
     - keyword extraction: text sentence to sequence of N words
     - word embedding: sequence of N words to batch of N highly-dimensional embeddings
     - dimensionality reduction: batch of N highly-dimensional embeddings to single low-dimensional embedding

    """

    def __init__(
        self, 
        keyword_extraction_model: KeywordExtractionModel,
        word_embedding_model: WordEmbeddingModel,
        dimensionality_reduction_model: MultipleEmbeddingsDimensionalityReductionModel
    ):
        """The constructor takes the definition of the models for each step of the pipeline

        Args:
            keyword_extraction_model (KeywordExtractionModel): 
                an embeddings_pipelines.model.KeywordExtractionModel
            word_embedding_model (WordEmbeddingModel): 
                an embeddings_pipelines.model.WordEmbeddingModel
            dimensionality_reduction_model (MultipleEmbeddingsDimensionalityReductionModel): 
                an embeddings_pipelines.model.MultipleEmbeddingsDimensionalityReductionModel
        """
        self.keyword_extraction_model=keyword_extraction_model
        self.word_embedding_model=word_embedding_model
        self.dimensionality_reduction_model=dimensionality_reduction_model

    def build(self):
        """builds the models
        """
        self.keyword_extraction_model.build()
        self.word_embedding_model.build()
        self.dimensionality_reduction_model.build()

    def embed(self, sentence: str) -> np.array:
        """uses the built models for each step of the pipeline

        Args:
            sentence (str): a string sentence

        Returns:
            np.array: a low-dimensional 1-D embedding
        """
        sequence_of_words = self.keyword_extraction_model.predict(sentence)
        big_embedding = self.word_embedding_model.predict(sequence_of_words)
        small_embedding = self.dimensionality_reduction_model.predict(big_embedding)
        return small_embedding

    def dispose(self):
        self.keyword_extraction_model.dispose()
        self.word_embedding_model.dispose()
        self.dimensionality_reduction_model.dispose()