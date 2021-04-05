from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class PredictionModel(ABC):
  @abstractmethod
  def build(self):
    """ This method builds a prediction-ready model
    """
    pass

  @abstractmethod
  def predict(self, input:Any):
    """Applies the built model to the given input

    Args:
        input (Any): a generic input

    Returns:
        a generic output
    """
    pass

  @abstractmethod
  def dispose(self):
    """This method frees all the resources occupied in the build method
    """
    pass


class KeywordExtractionModel(PredictionModel):
    @abstractmethod
    def predict(self, sentence:str) -> np.array:
      """Applies the built model to the input sentence

      Args:
          sentence (str): A string containing a sentence

      Returns:
          np.array: a 1-D numpy array of strings containing the keywords extracted from the sentences
      """
      pass


class WordEmbeddingModel(PredictionModel):
    @abstractmethod
    def predict(self, words:np.array) -> np.array:
      """Applies the built model to a 1-D numpy array of strings containing words

      Args:
          words (np.array): A 1-D numpy array of strings of length N containing words

      Returns:
          np.array: a 2-D numpy array of floats with shape (N,K) where N is is the 
                    number of input words and K is the embedding size
      """
      pass


class MultipleWordsEmbeddingModel(PredictionModel):
    @abstractmethod
    def predict(self, words:np.array) -> np.array:
      """Applies the built model to a 1-D numpy array of strings containing words

      Args:
          words (np.array): A 1-D numpy array of strings of length N containing words

      Returns:
          np.array: a 1-D numpy array of floats with size K, being the embedding size
      """
      pass


class EmbeddingDimensionalityReductionModel(PredictionModel):
    @abstractmethod
    def predict(self, embedding:np.array) -> np.array:
      """Applies the built model to reduce an embedding

      Args:
          embedding (np.array): A 1-D numpy array of floats, being the embedding to reduce

      Returns:
          np.array: a 1-D numpy array with dimensions lower than the input embedding
      """
      pass



class MultipleEmbeddingsDimensionalityReductionModel(PredictionModel):
    @abstractmethod
    def predict(self, embeddings:np.array) -> np.array:
      """Applies the built model to reduce an embedding

      Args:
          embedding (np.array): A 2-D numpy array of floats with shape (N,K) where N is is the 
                                number of input embeddings and K is the embedding size

      Returns:
          np.array: a 1-D numpy array with dimensions lower than the input embeddings dimesionality
      """
      pass