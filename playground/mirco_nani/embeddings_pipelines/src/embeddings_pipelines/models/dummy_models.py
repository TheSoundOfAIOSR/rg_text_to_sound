import numpy as np
from embeddings_pipelines.model.models import (
    KeywordExtractionModel,
    WordEmbeddingModel,
    MultipleWordsEmbeddingModel,
    EmbeddingDimensionalityReductionModel,
    MultipleEmbeddingsDimensionalityReductionModel
)


class DummyKeywordExtractionModel(KeywordExtractionModel):
    def __init__(self, separator: str=None):
        """This keyword extractor splits the sentences based on a separator or whitespaces if the sepatator is not privided

        Args:
            separator (str, optional): The separator to use during prediction. Defaults to None.
        """
        self.separator=separator


    def build(self):
        pass



    def predict(self, sentence:str) -> np.array:
      """Applies the built model to the input sentence

      Args:
          sentence (str): A string containing a sentence

      Returns:
          np.array: a 1-D numpy array of strings containing the keywords extracted from the sentences
      """
      return np.array(sentence.split() if self.separator is None else sentence.split(self.separator))


class DummyWordEmbeddingModel(WordEmbeddingModel):
    def __init__(self, embedding_size: int):
        """this embedder produces a random word embeddings for each given word

        Args:
            embedding_size (int): the size of the output embedding
        """
        self.embedding_size = embedding_size

    def build(self):
        pass


    def predict(self, words:np.array) -> np.array:
      """Applies the built model to a 1-D numpy array of strings containing words

      Args:
          words (np.array): A 1-D numpy array of strings of length N containing words

      Returns:
          np.array: a 2-D numpy array of floats with shape (N,K) where N is is the 
                    number of input words and K is the embedding size
      """
      return np.random.rand(words.shape[0],self.embedding_size)


class DummyMultipleWordsEmbeddingModel(MultipleWordsEmbeddingModel):
    def __init__(self, embedding_size: int):
        """this embedder produces a single random word embeddings regardless of how meny words it receives

        Args:
            embedding_size (int): the size of the output embedding
        """
        self.embedding_size = embedding_size


    def build(self):
        pass

    
    def predict(self, words:np.array) -> np.array:
      """Applies the built model to a 1-D numpy array of strings containing words

      Args:
          words (np.array): A 1-D numpy array of strings of length N containing words

      Returns:
          np.array: a 1-D numpy array of floats with size K, being the embedding size
      """
      return np.random.Generator.uniform(size=(self.embedding_size,))


class DummyEmbeddingDimensionalityReductionModel(EmbeddingDimensionalityReductionModel):
    def __init__(self, reduced_embedding_size: int):
        """this embedder takes the first reduced_embedding_size dimensions of a given embedding

        Args:
            embedding_size (int): the size of the output embedding
        """
        self.reduced_embedding_size = reduced_embedding_size

    
    def build(self):
        pass


    def predict(self, embedding:np.array) -> np.array:
      """Applies the built model to reduce an embedding

      Args:
          embedding (np.array): A 1-D numpy array of floats, being the embedding to reduce

      Returns:
          np.array: a 1-D numpy array with dimensions lower than the input embedding
      """
      return embedding[:self.reduced_embedding_size]



class DummyMultipleEmbeddingsDimensionalityReductionModel(MultipleEmbeddingsDimensionalityReductionModel):
    def __init__(self, reduced_embedding_size: int):
        """this embedder takes the first reduced_embedding_size dimensions 
           of the dimension-wise mean of a given batch of embedding 

        Args:
            embedding_size (int): the size of the output embedding
        """
        self.reduced_embedding_size = reduced_embedding_size

    
    def build(self):
        pass


    def predict(self, embeddings:np.array) -> np.array:
      """Applies the built model to reduce an embedding

      Args:
          embedding (np.array): A 2-D numpy array of floats with shape (N,K) where N is is the 
                                number of input embeddings and K is the embedding size

      Returns:
          np.array: a 1-D numpy array with dimensions lower than the input embeddings dimesionality
      """
      return np.mean(embeddings, axis=0)[:self.reduced_embedding_size]