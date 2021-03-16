import numpy as np
from embeddings_pipelines.model.models import WordEmbeddingModel


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
      return np.ones((words.shape[0],self.embedding_size))


    def dispose(self):
        pass