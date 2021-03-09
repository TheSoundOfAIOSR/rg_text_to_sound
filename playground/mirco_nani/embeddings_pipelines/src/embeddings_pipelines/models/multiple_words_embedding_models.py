import numpy as np
from embeddings_pipelines.model.models import MultipleWordsEmbeddingModel

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


    def dispose():
        pass