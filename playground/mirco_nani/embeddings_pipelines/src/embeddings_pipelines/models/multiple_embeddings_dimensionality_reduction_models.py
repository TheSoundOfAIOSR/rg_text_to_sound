import numpy as np
from embeddings_pipelines.model.models import MultipleEmbeddingsDimensionalityReductionModel


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


    def dispose(self):
        pass