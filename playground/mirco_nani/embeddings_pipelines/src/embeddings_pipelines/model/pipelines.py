from abc import ABC, abstractmethod
from typing import Any, List, Dict
import numpy as np
from embeddings_pipelines.model.models import PredictionModel

class EmbeddingPipeline(ABC):
    @abstractmethod
    def __init__(self, *args: List[PredictionModel], **kwargs: Dict[str, PredictionModel]):
        """Collects the input models to be used for the embedding task

        Args:
            *args (List[PredictionModel]): the prediction models needed by the pipeline
            **kwargs (Dict[str, PredictionModel]): the prediction models needed by the pipeline

        """
        pass

    @abstractmethod
    def build(self):
        """ This method builds the pipeline that will embed the sentences
        """
        pass

    @abstractmethod
    def embed(self, sentence:str) -> np.array:
        """Embeds a sentence in a low-dimensional array

        Args:
            sentence (str): a sentence

        Returns:
            np.array: a 1-D numpy array of floats
        """
        
    @abstractmethod
    def dispose(self):
        """This method frees all the resources occupied in the build method
        """
        pass