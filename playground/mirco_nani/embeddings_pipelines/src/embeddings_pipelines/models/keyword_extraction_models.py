import numpy as np
from embeddings_pipelines.model.models import KeywordExtractionModel

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


    def dispose(self):
        pass