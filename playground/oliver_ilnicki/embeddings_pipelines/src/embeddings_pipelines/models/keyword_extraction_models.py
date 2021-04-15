import numpy as np
from embeddings_pipelines.model.models import KeywordExtractionModel
import spacy

class PosExtraktor(KeywordExtractionModel):
    def __init__(self, separator: str=None):
        """This keyword extractor splits the sentences based on a separator or whitespaces if the sepatator is not privided

        Args:
            separator (str, optional): The separator to use during prediction. Defaults to None.
        """
        self.separator=separator  # FIXME  update seperator


    def build(self):
        nlp = spacy.load("en_core_web_sm")
        pass



    def predict(self, sentence:str) -> np.array:
      """Applies the built model to the input sentence

      Args:
          sentence (str): A string containing a sentence

      Returns:
          np.array: a 1-D numpy array of strings containing the keywords extracted from the sentences
      """
      doc = nlp(sentence)
      output = [word.text for word in doc if word.pos_ == "NOUN" or word.pos_ == "ADJ" ] #FIXME Update Part of Speech choice
      output = np.asarray(output)

      return output


    def dispose(self):
        pass
