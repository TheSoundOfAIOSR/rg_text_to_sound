from abc import ABC, abstractmethod

class PredictionModel(ABC):
  @abstractmethod
  def build(self):
    """ 
      This method builds a pre-trained model
    """
    pass

  @abstractmethod
  def predict(self, sentences):
    """
      Applies the built model to the given input sentences

      :param: sentences: an iterable of N strings

      :returns: an NxK numpy matrix where K is the embedding size
    """
    pass

  
  def additional_infos(self):
    """
      Provides additional informations that may be useful to track
      :returns: dictionary str->[str|number]
                example: {"family":"BERT","word_level_output_available":True}
    """
    return {}