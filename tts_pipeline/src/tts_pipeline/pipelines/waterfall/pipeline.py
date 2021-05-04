from abc import ABC, abstractmethod
from typing import Any, List, Dict
import numpy as np
from tts_pipeline.core import InferenceModel, InferencePipeline

_ = """
This pipeline applies the WATERFALL PATTERN: 
all the upstream informations are available downstream
"""

class WaterfallKeywordExtractor(InferenceModel):
    @abstractmethod
    def predict(self, sentence:str) -> dict:
        """ Extracts informations from a sentence.
                              
        Args:
            sentence (str): A string containing a sentence

        Returns:
            dict: a dictionary having these fields
                * "soundquality": a list of strings containing words.
                                    these words are the attributes of 
                                    an instrument 
                * "instrument"  : a string containing an instrument
                * "pitch"       : a number indicating the pitch 
                                    of a MIDI note
                * "velocity"    : a number indicating the velocity 
                                    of a MIDI note
        """
        pass


class WaterfallEmbedder(InferenceModel):
    @abstractmethod
    def predict(self, 
                sentence:str,
                keyword_extraction_results:dict) -> dict:
        """ receives informations from a KeywordExtractionModel
            and processes them to produce embedding data.

        Args:
            sentence (str): A string containing a sentence
            keyword_extraction_results (dict): a dictionary having 
                these fields
                * "soundquality": a list of strings containing words.
                                  these words are the attributes of 
                                  an instrument 
                * "instrument"  : a string containing an instrument
                * "pitch"       : a number indicating the pitch 
                                  of a MIDI note
                * "velocity"    : a number indicating the velocity 
                                  of a MIDI note

      Returns:
          dict: a dictionary having all or a subset of these fields
                * "sentence"    : the embedding of sentence
                * "soundquality": a list of containing the embeddings of 
                                  the words in the "soundquality" field
                                  of keyword_extraction_results
                * "instrument"  : an embedding of the word contained in 
                                  the "instrument" field of 
                                  keyword_extraction_results
        """
        pass


class WaterfallDimensionalityReducer(InferenceModel):
    @abstractmethod
    def predict(self, 
                sentence:str,
                keyword_extraction_results:dict,
                embedding_results:dict) -> np.array:
        """ receives informations from KeywordExtractionModel
            and EmbeddingModel and processes them to produce
            a single embedding which has low dimensionality.

        Args:
            sentence (str): A string containing a sentence
            keyword_extraction_results (dict): a dictionary having 
                these fields
                * "soundquality": a list of strings containing words.
                                  these words are the attributes of 
                                  an instrument 
                * "instrument"  : a string containing an instrument
                * "pitch"       : a number indicating the pitch 
                                  of a MIDI note
                * "velocity"    : a number indicating the velocity 
                                  of a MIDI note

            embedding_results (dict):  a dictionary having 
                these fields. ALL FIELDS ARE NULLABLE
                * "sentence"    : the embedding of sentence
                * "soundquality": a list of containing the embeddings of 
                                  the words in the "soundquality" field
                                  of keyword_extraction_results
                * "instrument"  : an embedding of the word contained in 
                                  the "instrument" field of 
                                  keyword_extraction_results

        Returns:
            np.array: a low-dimensional embedding vector
        """
        pass


class WaterfallPipeline(InferencePipeline):
    def __init__(self, 
                 keyword_extractor: WaterfallKeywordExtractor,
                 embedder: WaterfallEmbedder,
                 dimensionality_reducer: WaterfallDimensionalityReducer):
        """This pipeline receives a sentence as input and contains the following steps:
        
        - keyword extraction: outputs informations extracted from the sentence such as 
                            soundquality, instrument, pitch and velocity
        - word embedding: outputs embedded informations about the input sentence, 
                            soundquality, instrument. not all of these may be returned 
        - dimensionality reduction: outputs a low-dimensional embedding

        """
        self.keyword_extractor      = keyword_extractor
        self.embedder               = embedder
        self.dimensionality_reducer = dimensionality_reducer

    def build(self):
        self.keyword_extractor.build()
        self.embedder.build()
        self.dimensionality_reducer.build()

    def predict(self, sentence:str) -> dict:
        """Embeds a sentence in a low-dimensional array

        Args:
            sentence (str): a sentence

        Returns:
            dict: a dictionary having the following fields
                * "source"        : a string containing an instrument, or more 
                                    precisely, a kind of guitar (eg. acoustic)
                * "pitch"         : a number indicating the pitch of a MIDI note
                * "velocity"      : a number indicating the velocity of a MIDI note
                * "qualilites"    : a list of strings containing attributes
                * "latent_sample" : a low-dimensional embedding
        """
        keyword_extraction_results = self.keyword_extractor.predict(sentence)
        embedding_results          = self.embedder.predict(sentence,
                                                           keyword_extraction_results)
        complete_embedding_results = [embedding_results.get(k,None) 
                                      for k in ["sentence", "soundquality", "instrument"]]
        latent_sample              = self.dimensionality_reducer.predict(sentence,
                                                                         keyword_extraction_results,
                                                                         embedding_results)
        return {
            "source"        : keyword_extraction_results["instrument"],
            "pitch"         : keyword_extraction_results["pitch"],
            "velocity"      : keyword_extraction_results["velocity"],
            "qualities"     : keyword_extraction_results["soundquality"],
            "latent_sample" : latent_sample
        }

    def dispose(self):
        self.keyword_extractor.dispose()
        self.embedder.dispose()
        self.dimensionality_reducer.dispose()
