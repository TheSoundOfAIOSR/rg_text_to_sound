from abc import ABC, abstractmethod
from typing import Any, List, Dict, Callable
import numpy as np
from tts_pipeline.utils.explain import explain


class InferenceModel(ABC):
    @abstractmethod
    def build(self):
        """ This method builds a prediction-ready model
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Applies the built model to the given input

        Args:
            generic inputs, these get specified on concrete classes

        Returns:
            a generic output, it gets specified in the concrete classes
        """
        pass

    @abstractmethod
    def dispose(self):
        """This method frees all the resources occupied in the build method
        """
        pass

    def explain(self) -> str:
        """ returns docstrings for build, predict and dispose methods
        """
        build_doc = explain(self.build)
        if build_doc is None or len(build_doc)==0:
            build_doc = explain(InferenceModel.build)

        predict_doc = explain(self.predict)
        if predict_doc is None or len(predict_doc)==0:
            predict_doc = explain(InferenceModel.predict)

        dispose_doc = explain(self.dispose)
        if dispose_doc is None or len(dispose_doc)==0:
            dispose_doc = explain(InferenceModel.dispose)

        result = f""" class {type(self).__name__}
        method build: {build_doc}


        method predict: {predict_doc}


        method dispose: {dispose_doc}
        """
        return result


class InferencePipeline(ABC):
    @abstractmethod
    def __init__(self, *args: List[InferenceModel], **kwargs: Dict[str, InferenceModel]):
        """Collects the input models to be used for the embedding task

        Args:
            *args (List[InferenceModel]): the prediction models needed by the pipeline
            **kwargs (Dict[str, InferenceModel]): the prediction models needed by the pipeline

        """
        pass

    @abstractmethod
    def build(self):
        """ This method builds the pipeline that will embed the sentences
        """
        pass

    @abstractmethod
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
        
    @abstractmethod
    def dispose(self):
        """This method frees all the resources occupied in the build method
        """
        pass

    def explain(self) -> str:
        """ returns docstrings for build, predict and dispose methods
        """
        build_doc = explain(self.build)
        if build_doc is None or len(build_doc)==0:
            build_doc = explain(InferencePipeline.build)

        predict_doc = explain(self.predict)
        if predict_doc is None or len(predict_doc)==0:
            predict_doc = explain(InferencePipeline.predict)
            
        dispose_doc = explain(self.dispose)
        if dispose_doc is None or len(dispose_doc)==0:
            dispose_doc = explain(InferencePipeline.dispose)

        result = f""" class {type(self).__name__}


        method build: {build_doc}


        method predict: {predict_doc}


        method dispose: {dispose_doc}
        """
        return result