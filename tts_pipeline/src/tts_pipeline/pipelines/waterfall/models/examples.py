import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.
from tts_pipeline.pipelines.waterfall.pipeline import (
    WaterfallKeywordExtractor,
    WaterfallEmbedder,
    WaterfallDimensionalityReducer
)

class DummyWaterfallKeywordExtractor(WaterfallKeywordExtractor):
    def build(self):
        pass

    def predict(self, sentence: str) -> dict:
        tokens      = sentence.split(" ")
        lengths     = [len(x) for x in tokens]
        max_len_idx = [i for i,l in enumerate(lengths) if l==max(lengths)][0]
        instrument   = tokens[max_len_idx]
        return {
            "soundquality": tokens,
            "instrument"  : instrument,
            "velocity"    : 75,
            "pitch"       : 60
        }

    def dispose(self):
        pass


class BERTWaterfallEmbedder(WaterfallEmbedder):
    def __init__(self, 
        tf_hub_url: str = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1",
        preprocessor_url: str = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        ):
        """this embedder uses a pre-trained BERT model to embed input words.
           Input words are concatenated in a single "sentence". 
           This "sentence" is then fed to the model in order to produce an embedding

        Args:
            tf_hub_url (str): the tensorflow hub URL of the model (default: the fastest one).
                              For a list of availabel embedding models, see 
                              https://tfhub.dev/s?fine-tunable=yes&language=en&tf-version=tf2&q=bert
            tf_hub_url (str): the tensorflow hub URL of the preprocessor module fot this model (default: the only one).
        """
        self.tf_hub_url       = tf_hub_url
        self.preprocessor_url = preprocessor_url
        self.built            = False


    def build(self):
        text_input     = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor   = hub.KerasLayer(self.preprocessor_url)
        encoder_inputs = preprocessor(text_input)
        encoder        = hub.KerasLayer(self.tf_hub_url, trainable=False)
        outputs        = encoder(encoder_inputs)
        pooled_output  = outputs["pooled_output"]      # [batch_size, 128].
        # sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 128].
        self.model     = tf.keras.Model(text_input, pooled_output)
        self.built     = True

    def predict(self, 
                sentence:str,
                keyword_extraction_results:dict) -> dict:
        assert self.built, "you need to call build() first!"
        sentences        = [sentence, keyword_extraction_results["instrument"]] + keyword_extraction_results["soundquality"]
        sentences_tensor = tf.constant(sentences)
        output_tensor    = self.model(sentences_tensor).numpy()
        return {
            "sentence"    : output_tensor[0].tolist(),
            "instrument"  : output_tensor[1].tolist(),
            "soundquality": output_tensor[2:].tolist()
        }

    def dispose(self):
        del self.model
        self.built = False


class DummyWaterfallDimensionalityReducer(WaterfallDimensionalityReducer):
    def build(self):
        pass

    def predict(self, 
                sentence:str,
                keyword_extraction_results:dict,
                embedding_results:dict) -> np.array:
        return embedding_results["sentence"][:16]

    def dispose(self):
        pass