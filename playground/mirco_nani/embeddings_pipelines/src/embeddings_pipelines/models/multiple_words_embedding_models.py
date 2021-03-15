import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.
from embeddings_pipelines.model.models import MultipleWordsEmbeddingModel

class DummyMultipleWordsEmbeddingModel(MultipleWordsEmbeddingModel):
    def __init__(self, embedding_size: int):
        """this embedder produces a single random word embeddings regardless of how meny words it receives

        Args:
            embedding_size (int): the size of the output embedding
        """
        self.embedding_size = embedding_size
        self.built=False


    def build(self):
        pass

    
    def predict(self, words:np.array) -> np.array:
        """Applies the built model to a 1-D numpy array of strings containing words

        Args:
            words (np.array): A 1-D numpy array of strings of length N containing words

        Returns:
            np.array: a 1-D numpy array of floats with size K, being the embedding size
        """
        return np.ones((self.embedding_size,))


    def dispose(self):
        pass


class TFHubPreTrainedBERTMultipleWordsEmbeddingModel(MultipleWordsEmbeddingModel):
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
        self.tf_hub_url = tf_hub_url
        self.preprocessor_url = preprocessor_url
        self.built = False


    def build(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(self.preprocessor_url)
        encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(self.tf_hub_url, trainable=False)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]      # [batch_size, 128].
        # sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 128].
        self.model = tf.keras.Model(text_input, pooled_output)
        self.built = True

    
    def predict(self, words:np.array) -> np.array:
        """Applies the built model to a 1-D numpy array of strings containing words

        Args:
            words (np.array): A 1-D numpy array of strings of length N containing words

        Returns:
            np.array: a 1-D numpy array of floats
        """
        assert self.built, "you need to call build() first!"
        sentences = np.array([" ".join(words)])
        sentences_tensor = tf.constant(sentences)
        output_tensor = self.model(sentences_tensor)
        return output_tensor.numpy()[0]


    def dispose(self):
        del self.model
        self.built = False