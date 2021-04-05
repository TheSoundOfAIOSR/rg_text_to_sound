#!pip install transformers
#!pip install sentence_transformers
from benchmarking_tools.model.prediction_model import PredictionModel


from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
class GensimPredictionModelWithPreprocessor(PredictionModel):
  family='Gensim'
  HuggingFaceURL = 'sentence-transformers/bert-base-nli-mean-tokens'
  def build(self):
    #text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    #preprocessor = hub.KerasLayer(self.preprocessor_url)
    self.preprocessor = AutoTokenizer.from_pretrained(HuggingFaceURL)
    encoder_inputs = self.preprocessor
    self.model = SentenceTransformer(HuggingFaceURL)

  def predict(self, sentences):
    output_tensor = self.model.encode(sentences)
  
    return output_tensor

  def additional_infos(self):
    return {
        "source":"HuggingFace",
        #"preprocessor_url":self.preprocessor_url,
#        "tf_hub_url":self.tf_hub_url,
        "family":"BERT",
        "word_level_output_available":'NotSure'
    }

class HuggingFace_bert_base_nli_mean_tokens(GensimPredictionModelWithPreprocessor):
	HuggingFaceURL ='sentence-transformers/bert-base-nli-mean-tokens'

class HuggingFace_ce_roberta_large_quora(GensimPredictionModelWithPreprocessor):
    HuggingFaceURL = 'sentence-transformers/ce-roberta-large-quora'

class HuggingFace_ce_roberta_base_stsb(GensimPredictionModelWithPreprocessor):
    HuggingFaceURL = 'sentence-transformers/ce-roberta-base-stsb'

class HuggingFace_stsb_distilbert_base(GensimPredictionModelWithPreprocessor):
    HuggingFaceURL = 'sentence-transformers/stsb-distilbert-base'


