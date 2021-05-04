import tensorflow_hub as hub
from tts_pipeline.pipelines.waterfall.pipeline import WaterfallEmbedder

class GNewsWaterfallEmbedder(WaterfallEmbedder):
  def __init__(self, tf_hub_url:str = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"):
    self.tf_hub_url = tf_hub_url

  def build(self):
    self.model = hub.load(self.tf_hub_url)

  def predict(self, sentence:str, keyword_extraction_results:dict):
    result = {}
    result["sentence"] = self.model([sentence]).numpy()[0].tolist()
    result["soundquality"] = self.model(keyword_extraction_results["soundquality"]).numpy().tolist()
    result["instrument"] = self.model([keyword_extraction_results["instrument"]]).numpy()[0].tolist()
    return result

  def dispose(self):
    del self.model