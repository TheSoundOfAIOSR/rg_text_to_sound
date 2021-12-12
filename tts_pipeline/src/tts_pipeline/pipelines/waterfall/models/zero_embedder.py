from tts_pipeline.pipelines.waterfall.pipeline import WaterfallEmbedder
import numpy as np

class ZeroEmbedder(WaterfallEmbedder):
  def __init__(self, out_dim=64):
    self.out_dim=out_dim
    def model(inp):
        return np.zeros([len(inp),self.out_dim])
    self.model=model

  def build(self):
    pass

  def predict(self, sentence:str, keyword_extraction_results:dict):
    result = {}
    result["sentence"] = self.model([sentence])[0].tolist()
    result["soundquality"] = self.model(keyword_extraction_results["soundquality"]).tolist()
    result["instrument"] = self.model([keyword_extraction_results["instrument"]])[0].tolist()
    return result

  def dispose(self):
    pass