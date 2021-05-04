import spacy
import sys, os

from tts_pipeline.pipelines.waterfall.pipeline import WaterfallKeywordExtractor

def get_default_model_path(): # ugly 
    return  os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','..','assets','ner_model')

class NERKeywordExtractor(WaterfallKeywordExtractor):
    def __init__(self, model_path:str = get_default_model_path()):
        self.model_path = model_path

    def build(self):
        self.model = spacy.load(self.model_path)

    def predict(self, sentence: str) -> dict:
        prediction = self.model(sentence)
        instrument = [ent.text for ent in prediction.ents if ent.label_ == "INSTR"][0]
        qualities  = [ent.text for ent in prediction.ents if ent.label_ == "QLTY"]
        return {
            "soundquality": qualities,
            "instrument"  : instrument,
            "velocity"    : 75,
            "pitch"       : 60
        }

    def dispose(self):
        del self.model