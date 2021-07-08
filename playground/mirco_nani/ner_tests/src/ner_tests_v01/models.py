import spacy


class NERKeywordExtractor:
    def __init__(self, model_path:str = get_default_model_path()):
        self.model_path = model_path

    def build(self):
        self.model = spacy.load(self.model_path)


    def predict(self, sentence: str) -> dict:
        prediction = self.model(sentence)
        instruments = [ent.text for ent in prediction.ents if ent.label_ == "INSTR"]
        qualities  = [ent.text for ent in prediction.ents if ent.label_ == "QLTY"]
        return {
            "QLTY": qualities,
            "INSTR"  : instruments
        }


    def dispose(self):
        del self.model