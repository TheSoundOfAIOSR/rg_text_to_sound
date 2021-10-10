from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import UnifiedKeywordPairsExtractor

class UnifiedKeywordPairsExtractorV2(UnifiedKeywordPairsExtractor):
    def predict(self, sentence):
        """
        This class redefines the predict method in order to avoid an output empty keywords set.
        in order to do this, when the output set is empty, the word_to_words_matcher is used to 
        find a match to the full sentence instead of the single keywords
        """

        self._log("receved ", sentence)
        sentence = sentence.lower()
        ner_result = self.ner_keyword_extractor.predict(sentence)
        free_keywords = ner_result["soundquality"]
        self._log("recognized qualities: ", free_keywords)

        d = self.keyword_extractor_by_list.predict(sentence)
        velocity = d["velocity"]

        matched_words = self.word_to_words_matcher.predict(free_keywords) 
        self._log("matched keywords: ", matched_words)

        if len(matched_words) == 0:
            self._log("zero keywords detected, using WordsToWordsMatcher to match the full original sentence..")
            untokenizable_sentence=sentence.replace(" ","")
            matched_words = self.word_to_words_matcher.predict([untokenizable_sentence]) 
            self._log("matched keywords from full original sentence: ", matched_words)

        result = {
            "soundquality": list(set(matched_words)),
            "instrument"  : "acoustic",
            "velocity"    : 75, #velocity, #returning fixed velocity for now as our velocity estimator could produce negative values
            "pitch"       : 60
        }
        self._log("result: ", result)
        return result


def run():
    word_pairs = [
            ("Bright", "Dark"),
            ("Full",   "Hollow"),
            ("Smooth", "Rough"),
            ("Warm",   "Metallic"),
            ("Clear",  "Muddy"),
            ("Thin",   "Thick"),
            ("Pure",   "Noisy"),
            ("Rich",   "Sparse"),
            ("Soft",   "Hard")
        ]
    extractorPairs = UnifiedKeywordPairsExtractor(word_pairs, spacy_model='en_core_web_lg')
    extractorPairs.build()
    extractorPairsV2 = UnifiedKeywordPairsExtractorV2(word_pairs, spacy_model='en_core_web_lg', verbose=True)
    extractorPairsV2.build()
    for s in [
        "give me a bright guitar",
        "Lorem ipsum dolor sit amet"
    ]:
        print(s)
        print(extractorPairs.predict(s))
        print(extractorPairsV2.predict(s))
        print("---")


if __name__ == "__main__":
    run()