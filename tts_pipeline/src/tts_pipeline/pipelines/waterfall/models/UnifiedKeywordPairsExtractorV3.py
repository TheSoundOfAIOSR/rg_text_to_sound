from tts_pipeline.pipelines.waterfall.models.ner_model import NERKeywordExtractor
from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import KeywordExtractorByList
from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import WordToWordsPairsMatcher
from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import UnifiedKeywordPairsExtractor
import numpy as np



class NERKeywordExtractorV2(NERKeywordExtractor):
    def get_spacy_doc(self, sentence):
        return self.model(sentence)


class WordToWordsPairsMatcherV2(WordToWordsPairsMatcher):
    def get_spacy_doc(self, sentence):
        return self.nlp(sentence)

    def _compute_indices_and_distances(self, vector_array):
        clusterind = self.clusterer.predict(vector_array)
        distances = self.clusterer.transform(vector_array)
        return clusterind, distances


    def predict_vectors(self, vector_array):
        if len(vector_array) == 0:
            return []
        clusterind, distances = self._compute_indices_and_distances(vector_array)
        accumulators = np.zeros((len(self.words_pairs),2))
        accumulator_initialized = np.zeros((len(self.words_pairs),),dtype=bool)
        for i,dist in zip(clusterind,distances):
            accumulators_index = self.index_to_pair_index[i] # index in the list of pairs
            first_word_i, second_word_i = self.index_to_indices[i] # indices in the flattened list of pairs
            first_distance, second_distance = dist[first_word_i], dist[second_word_i] # distance of closest word and its opposite
            accumulator_initialized[accumulators_index] = True # mark the accumulator as initialized
            accumulators[accumulators_index][0] += first_distance # add the distances to the selected accumulator
            accumulators[accumulators_index][1] += second_distance # add the distances to the selected accumulator
        res = []
        for w,d,init in zip(self.words_pairs, accumulators, accumulator_initialized):
            if init:
                if d[0] - d[1] < -self.tolerance:
                    res.append(w[0])
                elif d[0] - d[1] > self.tolerance:
                    res.append(w[1])
                else: # if they are equal (up to a certain tolerance) we output neither of them
                    continue
        return res



class UnifiedKeywordPairsExtractorV3(UnifiedKeywordPairsExtractor):
    def __init__(self, 
        words_pairs = [
            ("Bright", "Dark"),
            ("Full",   "Hollow"),
            ("Smooth", "Rough"),
            ("Warm",   "Metallic"),
            ("Clear",  "Muddy"),
            ("Thin",   "Thick"),
            ("Pure",   "Noisy"),
            ("Rich",   "Sparse"),
            ("Soft",   "Hard")
        ],
        ner_model_path = None, 
        spacy_model='en_core_web_lg',
        verbose=False
    ):
        if ner_model_path is not None:
            self.ner_keyword_extractor = NERKeywordExtractorV2(ner_model_path)
        else:
            self.ner_keyword_extractor = NERKeywordExtractorV2()
        self.word_to_words_matcher = WordToWordsPairsMatcherV2(words_pairs, spacy_model)
        self.keyword_extractor_by_list = KeywordExtractorByList()
        self.verbose = verbose

    def predict(self, sentence):
        matched_words=self._match_keywords(sentence)
        result = {
            "soundquality": list(set(matched_words)),
            "instrument"  : "acoustic",
            "velocity"    : 75, #velocity, #returning fixed velocity for now as our velocity estimator could produce negative values
            "pitch"       : 60
        }
        self._log("result: ", result)
        return result

    def _match_keywords(self, sentence):
        """
        this method applies a series of functions to match kywords to the sentence in order to avoid an output empty keywords set.
        in order to do this, when the output set is empty, four fallback functions are applied, 
        every subsequent function is invoked if the previous produces an empty output. in order:
        - find the instrument, use the connected adjectives up to two levels of depth
        - use all adjectives in the sentence
        - use a vector representation of the full sentence
        """

        self._log("receved ", sentence)
        sentence = sentence.lower()
        ner_doc = self.ner_keyword_extractor.get_spacy_doc(sentence)
        spacy_doc = self.word_to_words_matcher.get_spacy_doc(sentence)

        self._log("step 0: finding adjectives with the NER model")
        tokens = self.get_ner_adjectives(ner_doc, spacy_doc)
        self._log("found adjectives: ", [t.text for t in tokens])
        matched_keywords=self.word_to_words_matcher.predict_vectors(self._tokens_to_vectors(tokens))
        self._log("matched keywords: ", matched_keywords)
        if len(matched_keywords) != 0:
            return matched_keywords
        
        self._log("step 1: finding adjectives connected to the instrument")
        tokens = self.get_instrument_related_adjectives(ner_doc, spacy_doc, search_depth=2)
        self._log("found adjectives: ", [t.text for t in tokens])
        matched_keywords=self.word_to_words_matcher.predict_vectors(self._tokens_to_vectors(tokens))
        self._log("matched keywords: ", matched_keywords)
        if len(matched_keywords) != 0:
            return matched_keywords

        self._log("step 2: finding all adjectives in the sentence")
        tokens = self.get_adjectives(spacy_doc)
        self._log("found adjectives: ", [t.text for t in tokens])
        matched_keywords=self.word_to_words_matcher.predict_vectors(self._tokens_to_vectors(tokens))
        self._log("matched keywords: ", matched_keywords)
        if len(matched_keywords) != 0:
            return matched_keywords

        self._log("step 3: using the vector representation of the full sentence")
        doc_vector = self.get_doc_vector(spacy_doc)
        matched_keywords=self.word_to_words_matcher.predict_vectors(np.array([doc_vector]))
        self._log("matched keywords: ", matched_keywords)
        if len(matched_keywords) != 0:
            return matched_keywords

        return matched_keywords

    def _tokens_to_vectors(self, tokens):
        return np.array([t.vector for t in tokens])

    def get_entity_tokens(self, entity, doc):
        return [token for token in doc if token.idx >= entity.start_char and token.idx+len(token.text) <= entity.end_char]


    def get_related_adjectives(self, tokens, search_depth):
        analysed_tokens = [t for t in tokens]
        for i in range(search_depth):
            analysed_tokens.extend([child for token in tokens for child in token.children if child not in analysed_tokens])
            analysed_tokens.extend([token.head for token in analysed_tokens if token.head not in analysed_tokens])
        self._log("analysed_tokens ",analysed_tokens)
        token_adjectives=[token for token in analysed_tokens if token.pos_=='ADJ']
        return token_adjectives


    # level 0
    def get_ner_adjectives(self, ner_doc, spacy_doc):
        adjective_entities=[ent for ent in ner_doc.ents if ent.label_ == "QLTY"]
        adjective_tokens=[self.get_entity_tokens(ent, spacy_doc) for ent in adjective_entities]
        adjective_tokens=[item for sublist in adjective_tokens for item in sublist] #flatten
        return adjective_tokens


    # level 1
    def get_instrument_related_adjectives(self, ner_doc, spacy_doc, search_depth=2):
        instrument_entities=[ent for ent in ner_doc.ents if ent.label_ == "INSTR"]
        self._log("instrument_entities ",instrument_entities)
        instrument_tokens=[self.get_entity_tokens(ent, spacy_doc) for ent in instrument_entities]
        self._log("instrument_tokens ",instrument_tokens)
        instrument_tokens=[item for sublist in instrument_tokens for item in sublist] #flatten
        self._log("instrument_tokens ",instrument_tokens)
        instrument_token_nouns=[token for token in instrument_tokens if token.pos_=='NOUN']
        self._log("instrument_token_nouns ",instrument_token_nouns)
        instrument_token_adjectives=self.get_related_adjectives(instrument_token_nouns, search_depth)
        self._log("instrument_token_adjectives ",instrument_token_adjectives)
        return instrument_token_adjectives


    # level 2
    def get_adjectives(self, spacy_doc):
        return [token for token in spacy_doc if token.pos_=='ADJ']


    # level 3
    def get_doc_vector(self, spacy_doc):
        return spacy_doc.vector



def explain_syntax(s, nlp):
  doc=nlp(s)

  print("TEXT	LEMMA	POS	TAG	DEP	SHAPE	ALPHA	STOP	MORPH	START	END".replace("	","\t "))
  for token in doc:
      print(f"{token.text}, {token.lemma_}, {token.pos_}, {token.tag_}, {token.dep_}, {token.shape_}, {token.is_alpha}, {token.is_stop}, {token.morph.to_json()}, {token.idx}, {token.idx+len(token.text)}".replace(",","\t"))

  print("")
  for ent in doc.ents:
        print(f" \t ".join(
          [str(x) for x in [ent.text, ent.label_, ent.start_char, ent.end_char]]
        ))

  tok=[t for t in doc]
  print(tok[0] in tok)
  print(tok[0] in tok[1:])
  print(tok[-1] in tok[:-1])


def run():
    extractorPairs = UnifiedKeywordPairsExtractorV3(spacy_model='en_core_web_lg', verbose=True)
    extractorPairs.build()
    for s in [
        "give me a very bright guitar",
        "give me a bright guitar",
        "give me a warm guitar",
        "GIVE ME A BRIGHT GUITAR",
        "GIVE ME A WARM GUIT",
        "HE GIVE ME A WARM GUITAR SOUND",
        "Lorem ipsum dolor sit amet"
    ]:
        print("--------------------------------------------------")
        print(extractorPairs.predict(s))
        print("---")


if __name__ == "__main__":
    run()