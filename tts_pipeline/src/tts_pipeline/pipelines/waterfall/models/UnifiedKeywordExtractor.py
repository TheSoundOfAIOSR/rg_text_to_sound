import numpy as np
from tts_pipeline.pipelines.waterfall.pipeline import WaterfallKeywordExtractor
from tts_pipeline.pipelines.waterfall.models.ner_model import NERKeywordExtractor
import spacy
from sklearn.cluster import KMeans


# from tts_pipeline.core import InferenceModel

class KeywordExtractorByList(WaterfallKeywordExtractor):
    def build(self):
        # Load English tokenizer, tagger, parser and NER
        # you might need to run
        # python -m spacy download en_core_web_sm
        # first
        self.nlp = spacy.load("en_core_web_sm")
        self.doc1 = self.nlp("""slow, super slow, snail, unhurried, leisurely, measured, moderate, deliberate, steady, 
        sedate, slow-moving, slow-going, easy, relaxed, unrushed, gentle, undemanding, comfortable, ponderous, 
        plodding, laboured, dawdling, loitering, lagging, laggard, sluggish, sluggardly, snail-like, tortoise-like, 
        leaden-footed, leaden, creeping, laggy, lollygagging, calm, gently, docile, friendly, easy, dull, tediously,
        lazy, sleepily, tardy, indolent, graceful, largo, adagio, sluggish, relaxed, casual, belatedly, tardily, 
        ritardando, latterly, lately, lenient, poor, dully, lethargically""")
        self.doc2 = self.nlp("""speedy, quick, swift, rapid, brisk, nimble, sprightly, lively, fast-moving, high-speed, 
        turbo, sporty, accelerated, express, flying, whirlwind, blistering, breakneck, pell-mell, meteoric, smart, 
        hasty, hurried, unhesitating, expeditious, fleet-footed, nippy, zippy, spanking, scorching, blinding, 
        supersonic, cracking, fleet, tantivy, alacritous, volant, secure, secured, fastened, tight, firmly fixed, 
        stuck, jammed, immovable, unbudgeable, stiff, closed, shut, to, attach, fasten, fix, affix, join, connect, 
        couple, link, tie, tie up, bind, fetter, strap, rope, tether, truss, lash, hitch, moor, anchor, yoke, chain, 
        quickly, rapidly, swiftly, speedily, briskly, at speed, at full speed, at full tilt, energetically, hastily, 
        with all haste, in haste, hurriedly, in a hurry, post-haste, pell-mell, without delay, expeditiously, 
        with dispatch, like a shot, like a flash, in a flash, in the blink of an eye, in a wink, 
        in a trice, in no time (at all), on the double, at the speed of light, like an arrow from a bow, 
        double quick, in double quick time, p.d.q. (pretty damn quick), nippily, like (greased) lightning, 
        at warp speed, hell for leather, like mad, like crazy, like the wind, like a bomb, like nobody's business, 
        like a scalded cat, like the deuce, a mile a minute, like a bat out of hell, like the clappers, 
        at a rate of knots, like billy-o, lickety-split, apace, 2., securely, tightly, immovably, fixedly""")

    def predict(self, sentence: str) -> dict:
        tokens = sentence.split(" ")
        lengths = [len(x) for x in tokens]
        max_len_idx = [i for i, l in enumerate(lengths) if l == max(lengths)][0]
        instrument = tokens[max_len_idx]
        doc = self.nlp(sentence)
        sim_doc1 = doc.similarity(self.doc1)
        sim_doc2 = doc.similarity(self.doc2)

        velocity = 0 + 100 * (sim_doc1 - sim_doc2) / (sim_doc1 + sim_doc2)
        return {
            "soundquality": tokens,
            "instrument": instrument,
            "velocity": velocity,
            "pitch": 60
        }

    def dispose(self):
        pass


class TrainedKeywordExtractorByList(WaterfallKeywordExtractor):
    def build(self):
        # Load English tokenizer, tagger, parser and NER
        self.nlp = spacy.load("en_core_web_lg")
        slow_str = "slow, super slow, snail, unhurried, leisurely, measured, moderate, deliberate, steady, sedate, slow-moving, slow-going, easy, relaxed, unrushed, gentle, undemanding, comfortable, ponderous, plodding, laboured, dawdling, loitering, lagging, laggard, sluggish, sluggardly, snail-like, tortoise-like, leaden-footed, leaden, creeping, laggy, lollygagging, calm, gently, docile, friendly, easy, dull, tediously, lazy, sleepily, tardy, indolent, graceful, largo, adagio, sluggish, relaxed, casual, belatedly, tardily, ritardando, latterly, lately, lenient, poor, dully, lethargically"
        quick_str = "speedy, quick, swift, rapid, brisk, nimble, sprightly, lively, fast-moving, high-speed, turbo, sporty, accelerated, express, flying, whirlwind, blistering, breakneck, pell-mell, meteoric, smart, hasty, hurried, unhesitating, expeditious, fleet-footed,supersonic, fleet, tantivy, alacritous, volant, secure, secured, fastened, tight, firmly fixed, stuck, jammed, immovable, unbudgeable, stiff, closed, shut, to, attach, fasten, fix, affix, join, connect, couple, quickly, rapidly, swiftly, speedily, briskly, at speed, at full speed, at full tilt, energetically, hastily,with all haste, in haste, hurriedly, in a hurry, post-haste, pell-mell, without delay, expeditiously, with dispatch, like a shot, like a flash, in a flash, in the blink of an eye, in a wink, in a trice, in no time, in no time at all, on the double, at the speed of light, like an arrow from a bow, double quick, in double quick time, pretty damn quick, nippily, like  lightning, at warp speed, like mad, like crazy, like the wind,"
        slow_list = [word.strip() for word in slow_str.split(',')]
        quick_list = [word.strip() for word in quick_str.split(',')]
        self.docs_slow = [self.nlp(f'Give me a {token.strip()} guitar. ') for token in slow_list]
        self.docs_quick = [self.nlp(f'Give me a {token.strip()} guitar. ') for token in quick_list]
        self.nlp = spacy.load("en_core_web_lg")

    def get_mean_similarity(self, doc):
        sim_vals_slow = [doc_c.similarity(doc) for doc_c in self.docs_slow]
        sim_vals_quick = [doc_c.similarity(doc) for doc_c in self.docs_quick]
        xs = np.median(sim_vals_slow)
        xq = np.median(sim_vals_quick)
        return xs, xq

    def predict(self, sentence: str) -> dict:
        tokens = sentence.split(" ")
        lengths = [len(x) for x in tokens]
        max_len_idx = [i for i, l in enumerate(lengths) if l == max(lengths)][0]
        instrument = tokens[max_len_idx]
        doc = self.nlp(sentence)
        doc1 = self.nlp(
            "slow, super slow, snail, unhurried, leisurely, measured, moderate, deliberate, steady, sedate, slow-moving, slow-going, easy, relaxed, unrushed, gentle, undemanding, comfortable, ponderous, plodding, laboured, dawdling, loitering, lagging, laggard, sluggish, sluggardly, snail-like, tortoise-like, leaden-footed, leaden, creeping, laggy, lollygagging, calm, gently, docile, friendly, easy, dull, tediously, lazy, sleepily, tardy, indolent, graceful, largo, adagio, sluggish, relaxed, casual, belatedly, tardily, ritardando, latterly, lately, lenient, poor, dully, lethargically")
        doc2 = self.nlp("""speedy, quick, swift, rapid, brisk, nimble, sprightly, lively, fast-moving, high-speed, turbo, 
        sporty, accelerated, express, flying, whirlwind, blistering, breakneck, pell-mell, meteoric, smart, hasty, 
        hurried, unhesitating, expeditious, fleet-footed, nippy, zippy, spanking, scorching, blinding, supersonic, 
        cracking, fleet, tantivy, alacritous, volant, secure, secured, fastened, tight, firmly fixed, stuck, jammed, 
        immovable, unbudgeable, stiff, closed, shut, to, attach, fasten, fix, affix, join, connect, couple, link, tie, 
        tie up, bind, fetter, strap, rope, tether, truss, lash, hitch, moor, anchor, yoke, chain, quickly, rapidly, 
        swiftly, speedily, briskly, at speed, at full speed, at full tilt, energetically, hastily,
        with all haste, in haste, hurriedly, in a hurry, post-haste, pell-mell, without delay, expeditiously, with dispatch,
        like a shot, like a flash, in a flash, in the blink of an eye, in a wink, in a trice, in no time (at all),
        on the double, at the speed of light, like an arrow from a bow, double quick, in double quick time, nippily,
        like (greased) lightning, at warp speed, hell for leather, like mad, like crazy, like the wind, like a bomb, 
        like nobody's business, like a scalded cat, like the deuce, a mile a minute, like a bat out of hell, 
        like the clappers, at a rate of knots, like billy-o, lickety-split, apace, 2., securely, tightly, 
        immovably, fixedly""")
        #sim_doc1 = doc.similarity(doc1)
        #sim_doc2 = doc.similarity(doc2)

        xs, xq = self.get_mean_similarity(self.nlp(sentence))
        velocity = (xs - xq) / 0.006

        return {
            "soundquality": None,
            "instrument": None,
            "velocity": velocity,
            "pitch": None
        }

    def dispose(self):
        del self.nlp
        del self.docs_slow
        del self.docs_quick


class WordToWordsMatcher(WaterfallKeywordExtractor):
    def __init__(self, target_words, spacy_model='en_core_web_lg'):
        self.target_words=target_words
        self.spacy_model = spacy_model

    def build(self):
        """
            Target words
        """
        target_words = self.target_words
        # you might need to run python -m spacy download en_core_web_lg first
        try:
            self.nlp = spacy.load(self.spacy_model)
        except OSError:
            print(
                f'Language model not found. You might need to run python -m spacy download {self.spacy_model} to download the model')
            raise

        self.target_words = target_words
        self.vector_array = self.get_vector_array(target_words)

        self.clusterer = KMeans(n_clusters=self.vector_array.shape[0], init='random')
        self.clusterer.fit(self.vector_array)
        self.clusterer.cluster_centers_ = self.vector_array

    def get_vector_array(self, word_list, verbose=False):
        docstr = " ".join(word_list)
        target_tokens_doc = self.nlp(docstr)
        vector_list = []
        for token in target_tokens_doc:
            if verbose:
                print(token.text, token.has_vector, token.vector_norm, token.is_oov)
            vector_list.append(token.vector)
        return np.array(vector_list)

    def match_word_to_words(self, words):
        vector_array = self.get_vector_array(words)
        return self.clusterer.predict(vector_array.reshape(1, -1))

    def predict(self, words):
        """
        for a list of words, return a list of target words
        >>> target_words = ['slow', 'quick', 'yellow', 'loud', 'hard']
        >>> wwm = WordToWordsMatcher()
        >>> wwm.build(target_words)
        >>> #wwm.predict(target_words)
        >>> wwm.predict(['rigid','stiff'])
        ['hard', 'hard']
        >>> target_words = ['slow', 'quick', 'yellow', 'loud', 'hard']
        >>> wwm.build(target_words)
        >>> wwm.predict(target_words)
        ['slow', 'quick', 'yellow', 'loud', 'hard']
        """
        vector_array = self.get_vector_array(words)
        clusterind = self.clusterer.predict(vector_array)
        ret_val = [self.target_words[i] for i in clusterind]
        return ret_val

    def dispose(self):
        del self.nlp


def test_word_to_words_matcher():
    """
    Code that might later be used to create tests.
    """
    target_words = ['slow', 'quick', 'yellow', 'loud', 'hard']
    wwm = WordToWordsMatcher()
    wwm.build()

    wwm.predict(target_words)

    wwm.predict(['rigid', 'stiff'])

    """# Test the matching on synsets of the target words:"""

    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet

    def get_synonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lm in syn.lemmas():
                synonyms.append(lm.name())
        return set(synonyms)

    ','.join(wwm.predict(get_synonyms('quick')))

    from collections import defaultdict
    good_dict = bad_dict = defaultdict(list)

    for target_word in target_words:
        for word in get_synonyms(target_word):
            prediction = wwm.predict([word])[0]
            if prediction == target_word:
                # print(f'{word}->{target_word}:ok!')
                good_dict[target_word] += [word]
            else:
                # print(word,':',f'{prediction} (should be {target_word})')
                bad_dict[target_word] += [word]

    print('synset words that were not mapped back to the target word:')
    for key, val in bad_dict.items():
        print(key, val)

    print('synset words that were correctly mapped back to the target word:')
    for key, val in good_dict.items():
        print(key, val)


class UnifiedKeywordExtractor(WaterfallKeywordExtractor):
    def __init__(self, target_words = ['slow', 'quick', 'yellow', 'loud', 'hard'], ner_model_path = None, spacy_model='en_core_web_lg'):
        if ner_model_path is not None:
            self.ner_keyword_extractor = NERKeywordExtractor(ner_model_path)
        else:
            self.ner_keyword_extractor = NERKeywordExtractor()
        self.word_to_words_matcher = WordToWordsMatcher(target_words, spacy_model)
        self.keyword_extractor_by_list = KeywordExtractorByList()
        

    def build(self):
        self.ner_keyword_extractor.build()
        self.word_to_words_matcher.build()
        self.keyword_extractor_by_list.build()

    def predict(self, sentence):
        ner_result = self.ner_keyword_extractor.predict(sentence)
        free_keywords = ner_result["soundquality"]

        d = self.keyword_extractor_by_list.predict(sentence)
        velocity = d["velocity"]

        matched_words = self.word_to_words_matcher.predict(free_keywords)  # TODO: use a more sophisticated tokenizer
        # matched_words is a list of target words. I'm not yet sure how to return them.
        # TODO: somehow merge matched_words into d before returning d
        return {
            "soundquality": list(set(matched_words)),
            "instrument"  : "acoustic",
            "velocity"    : 75, #velocity, #returning fixed velocity for now as our velocity estimator could produce negative values
            "pitch"       : 60
        }

    def dispose(self):
        self.ner_keyword_extractor.dispose()
        self.word_to_words_matcher.dispose()
        self.keyword_extractor_by_list.dispose()


if __name__ == '__main__':
    #ukwe = UnifiedKeywordExtractor()
    #target_words = ['slow', 'quick', 'yellow', 'loud', 'hard']
    #ukwe.build()
    target_words = ["Bright","Dark","Full","Hollow","Smooth","Rough","Warm","Metallic","Smooth","Rough","Clear","Muddy","Thin","thick","Pure","Noisy","Rich","Sparse","Soft","Hard"]

    wwm = WordToWordsMatcher(target_words)
    wwm.build()
    wwm.predict(target_words)
    print(wwm.predict(['rigid','stiff']))

    wwm = WordToWordsMatcher(target_words)
    wwm.build()
    print(wwm.predict(target_words))

    extractor = UnifiedKeywordExtractor(target_words, spacy_model='en_core_web_lg')
    extractor.build()
    print(extractor.predict("give me a rock sound guitar"))


