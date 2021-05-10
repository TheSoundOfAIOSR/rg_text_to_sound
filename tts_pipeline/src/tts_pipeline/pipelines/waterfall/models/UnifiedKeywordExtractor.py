import numpy as np
#import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_text as text  # Registers the ops.
from tts_pipeline.pipelines.waterfall.pipeline import (
    WaterfallKeywordExtractor,
    WaterfallEmbedder,
    WaterfallDimensionalityReducer
)
import spacy
from sklearn.cluster import KMeans
from tts_pipeline.core import InferenceModel


class VelocityEstimator(WaterfallKeywordExtractor):
    def __init__(self,model='en_core_web_lg',slow_str='',quick_str=''):
        self.slow_str = slow_str
        self.quick_str = quick_str
        self.model = model

    def build(self):
        # Load English tokenizer, tagger, parser and NER
        
        #TODO: Check if it is downloaded, and if needed download it.
        self.nlp = spacy.load(self.model)

        if not slow_str:
            slow_str=self.slow_str
            slow_str ="slow, super slow, snail, unhurried, leisurely, measured, moderate, deliberate, steady, sedate, slow-moving, slow-going, easy, relaxed, unrushed, gentle, undemanding, comfortable, ponderous, plodding, laboured, dawdling, loitering, lagging, laggard, sluggish, sluggardly, snail-like, tortoise-like, leaden-footed, leaden, creeping, laggy, lollygagging, calm, gently, docile, friendly, easy, dull, tediously, lazy, sleepily, tardy, indolent, graceful, largo, adagio, sluggish, relaxed, casual, belatedly, tardily, ritardando, latterly, lately, lenient, poor, dully, lethargically"
        if not quick_str:
            quick_str=self.quick_str
            quick_str = "speedy, quick, swift, rapid, brisk, nimble, sprightly, lively, fast-moving, high-speed, turbo, sporty, accelerated, express, flying, whirlwind, blistering, breakneck, pell-mell, meteoric, smart, hasty, hurried, unhesitating, expeditious, fleet-footed,supersonic, fleet, tantivy, alacritous, volant, secure, secured, fastened, tight, firmly fixed, stuck, jammed, immovable, unbudgeable, stiff, closed, shut, to, attach, fasten, fix, affix, join, connect, couple, quickly, rapidly, swiftly, speedily, briskly, at speed, at full speed, at full tilt, energetically, hastily,with all haste, in haste, hurriedly, in a hurry, post-haste, pell-mell, without delay, expeditiously, with dispatch, like a shot, like a flash, in a flash, in the blink of an eye, in a wink, in a trice, in no time, in no time at all, on the double, at the speed of light, like an arrow from a bow, double quick, in double quick time, pretty damn quick, nippily, like  lightning, at warp speed, like mad, like crazy, like the wind,"
        slow_list = [word.strip() for word in slow_str.split(',')]
        quick_list = [word.strip() for word in quick_str.split(',')]
        self.docs_slow = [self.nlp(f'Give me a {token.strip()} guitar. ') for token in slow_list]
        self.docs_quick = [self.nlp(f'Give me a {token.strip()} guitar. ') for token in quick_list]

    def get_mean_similarity(self,doc):
            sim_vals_slow = [doc_c.similarity(doc) for doc_c in self.docs_slow]
            sim_vals_quick = [doc_c.similarity(doc) for doc_c in self.docs_quick]
            xs=np.median(sim_vals_slow)
            xq=np.median(sim_vals_quick)
            return xs,xq

    def predict(self, sentence: str) -> dict:
        tokens      = sentence.split(" ")
        lengths     = [len(x) for x in tokens]
        max_len_idx = [i for i,l in enumerate(lengths) if l==max(lengths)][0]
        instrument   = tokens[max_len_idx]


        xs,xq = self.get_mean_similarity(self.nlp(sentence))
        velocity = (xs-xq)/0.006
        
        return velocity

    def dispose(self):
        del self.nlp
        del self.docs_slow
        del self.docs_quick


class WordToWordsMatcher(WaterfallKeywordExtractor):
    def __init__(self,target_words,model='en_core_web_lg'):
        self.target_words = target_words
        self.model=model

    def build(self):
        self.target_tokens = np.array(self.target_words)
        self.nlp = spacy.load(self.model)
        vector_array = self.get_vector_array(target_words)

        self.clusterer = KMeans(n_clusters=vector_array.shape[0],init='random')
        self.clusterer.cluster_centers_ = vector_array

    def get_vector_array(self,word_list,verbose=False):
        docstr = " ".join(word_list)
        target_tokens_doc = self.nlp(docstr)
        vector_list = []
        for token in target_tokens_doc:
            if verbose:
              print(token.text, token.has_vector, token.vector_norm, token.is_oov)
            vector_list.append(token.vector)
        return np.array(vector_list)

    def match_word_to_words(new_word):
        vector_array = self.get_vector_array(words)
        return clusterer.predict(vector_array.reshape(1,-1))

    def predict(self,words):
        """
        for a list of words, return a list of target words
        >>>target_words = ['slow', 'quick', 'yellow', 'loud', 'hard']
        >>>wwm = word_to_words_matcher()
        >>>wwm.build(target_words)
        >>>wwm.predict(target_words)
        >>>wwm.predict(['rigid','stiff']).tolist()
        <output still to be checked, hopefully ['hard','hard']>
        """
        vector_array = self.get_vector_array(words)
        clusterind = self.clusterer.predict(vector_array)
        return self.target_tokens[clusterind].tolist()

    def dispose(self):
        del self.nlp

    def test_word_to_words_matcher(self):
        """
        Code that might later be used to create tests.
        """
        import nltk
        from nltk.corpus import wordnet
        from collections import defaultdict
        target_words = ['slow', 'quick', 'yellow', 'loud', 'hard']
        wwm = word_to_words_matcher()
        wwm.build(target_words)

        wwm.predict(target_words)

        wwm.predict(['rigid','stiff']).tolist()

        """# Test the matching on synsets of the target words:"""

        nltk.download('wordnet')

        def get_synonyms(word):
          synonyms = []
          for syn in wordnet.synsets(word):
              for lm in syn.lemmas():
                      synonyms.append(lm.name())
          return set(synonyms)

        ','.join(wwm.predict(get_synonyms('quick')))

        good_dict = bad_dict=defaultdict(list)

        for target_word in target_words:
          for word in get_synonyms(target_word):
            prediction = wwm.predict([word])[0]
            if prediction==target_word:
              #print(f'{word}->{target_word}:ok!')
              good_dict[target_word]+=[word]
            else:
              #print(word,':',f'{prediction} (should be {target_word})')
              bad_dict[target_word]+=[word]

        print('synset words that were not mapped back to the target word:')
        for key,val in bad_dict.items():
          print(key,val)

        print('synset words that were correctly mapped back to the target word:')
        for key,val in good_dict.items():
          print(key,val)

class UnifiedKeywordExtractor(WaterfallKeywordExtractor):
    def __init__(self,target_words,slow_str='',quick_str=''):
        self.word_to_words_matcher = WordToWordsMatcher(target_words)
        self.velocity_estimator= VelocityEstimator(slow_str=slow_str,quick_str=quick_str)

    def build(self):        
        self.word_to_words_matcher.build()        
        self.velocity_estimator.build()

    def predict(self,sentence):
        #estimate the velocity given in sentence
        velocity = self.velocity_estimator.predict(sentence)
        #somehow generate a list of words given the sentence. This is still a bit too crude.
        #TODO: 
        # - stopword removal, 
        # - reduction to the relevant words in the sentence, e.g. only nounds or adjectives
        # - use more sophisticated tokenization?
        word_list = sentence.split(' ')
        #match the given word list to the target word list
        matched_words = self.word_to_words_matcher.predict(word_list) 
        d['soundquality']=matched_words
        d['velocity']=velocity
        return d

    def dispose(self):
        self.word_to_words_matcher.dispose()
        self.velocity_estimator.dispose()

if __name__=='__main__':
    target_words = ['slow', 'quick', 'yellow', 'loud', 'hard']
    uf = UnifiedKeywordExtractor(target_words)
    uf.build()
    
    uf.build(target_words)
    us.predict('give me a slow, dark guitar sound')
