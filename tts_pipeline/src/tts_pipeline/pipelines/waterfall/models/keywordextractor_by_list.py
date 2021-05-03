import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.
from tts_pipeline.pipelines.waterfall.pipeline import (
    WaterfallKeywordExtractor,
    WaterfallEmbedder,
    WaterfallDimensionalityReducer
)
import spacy
#from nltk.corpus import wordnet as wn

class KeywordExtractorByList(WaterfallKeywordExtractor):
    def build(self):

        # Load English tokenizer, tagger, parser and NER
        self.nlp = spacy.load("en_core_web_sm")

    def predict(self, sentence: str) -> dict:
        tokens      = sentence.split(" ")
        lengths     = [len(x) for x in tokens]
        max_len_idx = [i for i,l in enumerate(lengths) if l==max(lengths)][0]
        instrument   = tokens[max_len_idx]

        doc1 = nlp("slow, super slow, snail, unhurried, leisurely, measured, moderate, deliberate, steady, sedate, slow-moving, slow-going, easy, relaxed, unrushed, gentle, undemanding, comfortable, ponderous, plodding, laboured, dawdling, loitering, lagging, laggard, sluggish, sluggardly, snail-like, tortoise-like, leaden-footed, leaden, creeping, laggy, lollygagging, calm, gently, docile, friendly, easy, dull, tediously, lazy, sleepily, tardy, indolent, graceful, largo, adagio, sluggish, relaxed, casual, belatedly, tardily, ritardando, latterly, lately, lenient, poor, dully, lethargically")
        doc2 = nlp("
speedy, quick, swift, rapid, brisk, nimble, sprightly, lively, fast-moving, high-speed, turbo, sporty, accelerated, express, flying, whirlwind, blistering, breakneck, pell-mell, meteoric, smart, hasty, hurried, unhesitating, expeditious, fleet-footed, nippy, zippy, spanking, scorching, blinding, supersonic, cracking, fleet, tantivy, alacritous, volant, secure, secured, fastened, tight, firmly fixed, stuck, jammed, immovable, unbudgeable, stiff, closed, shut, to, attach, fasten, fix, affix, join, connect, couple, link, tie, tie up, bind, fetter, strap, rope, tether, truss, lash, hitch, moor, anchor, yoke, chain, quickly, rapidly, swiftly, speedily, briskly, at speed, at full speed, at full tilt, energetically, hastily,
with all haste, in haste, hurriedly, in a hurry, post-haste, pell-mell, without delay, expeditiously, with dispatch, like a shot, like a flash, in a flash, in the blink of an eye, in a wink, in a trice, in no time (at all), on the double, at the speed of light, like an arrow from a bow, double quick, in double quick time, p.d.q. (pretty damn quick), nippily, like (greased) lightning, at warp speed, hell for leather, like mad, like crazy, like the wind, like a bomb, like nobody's business, like a scalded cat, like the deuce, a mile a minute, like a bat out of hell, like the clappers, at a rate of knots, like billy-o, lickety-split, apace, 2., securely, tightly, immovably, fixedly" )
        sim_doc1 = doc.similarity(doc1)
        sim_doc2 = doc.similarity(doc2)

        velocity = 0+100*(sim_doc1-sim_doc2)/(sim_doc1+sim_doc2)
        return {
            "soundquality": tokens,
            "instrument"  : instrument,
            "velocity"    : 75,
            "pitch"       : 60
        }

    def dispose(self):
        pass

import numpy as np
import spacy
class TrainedKeywordExtractorByList(WaterfallKeywordExtractor):
    def build(self):

        
                
        # Load English tokenizer, tagger, parser and NER
        nlp = spacy.load("en_core_web_lg")
        slow_str ="slow, super slow, snail, unhurried, leisurely, measured, moderate, deliberate, steady, sedate, slow-moving, slow-going, easy, relaxed, unrushed, gentle, undemanding, comfortable, ponderous, plodding, laboured, dawdling, loitering, lagging, laggard, sluggish, sluggardly, snail-like, tortoise-like, leaden-footed, leaden, creeping, laggy, lollygagging, calm, gently, docile, friendly, easy, dull, tediously, lazy, sleepily, tardy, indolent, graceful, largo, adagio, sluggish, relaxed, casual, belatedly, tardily, ritardando, latterly, lately, lenient, poor, dully, lethargically"
        quick_str = "speedy, quick, swift, rapid, brisk, nimble, sprightly, lively, fast-moving, high-speed, turbo, sporty, accelerated, express, flying, whirlwind, blistering, breakneck, pell-mell, meteoric, smart, hasty, hurried, unhesitating, expeditious, fleet-footed,supersonic, fleet, tantivy, alacritous, volant, secure, secured, fastened, tight, firmly fixed, stuck, jammed, immovable, unbudgeable, stiff, closed, shut, to, attach, fasten, fix, affix, join, connect, couple, quickly, rapidly, swiftly, speedily, briskly, at speed, at full speed, at full tilt, energetically, hastily,with all haste, in haste, hurriedly, in a hurry, post-haste, pell-mell, without delay, expeditiously, with dispatch, like a shot, like a flash, in a flash, in the blink of an eye, in a wink, in a trice, in no time, in no time at all, on the double, at the speed of light, like an arrow from a bow, double quick, in double quick time, pretty damn quick, nippily, like  lightning, at warp speed, like mad, like crazy, like the wind,"
        slow_list = [word.strip() for word in slow_str.split(',')]
        quick_list = [word.strip() for word in quick_str.split(',')]
        self.docs_slow = [nlp(f'Give me a {token.strip()} guitar. ') for token in slow_list]
        self.docs_quick = [nlp(f'Give me a {token.strip()} guitar. ') for token in quick_list]
        self.nlp = spacy.load("en_core_web_lg")

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

        doc1 = nlp("slow, super slow, snail, unhurried, leisurely, measured, moderate, deliberate, steady, sedate, slow-moving, slow-going, easy, relaxed, unrushed, gentle, undemanding, comfortable, ponderous, plodding, laboured, dawdling, loitering, lagging, laggard, sluggish, sluggardly, snail-like, tortoise-like, leaden-footed, leaden, creeping, laggy, lollygagging, calm, gently, docile, friendly, easy, dull, tediously, lazy, sleepily, tardy, indolent, graceful, largo, adagio, sluggish, relaxed, casual, belatedly, tardily, ritardando, latterly, lately, lenient, poor, dully, lethargically")
        doc2 = nlp("
speedy, quick, swift, rapid, brisk, nimble, sprightly, lively, fast-moving, high-speed, turbo, sporty, accelerated, express, flying, whirlwind, blistering, breakneck, pell-mell, meteoric, smart, hasty, hurried, unhesitating, expeditious, fleet-footed, nippy, zippy, spanking, scorching, blinding, supersonic, cracking, fleet, tantivy, alacritous, volant, secure, secured, fastened, tight, firmly fixed, stuck, jammed, immovable, unbudgeable, stiff, closed, shut, to, attach, fasten, fix, affix, join, connect, couple, link, tie, tie up, bind, fetter, strap, rope, tether, truss, lash, hitch, moor, anchor, yoke, chain, quickly, rapidly, swiftly, speedily, briskly, at speed, at full speed, at full tilt, energetically, hastily,
with all haste, in haste, hurriedly, in a hurry, post-haste, pell-mell, without delay, expeditiously, with dispatch, like a shot, like a flash, in a flash, in the blink of an eye, in a wink, in a trice, in no time (at all), on the double, at the speed of light, like an arrow from a bow, double quick, in double quick time, p.d.q. (pretty damn quick), nippily, like (greased) lightning, at warp speed, hell for leather, like mad, like crazy, like the wind, like a bomb, like nobody's business, like a scalded cat, like the deuce, a mile a minute, like a bat out of hell, like the clappers, at a rate of knots, like billy-o, lickety-split, apace, 2., securely, tightly, immovably, fixedly" )
        sim_doc1 = doc.similarity(doc1)
        sim_doc2 = doc.similarity(doc2)

        xs,xq = self.get_mean_similarity(self.nlp(sentence))
        velocity = (xs-xq)/0.006
        

        return {
            "soundquality": None,
            "instrument"  : None,
            "velocity"    : velocity,
            "pitch"       : None
        }

    def dispose(self):
        del self.nlp
        del self.docs_slow
        del self.docs_quick