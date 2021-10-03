import os, sys
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','tts_pipeline','src') )

from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import UnifiedKeywordPairsExtractor
from tts_pipeline.pipelines.waterfall.models.gnews_models import GNewsWaterfallEmbedder
from tts_pipeline.pipelines.waterfall.models.examples import DummyWaterfallDimensionalityReducer
from tts_websocketserver.utils import assets_folder

def get_pipeline():
    return WaterfallPipeline(
        keyword_extractor = UnifiedKeywordPairsExtractor(
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
            ner_model_path = os.path.join(assets_folder, "ner_model"),
            verbose = True
        ),
        embedder = GNewsWaterfallEmbedder(), # this is very small, so it runs fast
        dimensionality_reducer = DummyWaterfallDimensionalityReducer())
