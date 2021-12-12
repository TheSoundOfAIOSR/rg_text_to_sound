import os, sys
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','tts_pipeline','src') )

from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordPairsExtractorV2 import UnifiedKeywordPairsExtractorV2
from tts_pipeline.pipelines.waterfall.models.zero_embedder import ZeroEmbedder
from tts_pipeline.pipelines.waterfall.models.examples import DummyWaterfallDimensionalityReducer
from tts_websocketserver.utils import assets_folder

def get_pipeline():
    return WaterfallPipeline(
        keyword_extractor = UnifiedKeywordPairsExtractorV2(
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
        embedder = ZeroEmbedder(),
        dimensionality_reducer = DummyWaterfallDimensionalityReducer())

if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.build()
    print(pipeline.predict("give me a bright guitar"))