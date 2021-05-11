from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import UnifiedKeywordExtractor
from tts_pipeline.pipelines.waterfall.models.gnews_models import GNewsWaterfallEmbedder
from tts_pipeline.pipelines.waterfall.models.examples import DummyWaterfallDimensionalityReducer
from tts_websocketserver.utils import assets_folder
import os

def get_pipeline():
    return WaterfallPipeline(
        keyword_extractor = UnifiedKeywordExtractor(
            target_words = ["Bright","Dark","Full","Hollow","Smooth","Rough","Warm","Metallic","Clear","Muddy","Thin","thick","Pure","Noisy","Rich","Sparse","Soft","Hard"],
            ner_model_path = os.path.join(assets_folder, "ner_model")
        ),
        embedder = GNewsWaterfallEmbedder(), # this is very small, so it runs fast
        dimensionality_reducer = DummyWaterfallDimensionalityReducer())
