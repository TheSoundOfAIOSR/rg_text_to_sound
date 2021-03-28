from tts_pipeline.pipelines.waterfall.pipeline import WaterfallPipeline
from tts_pipeline.pipelines.waterfall.models.examples import (
    DummyWaterfallKeywordExtractor,
    BERTWaterfallEmbedder,
    DummyWaterfallDimensionalityReducer
)

def get_pipeline():
    return WaterfallPipeline(
        keyword_extractor = DummyWaterfallKeywordExtractor(),
        embedder = BERTWaterfallEmbedder(),
        dimensionality_reducer = DummyWaterfallDimensionalityReducer())
