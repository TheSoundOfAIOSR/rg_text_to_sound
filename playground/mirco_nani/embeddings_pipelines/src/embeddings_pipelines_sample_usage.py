def run_example1():
    # import the pipeline that you want to build
    from embeddings_pipelines.pipelines.three_stages_pipelines import MergeAtDimensionalityReductionStepPipeline
    # import the steps that will compose the pipeline
    from embeddings_pipelines.models.keyword_extraction_model import DummyKeywordExtractionModel
    from embeddings_pipelines.models.word_embedding_models import DummyWordEmbeddingModel
    from embeddings_pipelines.models.multiple_embeddings_dimensionality_reduction_model import DummyMultipleEmbeddingsDimensionalityReductionModel

    # define the pipeline steps with your favourite parameters
    step1 = DummyKeywordExtractionModel(separator = " ")
    step2 = DummyWordEmbeddingModel(embedding_size = 256)
    step3 = DummyMultipleEmbeddingsDimensionalityReductionModel(reduced_embedding_size=16)
    # define the pipeline with these steps 
    pipeline = MergeAtDimensionalityReductionStepPipeline(
        keyword_extraction_model = step1,
        word_embedding_model = step2,
        dimensionality_reduction_model = step3
    )
    # build the pipeline, this way the three models will be built as well
    pipeline.build()
    # use the pipeline on input sentences to get their embeddings
    sentence1 = "this is a sentence"
    sentence2 = "this is another sentence, longer than the first one"
    emb1 = pipeline.embed(sentence1)
    emb2 = pipeline.embed(sentence2)
    print(f""" "{sentence1}" -> {emb1} """)
    print(f""" "{sentence2}" -> {emb2} """)


if __name__ == "__main__":
    run_example1()