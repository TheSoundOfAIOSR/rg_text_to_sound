from abc import ABC, abstractmethod
import pytest
import numpy as np

class AbstractTestPredictionModel(ABC):
    @abstractmethod
    def model(self):
        pass

    @pytest.fixture
    def simple_prediction(self, built_model):
        return built_model.predict(self.predict_input)

    @pytest.fixture
    def same_simple_prediction(self, built_model):
        return built_model.predict(self.predict_input)

    @pytest.fixture
    def built_model(self, model):
        model.build()
        yield model
        model.dispose()

    def test_invariance(self, simple_prediction, same_simple_prediction):
        np.testing.assert_array_equal(simple_prediction, same_simple_prediction)

    def test_prediction_output_dimensions(self, simple_prediction):
        assert len(simple_prediction.shape) == self.output_dimensions


class AbstractTestKeywordExtractionModel(AbstractTestPredictionModel):
    predict_input = "a simple sentence"
    output_dimensions = 1

    def test_prediction_contains_strings(self, simple_prediction):
        assert simple_prediction.dtype.char == 'S' or simple_prediction.dtype.char == 'U'


class AbstractTestWordEmbeddingModel(AbstractTestPredictionModel):
    predict_input = np.array(["word","another","word","more","words"])
    output_dimensions = 2

    def test_prediction_embeds_all_sentences(self, simple_prediction):
        assert simple_prediction.shape[0] == self.predict_input.shape[0]


class AbstractTestMultipleWordsEmbeddingModel(AbstractTestPredictionModel):
    predict_input = np.array(["word","another","word","more","words"])
    output_dimensions = 1


class AbstractTestEmbeddingDimensionalityReductionModel(AbstractTestPredictionModel):
    predict_input = np.array([1.0]*512)
    output_dimensions = 1

    def test_dimensionality_reduction(self, simple_prediction):
        assert simple_prediction.shape[0] <= self.predict_input.shape[0]


class AbstractTestMultipleEmbeddingDimensionalityReductionModel(AbstractTestPredictionModel):
    predict_input = np.array([[0.0]*512,[1.0]*512])
    output_dimensions = 1

    def test_dimensionality_reduction(self, simple_prediction):
        assert simple_prediction.shape[0] <= self.predict_input.shape[1]