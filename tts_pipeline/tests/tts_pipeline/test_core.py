from abc import ABC, abstractmethod
import pytest
import numpy as np

DEFAULT_SENTENCE="give me a bright guitar"

class AbstractTestInferenceModel(ABC): #since this doesn't start with "Test" it won't get executed
    @abstractmethod
    def model(self):
        pass

    @pytest.fixture
    def built_model(self, model):
        model.build()
        yield model
        model.dispose()

    @pytest.fixture
    def simple_prediction(self, built_model):
        return built_model.predict(**self.predict_input)

    @pytest.fixture
    def same_simple_prediction(self, built_model):
        return built_model.predict(**self.predict_input)


    def test_invariance(self, simple_prediction, same_simple_prediction):
        np.testing.assert_array_equal(simple_prediction, same_simple_prediction)


class AbstractTestInferencePipeline: 
    predict_input = DEFAULT_SENTENCE

    @pytest.fixture
    def built_pipeline(self, pipeline):
        pipeline.build()
        yield pipeline
        pipeline.dispose()


    @pytest.fixture
    def simple_prediction(self, built_pipeline):
        return built_pipeline.predict(self.predict_input)


    def test_output_field_velocity(self, simple_prediction):
        assert "velocity" in simple_prediction


    def test_output_field_pitch(self, simple_prediction):
        assert "pitch" in simple_prediction


    def test_output_field_source(self, simple_prediction):
        assert "source" in simple_prediction


    def test_output_field_qualities(self, simple_prediction):
        assert "qualities" in simple_prediction


    def test_output_field_latent_sample(self, simple_prediction):
        assert "latent_sample" in simple_prediction