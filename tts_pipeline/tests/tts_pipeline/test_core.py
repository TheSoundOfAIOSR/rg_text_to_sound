from abc import ABC, abstractmethod
import pytest
import numpy as np

PIPELINE_INPUT_SENTENCES=[
    "give me a bright guitar",
    "GIVE ME A BRIGHT GUITAR",
    "GIVE ME A WARM GUIT",
    "HE GIVE ME A WARM GUITAR SOUND",
    "Lorem ipsum dolor sit amet"
]

class AbstractTestInferenceModel(ABC): #since this doesn't start with "Test" it won't get executed
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predict_input(self):
        pass

    @pytest.fixture
    def built_model(self, model):
        model.build()
        yield model
        model.dispose()

    @pytest.fixture
    def simple_prediction(self, built_model, predict_input):
        return built_model.predict(**predict_input)

    @pytest.fixture
    def same_simple_prediction(self, built_model, predict_input):
        return built_model.predict(**predict_input)


    def test_invariance(self, simple_prediction, same_simple_prediction):
        np.testing.assert_array_equal(simple_prediction, same_simple_prediction)


class AbstractTestInferencePipeline: 

    @pytest.fixture(params=PIPELINE_INPUT_SENTENCES)
    def predict_input(self, request):
        return request.param

    @pytest.fixture
    def built_pipeline(self, pipeline):
        pipeline.build()
        yield pipeline
        pipeline.dispose()


    @pytest.fixture
    def simple_prediction(self, built_pipeline, predict_input):
        return built_pipeline.predict(predict_input)


    def test_output_fields(self, simple_prediction):
        assert "velocity" in simple_prediction
        assert "pitch" in simple_prediction
        assert "source" in simple_prediction
        assert "qualities" in simple_prediction
        assert "latent_sample" in simple_prediction
