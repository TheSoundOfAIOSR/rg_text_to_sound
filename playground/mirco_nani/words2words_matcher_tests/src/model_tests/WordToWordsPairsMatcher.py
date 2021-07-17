from test import test_model_and_save_results
import os, sys

repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', '..', '..', '..', '..')
tts_pipeline_path = os.path.join(repo_root, 'tts_pipeline', 'src')
sys.path.append(tts_pipeline_path)

from tts_pipeline.pipelines.waterfall.models.UnifiedKeywordExtractor import WordToWordsPairsMatcher

def test_model(data_path, results_dest):
    model = WordToWordsPairsMatcher()
    model.build()
    test_model_and_save_results(data_path,model,results_dest)
    model.dispose()

if __name__ == '__main__':
    test_model(*sys.argv[1:])