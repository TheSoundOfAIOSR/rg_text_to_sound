from ner_tests.utils import assets_path, ensure_dir, filename
from ner_tests.preprocess.csv_to_spacy import convert_file
import luigi
from luigi.contrib.external_program import ExternalProgramTask
import os


class Preprocess(luigi.Task):
    csv_file: str = luigi.Parameter()
    spacy_file: str = luigi.Parameter()

    def output(self): return luigi.LocalTarget(self.spacy_file)
    def run(self): 
        ensure_dir(os.path.dirname(self.spacy_file))
        convert_file(input_path=self.csv_file, output_path=self.spacy_file) 
        

class Test(ExternalProgramTask):
    csv_file: str = luigi.Parameter()
    spacy_file: str = luigi.Parameter(default="")
    model_path: str = luigi.Parameter()
    output_file: str = luigi.Parameter()

    def get_spacy_file(self):
        spacy_file = self.spacy_file
        if len(self.spacy_file) == 0:
            spacy_file = os.path.join(assets_path, "spacy", filename(self.csv_file)+".spacy")
        return spacy_file

    def output(self): return luigi.LocalTarget(self.output_file)

    def requires(self):
        return Preprocess(csv_file=self.csv_file, spacy_file=self.get_spacy_file())

    def program_args(self):
        ensure_dir(os.path.dirname(self.output_file))
        command = "python -m spacy evaluate".split(" ")
        command.append(self.model_path)
        command.append(self.get_spacy_file())
        command.append("--output="+self.output_file)
        return command


class AllTests(luigi.WrapperTask):
    csv_folder     = luigi.Parameter(default=os.path.join(assets_path, "data"))
    models_folder  = luigi.Parameter(default=os.path.join(assets_path, "models"))
    results_folder = luigi.Parameter(default=os.path.join(assets_path, "results"))

    def tests_parameters(self):
        for model_dirname in os.listdir(self.models_folder):
            for inner_model_dir in ["model-best", "model-last"]:
                for csv_file in os.listdir(self.csv_folder):
                    csv_filename = os.path.splitext(csv_file)[0]
                    input_file = os.path.join(self.csv_folder,csv_file)
                    model_path = os.path.join(self.models_folder, model_dirname, "training", inner_model_dir)
                    out_file_p = os.path.join(self.results_folder, model_dirname, inner_model_dir, csv_filename+".json")
                    yield input_file, model_path, out_file_p

    def requires(self):
        for input_file, model_path, out_file_p in self.tests_parameters():
            yield Test(csv_file=input_file, model_path=model_path, output_file=out_file_p)


