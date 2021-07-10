from ner_tests.utils import flatten_dict, assets_path
import pandas as pd
import json
import os

import typer


def results(results_folder, flattened=True):
    for model in os.listdir(results_folder):
        model_path = os.path.join(results_folder,model)
        if os.path.isfile(model_path): continue
        for model_export in os.listdir(model_path):
            if os.path.isfile(model_export): continue
            model_export_path = os.path.join(model_path,model_export)
            for test in os.listdir(model_export_path):
                if os.path.isdir(test): continue
                test_path = os.path.join(model_export_path, test)
                test_name = os.path.splitext(test)[0]
                result = json.load(open(test_path,"r"))
                result["model"] = model
                result["export"] = model_export
                result["test"] = test
                yield flatten_dict(result) if flattened else result 


def collect_results(
    results_folder = os.path.join(assets_path,"results"), 
    output_csv = os.path.join(assets_path,"results", "summary.csv")
):
    pd.DataFrame(list(results(results_folder))).to_csv(output_csv,index=False)


if __name__ == '__main__':
    typer.run(collect_results)