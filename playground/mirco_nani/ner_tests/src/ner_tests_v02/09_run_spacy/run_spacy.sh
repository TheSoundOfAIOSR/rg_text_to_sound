#!/bin/bash

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

assets_dir=${current_dir}/../../../assets
model_dir=${assets_dir}/models/model_both/training/model-best
data_file=${assets_dir}/data/spacy/test_context.spacy
output_file=${assets_dir}/results/test_context_results.json

displacy_path=${assets_dir}/results/test2/displacy
displacy_limit=1696

python -m spacy evaluate ${model_dir} ${data_file} --output ${output_file} --displacy-path ${displacy_path} --displacy-limit ${displacy_limit}
