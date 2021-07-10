# ner_tests

## setup
* install requiremments: ```pip install -r requirements.txt```
* download testset csv files to assets/data (see assets/where_to_find_the_data.txt for more info)
* download trained models assets/models (see assets/where_to_find_the_models.txt for more info)

## running the tests:
```bash
cd src
python -m ner_tests.run AllTests
```  
results will be located under assets/results


## summary
To produce a summary csv of the results, run the following:
```bash
cd src
python -m ner_tests.summarize
``` 
This command will produce the file assets/results/summary.csv