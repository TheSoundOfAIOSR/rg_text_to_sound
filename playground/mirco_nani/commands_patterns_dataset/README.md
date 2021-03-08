# Commands Patterns Dataset
Simple tools to generate a sentences dataset from the combinations of patterns with tokens replaced by all possible associated keywords.  
Example: 
 * pattern: "give me a \<ADJECTIVE\> \<INSTRUMENT\>"
 * tokens to keywords: {  
    "\<ADJECTIVE\>" : ["dark","bright","soft","hard","lovely"],  
    "\<INSTRUMENT\>" : ["guitar","violin","piano","cello"]  
}  
 * result: 
    * give me a dark guitar
    * give me a bright guitar
    * give me a dark violin
    * give me a bright violin
    * ...

An intuitive sample usage can be found in [CommandsPatternsDataset_sample_usage.ipynb](notebooks/CommandsPatternsDataset_sample_usage.ipynb)