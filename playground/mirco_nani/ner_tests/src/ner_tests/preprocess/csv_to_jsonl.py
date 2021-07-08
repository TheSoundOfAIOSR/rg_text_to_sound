from io import StringIO
from tqdm import tqdm
import pandas as pd
import typer
import json


class KeywordNotFoundError(Exception): pass


def line_to_dict(sentence, in_tokens):
    original_sentence=sentence
    cursor=0
    tokens=[]
    spans=[]
    for keyword, token in in_tokens:
        start = sentence.find(keyword)
        if start == -1:
            raise KeywordNotFoundError("could not find {keyword} in {original_sentence}")
        end   = start + len(keyword)
        tokens.append({"text": keyword, "start": start+cursor, "end": end+cursor})
        if token != "NONE":
            spans.append({"start": start+cursor, "end": end+cursor, "label": token})
        cursor += end
        sentence = sentence[end:]
    return {"text":original_sentence, "tokens":tokens, "spans":spans, "answer":"accept"}


def convert_file(input_path: str, output_path: str):
    header = True
    with open(input_path, "r") as in_f, open(output_path, "w") as out_f:
        for line in tqdm(in_f):
            if header:
                header=False
                continue
            sentence, tokens = pd.read_csv(StringIO(line), header=None, usecols=[0,1]).values[0]
            tokens = eval(tokens)
            dict_line = line_to_dict(sentence, tokens)
            json_line = json.dumps(dict_line)
            out_f.write(json_line+"\n")


if __name__ == '__main__':
    typer.run(convert_file)