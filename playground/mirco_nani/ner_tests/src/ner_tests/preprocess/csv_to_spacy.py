import typer
import srsly
from pathlib import Path
from spacy.util import get_words_and_spaces
from spacy.tokens import Doc, DocBin
import spacy
from tqdm import tqdm
from io import StringIO
import pandas as pd


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


def convert_file(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Argument(..., dir_okay=False),
):
    nlp = spacy.blank("en")
    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])

    header=True
    with open(input_path, "r") as in_f, open(output_path, "w") as out_f:
        for line in tqdm(in_f):
            if header:
                header=False
                continue
            sentence, tokens = pd.read_csv(StringIO(line), header=None, usecols=[0,1]).values[0]
            tokens = eval(tokens)
            dict_line = line_to_dict(sentence, tokens)
            eg = dict_line

            if eg["answer"] != "accept":
                continue
            tokens = [token["text"] for token in eg["tokens"]]
            words, spaces = get_words_and_spaces(tokens, eg["text"])
            doc = Doc(nlp.vocab, words=words, spaces=spaces)
            doc.ents = [
                doc.char_span(s["start"], s["end"], label=s["label"])
                for s in eg.get("spans", [])
            ]
            doc_bin.add(doc)
        doc_bin.to_disk(output_path)
        print(f"Processed {len(doc_bin)} documents: {output_path}")


if __name__ == "__main__":
    typer.run(convert_file)