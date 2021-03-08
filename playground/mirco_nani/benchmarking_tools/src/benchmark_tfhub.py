from benchmarking_tools.model.tf_hub_models import *
from benchmarking_tools.benchmarking import benchmark
import sys, argparse

prediction_models=[
 talkheads_ggelu_bert_en_large(),
 bert_en_uncased_L12_H768_A12(),
 small_bert_en_uncased_L4_H512_A8(),
 bert_en_uncased_L2_H128_A2(),
 bert_en_uncased_L24_H1024_A16(),
 bert_en_cased_L12_H768_A12(),
 bert_en_uncased_L2_H512_A8(),
 bert_en_uncased_L4_H128_A2(),
 bert_en_uncased_L2_H768_A12(),
 bert_en_uncased_L2_H256_A4(),
 lambert_en_uncased_L24_H1024_A16(),
 small_bert_en_uncased_L12_H768_A12(),
 bert_en_uncased_L4_H256_A4(),
 bert_en_uncased_L4_H768_A12(),
 bert_en_uncased_L8_H512_A8(),
 bert_en_cased_L24_H1024_A16(),
 bert_en_wwm_cased_L24_H1024_A16(),
 bert_en_uncased_L8_H256_A4(),
 bert_en_uncased_L6_H256_A4(),
 bert_en_uncased_L12_H512_A8(),
 bert_en_uncased_L8_H128_A2(),
 bert_en_wwm_uncased_L24_H1024_A16(),
 bert_en_uncased_L12_H256_A4(),
 talkheads_ggelu_bert_en_base(),
 bert_en_uncased_L8_H768_A12(),
 bert_en_uncased_L6_H768_A12(),
 bert_en_uncased_L6_H512_A8(),
 bert_en_uncased_L6_H128_A2(),
 bert_en_uncased_L12_H128_A2(),
 bert_en_uncased_L10_H768_A12(),
 bert_en_uncased_L10_H512_A8(),
 bert_en_uncased_L10_H256_A4(),
 bert_en_uncased_L10_H128_A2(),
 bert_wiki_books(),
 bert_wiki_books_stt2(),
 bert_wiki_books_squad2(),
 bert_wiki_books_qqp(),
 bert_wiki_books_qnli(),
 bert_wiki_books_mnli(),
 WikiWords250(),
 WikiWords250WithNormalization(),
 WikiWords500(),
 WikiWords500WithNormalization(),
 NnlmEnDim128(),
 NnlmEnDim128WithNormalization(),
 NnlmEnDim50(),
 NnlmEnDim50WithNormalization(),
 UniversalSentenceEncoderCmlmEnBase(),
 UniversalSentenceEncoderCmlmMultilingualBaseBr(),
 UniversalSentenceEncoderCmlmMultilingualBase(),
 UniversalSentenceEncoderCmlm(),
 UniversalSentenceEncoder(),
 UniversalSentenceEncoderMultilingual(),
 UniversalSentenceEncoderLarge(),
 UniversalSentenceEncoderMultilingualLarge(),
 GnewsSwivel20dim()
 ]

sentences=[
"Give me a bright guitar",
"I'd like a sharp cello",
"give me a dry acoustic guitar",
"give me a metallic harp",
"give me a dirty organ",
"give me a hollow piano",
"give me a sharp trumpet",
"give me a cold triangle",
"give me dark drums",
"give me a soft french horn",
"give me a dull clarinet",
"give me a smooth operator",
"Give me a simple square bass",
"Give me an orchestral string",
"Give me an analog pad",
"Give me a simple sine bass",
"Give me a chord preset",
"Get me a 909 closed hi-hat",
"Get me an 808 open hi-hat",
"Give me a round bass",
"Give me a sharp synth",
"Give me a warm pad",
"Give me a wide stereo pad",
"Give me a mono, warm, round synth bass",
"Make me a soft flute that sounds like a chirping bird ",
"Give me a dark brassy sound",
"Can you give me a wailing guitar?",
"Get me a scratchy violin",
"Give me a Star Wars laser beam sound",
"Can you combine a low piano sound with a roaring lion?",
"Get me something like a compact bleep",
"Give me a funky guitar"
 ]


def run(results_destination):
  df = benchmark(prediction_models, sentences)
  df.to_csv(results_destination, index=False)

  
def parse_cli(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--results_destination', type=str, required=True, help="csv file to store benchmark results")
  
  return parser.parse_known_args(argv)
  
if __name__ == "__main__":
  known_args, unknown_args = parse_cli(sys.argv[1:])
  run(**vars(known_args))