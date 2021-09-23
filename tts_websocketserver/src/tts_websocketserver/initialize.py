from tts_websocketserver.tts_pipeline import get_pipeline
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_model():
    import urllib.request

    url="https://drive.google.com/file/d/1F23n09CzsUuMMtEfgMNfAmqyR8_8U04L/view?usp=sharing"
    dest=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..',
                      'assets/ner_model/transformer/model/pytorch_model.bin')
    ensure_dir(os.path.dirname(dest))

    g = urllib.request.urlopen(url)
    with open(dest, 'b+w') as f:
        f.write(g.read())

if __name__ == "__main__":
    download_model()
    get_pipeline().build()
