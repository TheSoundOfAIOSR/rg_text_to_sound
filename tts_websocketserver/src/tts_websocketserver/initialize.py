from tts_websocketserver.tts_pipeline import get_pipeline
import requests
import logging
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    params = { 'id' : id, 'confirm' : 1 }
    response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 

def download_model():
    drive_id="1F23n09CzsUuMMtEfgMNfAmqyR8_8U04L"
    dest=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..',
                      'assets/ner_model/transformer/model/pytorch_model.bin')
    ensure_dir(os.path.dirname(dest))
    logging.info("DOWNLOADING MODEL (this may take a while)")
    print("DOWNLOADING MODEL (this may take a while)")
    download_file_from_google_drive(drive_id, dest)

if __name__ == "__main__":
    download_model()
    get_pipeline().build()
