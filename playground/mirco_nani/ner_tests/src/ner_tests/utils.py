import os, sys

assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','assets')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def filename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]



