import os, sys
import collections

assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','assets')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def filename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)